"""Collect true force-labelled transitions with a frozen official actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.phy.disturbances import ExternalWrenchCfg
from supermimic.phy.environment import ForceTrackingEnv
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.scripts.train_utils import setup_device
from supermimic.tasks.task_spec import build_actor_obs_td


def collect(args):
  if args.num_envs < 4 or args.steps < 33:
    raise ValueError("use at least 4 environments and 33 steps for split/evaluation")
  if args.output.exists():
    raise FileExistsError(args.output)
  device = setup_device(args.device)
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.auto_reset = False  # returned next_state must precede a reset
  cfg.commands["motion"].motion_file = str(args.motion_file)
  cfg.commands["motion"].sampling_mode = "uniform"
  disturbance = ExternalWrenchCfg(
    enabled=True,
    body_names=tuple(args.bodies),
    force_range=(-args.force, args.force),
    torque_range=(-args.torque, args.torque),
    seed=args.seed + 10000,
  )
  env = RslRlVecEnvWrapper(ForceTrackingEnv(cfg, device, disturbance))
  try:
    runner_cfg = SuperMimicRunnerCfg(
      actor_obs=args.actor_obs, history_length=args.history_length
    )
    wm = build_world_model(runner_cfg, env, device)
    wm.load_bundle(
      torch.load(
        args.checkpoint_dir / "world_model.pt",
        map_location=device,
        weights_only=True,
      )
    )
    actor = build_actor(runner_cfg, wm, device)
    actor.load_state_dict(
      torch.load(
        args.checkpoint_dir / "actor.pt", map_location=device, weights_only=True
      )
    )
    actor.eval()
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    obs, _ = env.reset()
    history = None
    previous_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    records = []
    with torch.inference_mode():
      for step in range(args.steps):
        full = obs["full_state"].clone()
        rep = wm.representation.represent(full)
        aux = wm.representation.extract_aux_state(full)
        sim_ctx = wm.sim_context.read(env.unwrapped).clone()
        sensor = wm.representation.compute_actor_sensor(
          rep,
          sim_ctx,
          aux_state=aux,
          sim_actor_obs=obs["actor"] if args.actor_obs == "proprio" else None,
          noisy=args.actor_obs == "proprio",
        )
        if history is None:
          history = sensor[:, None].repeat(1, args.history_length, 1)
        else:
          history = torch.cat([history[:, 1:], sensor[:, None]], dim=1)
          history[previous_done] = sensor[previous_done, None]
        goals = {
          ts.name: ts.extract_training_goal(
            motion_cmd.motion, motion_cmd.time_steps, aux
          )
          for ts in actor.task_specs
          if runner_cfg.slot_mix.get(ts.name, 0) > 0
        }
        # Wrench/timer information is deliberately absent from the actor TD.
        actions = actor(build_actor_obs_td(goals, history))
        times = motion_cmd.time_steps.clone()
        next_obs, reward, done, extras = env.step(actions)
        timeouts = extras["time_outs"].bool()
        record = {
          "state": full,
          "next_state": next_obs["full_state"].clone(),
          "action": actions.clone(),
          "sim_ctx": sim_ctx,
          "terminated": done.bool() & ~timeouts,
          "truncated": timeouts.clone(),
          "motion_time": times,
          # MotionCommand may teleport on clip wrap even with auto_reset off.
          "transition_valid": (motion_cmd.time_steps == times + 1).clone(),
          "reward": reward.clone(),
          **extras["external_disturbance"],
        }
        for name, value in record.items():
          if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite collected {name} at step {step}")
        records.append({k: v.cpu() for k, v in record.items()})
        previous_done = done.bool()
        if previous_done.any():
          env.unwrapped.reset(env_ids=previous_done.nonzero().flatten())
        obs = env.get_observations()
        if step % 100 == 0:
          print(f"[PHY] collected {step + 1}/{args.steps}", flush=True)
    data = {k: torch.stack([r[k] for r in records], 1) for k in records[0]}
    robot = env.unwrapped.scene["robot"]
    metadata = {
      "schema_version": 1,
      "source_commit": subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        text=True,
      ).strip(),
      "scope": "phase1 functional pilot; not policy training",
      "task": "SuperMimic-GenCoord-G1",
      "seed": args.seed,
      "steps": args.steps,
      "num_envs": args.num_envs,
      "motion_file": str(args.motion_file),
      "motion_sha256": hashlib.sha256(args.motion_file.read_bytes()).hexdigest(),
      "actor_checkpoint": str(args.checkpoint_dir / "actor.pt"),
      "actor_sha256": hashlib.sha256(
        (args.checkpoint_dir / "actor.pt").read_bytes()
      ).hexdigest(),
      "actor_obs": args.actor_obs,
      "history_length": args.history_length,
      "representation": "gencoord",
      "full_state_dim": wm.representation.full_state_dim,
      "num_actions": env.num_actions,
      "sim_ctx_dim": wm.sim_context.output_dim,
      "dt": env.unwrapped.step_dt,
      "decimation": cfg.decimation,
      "disturbance": env.unwrapped.external_disturbance.metadata(env.unwrapped.step_dt),
      "env_config": asdict(cfg),
      "terminal_next_state": "pre-reset except command teleports, marked transition_valid=False",
      "invalid_command_transitions": int((~data["transition_valid"]).sum()),
      "nonzero_force_fraction": float(
        (data["external_wrench_w"].abs().sum(-1) > 0).float().mean()
      ),
    }
    # cfg contains callables; keep a readable sidecar, not nonportable pickle objects.
    metadata = json.loads(json.dumps(metadata, default=str))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "data": data,
        "metadata": metadata,
        "action_scale": wm.dynamics.action_scale.cpu(),
        "action_offset": wm.dynamics.action_offset.cpu(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu(),
        "sim_context": wm.sim_context.state_dict(),
      },
      args.output,
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
      json.dumps(
        {
          "dataset": str(args.output),
          "nonzero_force_fraction": metadata["nonzero_force_fraction"],
        }
      )
    )
  finally:
    env.close()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--motion-file", type=Path, required=True)
  p.add_argument("--checkpoint-dir", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--num-envs", type=int, default=32)
  p.add_argument("--steps", type=int, default=256)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--bodies", nargs="+", default=["pelvis"])
  p.add_argument("--force", type=float, default=10.0)
  p.add_argument("--torque", type=float, default=0.0)
  p.add_argument("--actor-obs", choices=["rep_state", "proprio"], default="rep_state")
  p.add_argument("--history-length", type=int, default=1)
  collect(p.parse_args())


if __name__ == "__main__":
  main()
