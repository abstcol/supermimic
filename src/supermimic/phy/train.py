"""Opt-in small-scale train/resume/evaluate entry point for SuperMimic-Phy."""

from __future__ import annotations

import argparse
import traceback
from dataclasses import replace
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.phy.disturbances import ExternalWrenchCfg
from supermimic.phy.environment import ForceTrackingEnv
from supermimic.phy.loop import ClosedLoopRunner, LoopConfig, source_commit, write_json
from supermimic.scripts.train_utils import setup_device


def run(args):
  if args.output_dir.exists():
    raise FileExistsError(args.output_dir)
  if args.resume and args.evaluate:
    raise ValueError("choose resume or evaluate")
  if args.replay_output and args.evaluate:
    raise ValueError("replay snapshots are training-only")
  if args.replay_output and args.replay_output.exists():
    raise FileExistsError(args.replay_output)
  checkpoint = args.resume or args.evaluate
  payload = (
    torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint
    else None
  )
  if payload:
    if payload.get("format") != "supermimic-phy-loop-v1":
      raise ValueError(
        "requires a closed-loop checkpoint, not a phase1 WM-only snapshot"
      )
    values = payload["config"].copy()
    values["disturbance"] = ExternalWrenchCfg(**values["disturbance"])
    config = LoopConfig(**values)
  else:
    config = LoopConfig()
  for name in (
    "cycles",
    "num_envs",
    "warmup_steps",
    "collect_steps",
    "policy_horizon",
    "history_length",
    "hidden_dim",
    "actor_obs",
  ):
    value = getattr(args, name)
    if value is not None:
      if checkpoint and name not in {
        "cycles",
        "num_envs",
        "warmup_steps",
        "collect_steps",
      }:
        raise ValueError(f"cannot override checkpoint architecture/config: {name}")
      setattr(config, name, value)
  if args.force_mode == "zero":
    if not args.evaluate:
      raise ValueError("force-mode override is evaluation-only")
    config.disturbance = replace(config.disturbance, enabled=False)
  config.validate()
  if not 1 <= args.evaluation_steps <= 256:
    raise ValueError("evaluation_steps must be in [1,256]")
  motion = args.motion_file or (Path(payload["motion_file"]) if payload else None)
  if motion is None or not motion.is_file():
    raise ValueError("a local motion file is required")
  device = setup_device(args.device)
  env_cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  env_cfg.scene.num_envs = config.num_envs
  env_cfg.seed = config.seed
  env_cfg.auto_reset = False
  env_cfg.commands["motion"].motion_file = str(motion)
  env_cfg.commands["motion"].sampling_mode = "uniform"
  args.output_dir.mkdir(parents=True)
  write_json(
    args.output_dir / "status.json",
    {"state": "running", "source_commit": source_commit()},
  )
  env = None
  try:
    env = RslRlVecEnvWrapper(ForceTrackingEnv(env_cfg, device, config.disturbance))
    runner = ClosedLoopRunner(env, config, device, motion)
    if payload:
      runner.restore(payload, training=not bool(args.evaluate))
    if args.evaluate:
      runner.evaluate_in_simulator(args.output_dir, args.evaluation_steps)
    else:
      report = runner.train_closed_loop(args.output_dir)
      if args.replay_output:
        runner.save_replay(args.replay_output)
        report["replay_output"] = str(args.replay_output)
        write_json(args.output_dir / "training.json", report)
    write_json(
      args.output_dir / "status.json",
      {"state": "completed", "source_commit": source_commit()},
    )
  except Exception as error:
    write_json(
      args.output_dir / "status.json",
      {
        "state": "failed",
        "source_commit": source_commit(),
        "error": repr(error),
        "traceback": traceback.format_exc(),
      },
    )
    raise
  finally:
    if env is not None:
      env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--replay-output", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--resume", type=Path)
  parser.add_argument("--evaluate", type=Path)
  parser.add_argument("--force-mode", choices=("trained", "zero"), default="trained")
  parser.add_argument("--evaluation-steps", type=int, default=64)
  for name in (
    "cycles",
    "num-envs",
    "warmup-steps",
    "collect-steps",
    "policy-horizon",
    "history-length",
    "hidden-dim",
  ):
    parser.add_argument(f"--{name}", type=int)
  parser.add_argument("--actor-obs", choices=("proprio", "rep_state"))
  run(parser.parse_args())


if __name__ == "__main__":
  main()
