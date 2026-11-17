"""Bounded online force-WM/actor-critic training and simulator validation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.models import build_mlp
from supermimic.phy.disturbances import ExternalWrenchCfg
from supermimic.phy.dynamics import ForceDynamics, ForceWorldModelLearner
from supermimic.phy.replay import ForceReplayBuffer
from supermimic.phy.rollout import ForcePolicyLearner, ForceWorldModelEnv
from supermimic.replay_buffer import FieldSpec
from supermimic.runners.supermimic_runner import (
  SuperMimicRunner,
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)


@dataclass
class LoopConfig:
  """Small verification budgets; formal experiment scheduling is out of scope."""

  seed: int = 0
  num_envs: int = 16
  buffer_size: int = 128
  warmup_steps: int = 32
  warmup_updates: int = 4
  cycles: int = 6
  collect_steps: int = 8
  batch_size: int = 16
  wm_horizon: int = 8
  policy_horizon: int = 4
  history_length: int = 3
  actor_obs: str = "proprio"
  hidden_dim: int = 64
  save_interval: int = 2
  disturbance: ExternalWrenchCfg = field(
    default_factory=lambda: ExternalWrenchCfg(
      enabled=True,
      cooldown_steps=(2, 6),
      duration_steps=(2, 4),
      zero_probability=0.25,
      seed=10000,
    )
  )

  def validate(self):
    self.disturbance.validate()
    for name in (
      "num_envs",
      "buffer_size",
      "warmup_steps",
      "cycles",
      "collect_steps",
      "batch_size",
      "wm_horizon",
      "policy_horizon",
      "history_length",
      "hidden_dim",
      "save_interval",
    ):
      if getattr(self, name) < 1:
        raise ValueError(f"positive {name} required")
    if self.warmup_updates < 0 or self.wm_horizon < 2:
      raise ValueError("warmup_updates >= 0 and wm_horizon >= 2 required")
    if self.warmup_steps < max(self.wm_horizon, self.history_length):
      raise ValueError("warmup_steps must fill WM and sensor-history windows")
    if self.buffer_size < max(self.warmup_steps, self.wm_horizon, self.history_length):
      raise ValueError("buffer_size too small")
    if self.history_length > 1 and self.actor_obs != "proprio":
      raise ValueError("history_length > 1 requires proprio actor observations")
    if self.actor_obs not in {"proprio", "rep_state"}:
      raise ValueError("unsupported actor_obs")
    if self.batch_size * self.policy_horizon < 4:
      raise ValueError("critic needs at least four rollout samples")
    if (
      self.num_envs > 64
      or self.batch_size > 128
      or self.cycles > 32
      or self.collect_steps > 64
      or self.warmup_steps > 256
      or self.policy_horizon > 16
      or self.wm_horizon > 16
      or self.hidden_dim > 256
    ):
      raise ValueError("this entry point is bounded to small-scale verification")

  def runner_cfg(self):
    hidden = (self.hidden_dim, self.hidden_dim)
    return SuperMimicRunnerCfg(
      actor_obs=self.actor_obs,
      history_length=self.history_length,
      buffer_size=self.buffer_size,
      dynamics_hidden_dims=hidden,
      actor_hidden_dims=hidden,
      critic_hidden_dims=hidden,
      latent_dim=8,
      sampling_mode="uniform",
      wm=WorldModelLearnerCfg(
        batch_size=self.batch_size, num_batches=1, horizon=self.wm_horizon
      ),
      policy=ActorCriticPolicyLearnerCfg(
        batch_size=self.batch_size, horizon=self.policy_horizon
      ),
    )


def source_commit():
  return subprocess.check_output(
    ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
    text=True,
  ).strip()


def file_sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
  temp = path.with_suffix(path.suffix + ".tmp")
  temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  os.replace(temp, path)


def parameter_snapshot(module):
  return [p.detach().clone() for p in module.parameters()]


def parameter_delta(module, before):
  return max(
    float((p.detach() - old).abs().max())
    for p, old in zip(module.parameters(), before, strict=True)
  )


def finite_stats(stats):
  result = {}
  for key, value in stats.items():
    value = value.detach()
    if not torch.isfinite(value).all():
      raise ValueError(f"nonfinite training metric {key}")
    result[key] = float(value)
  return result


def same_checkpoint_state(actual, expected):
  if torch.is_tensor(actual):
    return torch.is_tensor(expected) and torch.equal(actual.cpu(), expected.cpu())
  if isinstance(actual, dict):
    return actual.keys() == expected.keys() and all(
      same_checkpoint_state(v, expected[k]) for k, v in actual.items()
    )
  if isinstance(actual, (tuple, list)):
    return len(actual) == len(expected) and all(
      same_checkpoint_state(a, b) for a, b in zip(actual, expected, strict=True)
    )
  return actual == expected


class ClosedLoopRunner(SuperMimicRunner):
  def __init__(self, env, config, device, motion_file):
    config.validate()
    self.loop_cfg = config
    self.motion_file = Path(motion_file)
    self.counters = dict(
      cycles=0,
      world_update_calls=0,
      policy_update_calls=0,
      sim_steps=0,
      sim_transitions=0,
    )
    self.resume_info = None
    self.code_commit = source_commit()
    super().__init__(env, config.runner_cfg(), "SuperMimic-GenCoord-G1", device=device)

  def _init_common(self):
    wm = build_world_model(self.cfg, self.env, self.device)
    base = wm.dynamics
    wm.dynamics = ForceDynamics(
      wm.representation,
      wm.num_actions,
      wm.sim_context.output_dim,
      base.action_scale,
      base.action_offset,
      self.cfg.dynamics_hidden_dims,
      num_bodies=len(self.loop_cfg.disturbance.body_names),
      use_wrench=True,
    ).to(self.device)
    self._wm = wm
    self._dynamics, self._representation, self._sim_context = (
      wm.dynamics,
      wm.representation,
      wm.sim_context,
    )
    self._num_actions, self._num_goals = wm.num_actions, wm.num_goals
    self.action_manager = self.env.unwrapped.action_manager
    self.motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    self.motion = self.motion_cmd.motion
    self.replay_buffer = ForceReplayBuffer(
      self._buffer_schema(), self.cfg.buffer_size, self.env.num_envs, self.device
    )
    self.world_model_learner = ForceWorldModelLearner(
      self._dynamics, self._representation, self.cfg.wm, device=self.device
    )
    self._pending_norm_updates, self._sim_history = [], None

  def _buffer_schema(self):
    return super()._buffer_schema() | {
      "next_state": FieldSpec(self._representation.full_state_dim),
      "external_wrench_w": FieldSpec(len(self.loop_cfg.disturbance.body_names) * 6),
      "disturbance_state": FieldSpec(2, torch.long),
      "transition_valid": FieldSpec(dtype=torch.bool),
      "actor_version": FieldSpec(dtype=torch.long),
    }

  def _init_policy(self):
    actor = build_actor(self.cfg, self._wm, self.device)
    self._active_task_specs = [
      ts for ts in actor.task_specs if self.cfg.slot_mix.get(ts.name, 0) > 0
    ]
    _, critic_dim = self._policy_input_dims()
    critic = build_mlp(
      critic_dim + self._dynamics.wrench_scale.numel() + 2,
      self.cfg.critic_hidden_dims,
      output_dim=1,
    ).to(self.device)
    self.wm_env = ForceWorldModelEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=self.cfg.policy.batch_size,
      replay_buffer=self.replay_buffer,
      task_specs=self._active_task_specs,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
      disturbance=self.loop_cfg.disturbance,
      horizon=self.cfg.policy.horizon,
      plan_seed=self.loop_cfg.disturbance.seed + 1,
    )
    if torch.device(self.device).type == "cuda":
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)
    self.policy_learner = ForcePolicyLearner(
      actor, critic, self.wm_env, self.cfg.policy, device=self.device
    )

  def reset_collection(self):
    self.env.reset()
    self._seed_sim_history()

  def collect(self, steps, *, training=True):
    """Execute the CURRENT actor, store true pre-reset labels, then reset rows."""
    actor = self.policy_learner.actor
    actor.eval()
    self._dynamics.eval()
    rewards, forced, valid_predictions = [], [], []
    resets = invalid = 0
    with torch.inference_mode():
      for _ in range(steps):
        obs = self.env.get_observations()
        state, sensor = obs["full_state"].clone(), obs["actor"].clone()
        times = self.motion_cmd.time_steps.clone()
        ctx = self._sim_context.read(self.env.unwrapped).clone()
        rep, aux = (
          self._representation.represent(state),
          self._representation.extract_aux_state(state),
        )
        actor_td = self._build_sim_obs_td(rep, aux, sensor, times, ctx)
        actions = actor(actor_td)
        previous = self.action_manager.action.clone()
        next_obs, reward, done, extras = self.env.step(actions)
        target = next_obs["full_state"].clone()
        valid = (self.motion_cmd.time_steps == times + 1).clone()
        timeouts = extras["time_outs"].bool().clone()
        disturbance = extras["external_disturbance"]
        fields = {
          "state": state,
          "next_state": target,
          "action": actions.clone(),
          "prev_action": previous,
          "motion_time": times,
          "sim_ctx": ctx,
          "terminated": done.bool() & ~timeouts,
          "truncated": timeouts,
          "transition_valid": valid,
          **disturbance,
          "actor_version": torch.full(
            (self.env.num_envs,),
            self.counters["policy_update_calls"],
            device=self.device,
            dtype=torch.long,
          ),
        }
        if self._stores_actor_obs:
          fields["actor_obs"] = sensor
        for key, value in fields.items():
          if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite collected {key}")
        if not torch.isfinite(reward).all():
          raise ValueError("nonfinite simulator reward")
        if training:
          self.replay_buffer.insert(TensorDict(fields, batch_size=[self.env.num_envs]))
          self._pending_norm_updates.append(
            (state.clone(), actions.clone(), actor_td.detach().clone())
          )
        else:
          pred_aux, pred_rep = self._dynamics(
            rep, actions, aux, ctx, disturbance["external_wrench_w"]
          )
          if not torch.isfinite(pred_aux).all() or not torch.isfinite(pred_rep).all():
            raise ValueError("nonfinite reloaded world-model inference")
          if valid.any():
            valid_predictions.append(
              (pred_rep[valid] - self._representation.represent(target[valid]))
              .square()
              .mean()
            )
        boundary = done.bool() | ~valid
        resets += int(boundary.sum())
        invalid += int((~valid).sum())
        if boundary.any():
          self.env.unwrapped.reset(env_ids=boundary.nonzero().flatten())
        fresh_obs = self.env.get_observations()
        self._advance_sim_history(fresh_obs["actor"], boundary)
        rewards.append(reward.mean())
        forced.append(
          (disturbance["external_wrench_w"].abs().sum(-1) > 0).float().mean()
        )
    if training:
      self._flush_normalization()
    self.counters["sim_steps"] += steps
    self.counters["sim_transitions"] += steps * self.env.num_envs
    result = {
      "actor_version": self.counters["policy_update_calls"],
      "steps": steps,
      "mean_reward": float(torch.stack(rewards).mean()),
      "nonzero_force_fraction": float(torch.stack(forced).mean()),
      "resets": resets,
      "invalid_transitions": invalid,
    }
    if valid_predictions:
      result["one_step_rep_state_mse"] = float(torch.stack(valid_predictions).mean())
    elif not training:
      raise RuntimeError("evaluation produced no valid next-state predictions")
    return result

  def save_replay(self, path):
    if path.exists():
      raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "data": self.replay_buffer.snapshot(),
      "metadata": {
        "schema_version": 1,
        "source_commit": self.code_commit,
        "scope": "bounded online closed-loop verification replay",
        "seed": self.loop_cfg.seed,
        "counters": self.counters.copy(),
        "num_envs": self.env.num_envs,
        "steps": self.replay_buffer.num_transitions,
        "full_state_dim": self._representation.full_state_dim,
        "num_actions": self._num_actions,
        "sim_ctx_dim": self._sim_context.output_dim,
        "dt": self.env.unwrapped.step_dt,
        "decimation": self.env.cfg.decimation,
        "representation": "gencoord",
        "actor_obs": self.cfg.actor_obs,
        "history_length": self.cfg.history_length,
        "motion_file": str(self.motion_file),
        "motion_sha256": file_sha256(self.motion_file),
        "disturbance": self.env.unwrapped.external_disturbance.metadata(
          self.env.unwrapped.step_dt
        ),
      },
      "action_scale": self._dynamics.action_scale.cpu(),
      "action_offset": self._dynamics.action_offset.cpu(),
      "default_joint_pos": self.env.unwrapped.scene["robot"]
      .data.default_joint_pos[0]
      .cpu(),
      "sim_context": self._sim_context.state_dict(),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)

  def update_world_model(self):
    data = [self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)]
    before = parameter_snapshot(self._dynamics)
    logs = finite_stats(self.world_model_learner.update(data))
    logs["parameter_delta"] = parameter_delta(self._dynamics, before)
    self.counters["world_update_calls"] += 1
    return logs

  def update_policy(self):
    before_wm = parameter_snapshot(self._dynamics)
    before_actor = parameter_snapshot(self.policy_learner.actor)
    logs = finite_stats(self.policy_learner.update())
    delta_wm = parameter_delta(self._dynamics, before_wm)
    if delta_wm != 0 or any(p.requires_grad for p in self._dynamics.parameters()):
      raise RuntimeError("policy update must keep world-model parameters frozen")
    logs.update(
      actor_parameter_delta=parameter_delta(self.policy_learner.actor, before_actor),
      frozen_world_parameter_delta=delta_wm,
      plan_nonzero_force_fraction=float(self.wm_env.last_plan_nonzero_fraction),
    )
    self.counters["policy_update_calls"] += 1
    return logs

  def modules_to_save(self):
    return {
      "dynamics": self._dynamics,
      "sim_context": self._sim_context,
      "actor": self.policy_learner.actor,
      "critic": self.policy_learner.critic,
    }

  def optimizers_to_save(self):
    return {
      "world": self.world_model_learner.optimizers,
      "actor": self.policy_learner.actor_optimizers,
      "critic": self.policy_learner.critic_optimizers,
    }

  def save(self, path):
    modules = {
      key: module.state_dict() for key, module in self.modules_to_save().items()
    }
    for name, state in modules.items():
      for key, value in state.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
          raise ValueError(f"nonfinite checkpoint {name}/{key}")
    numpy_rng = np.random.get_state()
    payload = {
      "format": "supermimic-phy-loop-v1",
      "source_commit": self.code_commit,
      "config": asdict(self.loop_cfg),
      "counters": self.counters.copy(),
      "motion_file": str(self.motion_file.resolve()),
      "motion_sha256": file_sha256(self.motion_file),
      "modules": modules,
      "optimizers": {
        key: [opt.state_dict() for opt in opts]
        for key, opts in self.optimizers_to_save().items()
      },
      "rng": {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(self.device)
        if torch.device(self.device).type == "cuda"
        else None,
        "python": random.getstate(),
        "numpy": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
        "plan": self.wm_env.plan_sampler.generator.get_state(),
        "physical_force": self.env.unwrapped.external_disturbance.generator.get_state(),
      },
      "resume_semantics": "restore models/optimizers/counters/RNG; reset simulator and refill replay, not bitwise physics continuation",
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)

  def restore(self, payload, *, training):
    if payload["format"] != "supermimic-phy-loop-v1":
      raise ValueError("not a closed-loop checkpoint")
    if training and payload["motion_sha256"] != file_sha256(self.motion_file):
      raise ValueError("resume requires the same motion file contents")
    for key, module in self.modules_to_save().items():
      expected = payload["modules"][key]
      module.load_state_dict(expected)
      for name, tensor in module.state_dict().items():
        if torch.is_tensor(tensor) and not torch.equal(
          tensor.cpu(), expected[name].cpu()
        ):
          raise RuntimeError(f"checkpoint restore mismatch: {key}/{name}")
    if training:
      for key, opts in self.optimizers_to_save().items():
        for opt, state in zip(opts, payload["optimizers"][key], strict=True):
          opt.load_state_dict(state)
          # load_state_dict does not place a tensor learning rate on the GPU.
          for group in opt.param_groups:
            if torch.is_tensor(group["lr"]):
              group["lr"] = group["lr"].to(self.device)
          if not same_checkpoint_state(opt.state_dict(), state):
            raise RuntimeError(f"optimizer restore mismatch: {key}")
      rng = payload["rng"]
      torch.set_rng_state(rng["torch"].cpu())
      if rng["cuda"] is not None and torch.device(self.device).type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"].cpu(), self.device)
      random.setstate(rng["python"])
      nr = rng["numpy"]
      np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), *nr[2:]))
      self.wm_env.plan_sampler.generator.set_state(rng["plan"].cpu())
      self.env.unwrapped.external_disturbance.generator.set_state(
        rng["physical_force"].cpu()
      )
    self.counters = payload["counters"].copy()
    self.resume_info = {
      "checkpoint_source_commit": payload["source_commit"],
      "restored_counters": self.counters.copy(),
      "models_equal_after_restore": True,
      "optimizers_restored": training,
      "optimizers_equal_after_restore": training,
      "fresh_simulator_and_replay": True,
    }

  def train_closed_loop(self, output_dir):
    self.reset_collection()
    report = {
      "scope": "bounded closed-loop functional verification; no formal experiment",
      "source_commit": self.code_commit,
      "config": asdict(self.loop_cfg),
      "resume": self.resume_info,
      "cycles": [],
      "warmup_collection": self.collect(self.loop_cfg.warmup_steps),
    }
    if self.resume_info is None:
      report["warmup_world_updates"] = [
        self.update_world_model() for _ in range(self.loop_cfg.warmup_updates)
      ]
    for _ in range(self.loop_cfg.cycles):
      collection = self.collect(self.loop_cfg.collect_steps)
      row = {
        "collection": collection,
        "world_model": self.update_world_model(),
        "policy": self.update_policy(),
      }
      self.counters["cycles"] += 1
      row["cycle"] = self.counters["cycles"]
      report["cycles"].append(row)
      print(json.dumps(row, allow_nan=False), flush=True)
      if self.counters["cycles"] % self.loop_cfg.save_interval == 0:
        self.save(output_dir / f"checkpoint-cycle-{self.counters['cycles']:06d}.pt")
    report.update(
      counters=self.counters.copy(),
      world_cuda_graph_captured=self.world_model_learner._captured_update._graph
      is not None,
      policy_cuda_graph_captured=self.policy_learner._captured_update._graph
      is not None,
    )
    if not any(r["world_model"]["parameter_delta"] > 0 for r in report["cycles"]):
      raise RuntimeError("world-model parameters did not update")
    if not any(r["policy"]["actor_parameter_delta"] > 0 for r in report["cycles"]):
      raise RuntimeError("actor parameters did not update")
    if (
      self.loop_cfg.cycles > 1
      and report["cycles"][-1]["collection"]["actor_version"]
      <= report["cycles"][0]["collection"]["actor_version"]
    ):
      raise RuntimeError("updated actor was not used for subsequent collection")
    self.save(output_dir / "checkpoint.pt")
    write_json(output_dir / "training.json", report)
    return report

  def evaluate_in_simulator(self, output_dir, steps):
    before = {
      key: {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
      }
      for key, module in self.modules_to_save().items()
    }
    self.reset_collection()
    result = self.collect(steps, training=False)
    for key, module in self.modules_to_save().items():
      if any(
        not torch.equal(value, before[key][name])
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
      ):
        raise RuntimeError("simulator evaluation must not modify learned state")
    report = {
      "scope": "reloaded actor/world-model simulator smoke; not a robustness benchmark",
      "source_commit": self.code_commit,
      "restore": self.resume_info,
      "disturbance": asdict(self.loop_cfg.disturbance),
      "results": result,
      "learned_state_unchanged": True,
    }
    write_json(output_dir / "evaluation.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)
    return report
