"""Opt-in real-G1 checks of the existing train entry and saved model reload.

Run the train and reload stages in separate pytest processes on the research
GPU. These observers do not replace the runner's scheduling, updates or save.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner


def _params(module):
  return [p.detach().clone() for p in module.parameters()]


def _delta(module, before):
  return max(
    float((p - old).detach().abs().max())
    for p, old in zip(module.parameters(), before, strict=True)
  )


@pytest.mark.skipif(
  os.environ.get("SUPERMIMIC_PHY_PIPELINE_STAGE") != "train",
  reason="opt-in GPU pipeline train stage",
)
def test_original_train_entry_collects_updates_and_collects_again(monkeypatch):
  import supermimic.scripts.train as entry

  root = Path(os.environ["SUPERMIMIC_PHY_PIPELINE_ROOT"])
  weights = Path(os.environ["SUPERMIMIC_PHY_PIPELINE_CHECKPOINTS"])
  work = root / "train"
  work.mkdir(parents=True)
  weights.mkdir(parents=True)
  (work / "logs").symlink_to(weights, target_is_directory=True)
  (work / "wandb").mkdir()
  monkeypatch.chdir(work)
  monkeypatch.setenv("WANDB_MODE", "offline")
  monkeypatch.setenv("WANDB_DIR", str(work / "wandb"))
  report = {"world": [], "policy": [], "collection_versions": []}
  run_original = SuperMimicRunner.run

  def observed_run(runner):
    assert runner.cfg.force_conditioned
    report["runner_cfg"] = asdict(runner.cfg)
    report["motion_file"] = os.environ["SUPERMIMIC_PHY_GPU_MOTION"]
    world_update, policy_update, step = (
      runner.world_model_learner.update,
      runner.policy_learner.update,
      runner.step,
    )
    expected_actor = None

    def observe_world(data):
      before = _params(runner._dynamics)
      stats = world_update(data)
      assert all(torch.isfinite(v).all() for v in stats.values())
      report["world"].append(
        {
          "loss": float(stats["Loss/state"]),
          "parameter_delta": _delta(runner._dynamics, before),
        }
      )
      print(f"[PHY CHECK] world update {len(report['world'])}", flush=True)
      return stats

    def observe_policy():
      nonlocal expected_actor
      before_wm, before_actor = (
        _params(runner._dynamics),
        _params(runner.policy_learner.actor),
      )
      stats = policy_update()
      assert all(torch.isfinite(v).all() for v in stats.values())
      assert _delta(runner._dynamics, before_wm) == 0
      assert not any(p.requires_grad for p in runner._dynamics.parameters())
      report["policy"].append(
        {
          "actor_delta": _delta(runner.policy_learner.actor, before_actor),
          "grad_norm": float(stats["Policy/grad_norm"]),
          "plan_force_fraction": float(runner.wm_env.last_plan_nonzero_fraction),
        }
      )
      expected_actor = _params(runner.policy_learner.actor)
      print(f"[PHY CHECK] policy update {len(report['policy'])}", flush=True)
      return stats

    def observe_collection():
      if expected_actor is not None:
        assert _delta(runner.policy_learner.actor, expected_actor) == 0
      report["collection_versions"].append(len(report["policy"]))
      return step()

    monkeypatch.setattr(runner.world_model_learner, "update", observe_world)
    monkeypatch.setattr(runner.policy_learner, "update", observe_policy)
    monkeypatch.setattr(runner, "step", observe_collection)
    run_original(runner)
    assert len(report["world"]) >= 5 and len(report["policy"]) >= 5
    assert max(report["collection_versions"]) >= 4
    assert any(r["parameter_delta"] > 0 for r in report["world"])
    assert any(r["actor_delta"] > 0 and r["grad_norm"] > 0 for r in report["policy"])
    report["world_cuda_graph"] = (
      runner.world_model_learner._captured_update._graph is not None
    )
    report["policy_cuda_graph"] = (
      runner.policy_learner._captured_update._graph is not None
    )
    assert report["world_cuda_graph"] and report["policy_cuda_graph"]
    n = runner.replay_buffer.num_transitions
    stored = runner.replay_buffer._storage[:, :n]
    report["force_fraction"] = float(
      (stored["external_wrench_w"].abs().sum(-1) > 0).float().mean()
    )
    assert report["force_fraction"] > 0
    for key in ("state", "next_state", "action", "external_wrench_w"):
      assert torch.isfinite(stored[key]).all()
    paths = list(weights.rglob("world_model.pt"))
    assert len(paths) == 1
    report["model_dir"] = str(paths[0].parent)
    saved = torch.load(paths[0], map_location="cpu", weights_only=True)
    for key, value in runner._dynamics.state_dict().items():
      torch.testing.assert_close(saved["dynamics"][key], value.cpu(), rtol=0, atol=0)
    saved_actor = torch.load(
      paths[0].parent / "actor.pt", map_location="cpu", weights_only=True
    )
    for key, value in runner.policy_learner.actor.state_dict().items():
      torch.testing.assert_close(saved_actor[key], value.cpu(), rtol=0, atol=0)

  monkeypatch.setattr(SuperMimicRunner, "run", observed_run)
  argv = [
    "train.py",
    "SuperMimic-GenCoord-G1",
    "--motion-file",
    os.environ["SUPERMIMIC_PHY_GPU_MOTION"],
    "--num-envs",
    "16",
    "--device",
    "cuda:0",
    "--env.seed",
    "0",
    "--runner.force-conditioned",
    "True",
    "--runner.sampling-mode",
    "uniform",
    "--runner.actor-obs",
    "proprio",
    "--runner.history-length",
    "3",
    "--runner.dynamics-hidden-dims",
    "(64,64)",
    "--runner.actor-hidden-dims",
    "(64,64)",
    "--runner.critic-hidden-dims",
    "(64,64)",
    "--runner.latent-dim",
    "8",
    "--runner.max-iterations",
    "81",
    "--runner.buffer-size",
    "128",
    "--runner.replay-ratio",
    "1",
    "--runner.wm.batch-size",
    "16",
    "--runner.wm.num-batches",
    "1",
    "--runner.policy.batch-size",
    "16",
    "--runner.policy.horizon",
    "4",
    "--runner.log-interval",
    "8",
    "--runner.save-interval",
    "80",
    "--runner.run-name",
    "phy-native-pipeline-smoke",
  ]
  monkeypatch.setattr(sys, "argv", argv)
  entry.main()
  report["source_commit"] = subprocess.check_output(
    ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
    text=True,
  ).strip()
  (root / "training.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
  )


@pytest.mark.skipif(
  os.environ.get("SUPERMIMIC_PHY_PIPELINE_STAGE") != "reload",
  reason="opt-in separate-process reload stage",
)
def test_saved_native_models_reload_and_run_in_simulator():
  from copy import deepcopy

  from mjlab.tasks.registry import load_env_cfg

  from supermimic.scripts.train_utils import create_env, setup_device

  root = Path(os.environ["SUPERMIMIC_PHY_PIPELINE_ROOT"])
  training = json.loads((root / "training.json").read_text())
  config = load_runner_cfg({"runner_cfg": training["runner_cfg"]})
  model_dir = Path(training["model_dir"])
  results = []
  for forced in (True, False):
    cfg = deepcopy(config)
    cfg.disturbance.enabled = forced
    device = setup_device("cuda:0")
    env_cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
    env_cfg.seed = 0
    env_cfg.commands["motion"].motion_file = training["motion_file"]
    env_cfg.commands["motion"].sampling_mode = cfg.sampling_mode
    env = create_env(env_cfg, 16, device, cfg.disturbance)
    try:
      runner = SuperMimicRunner(env, cfg, "SuperMimic-GenCoord-G1", device=device)
      runner._wm.load_bundle(
        torch.load(model_dir / "world_model.pt", map_location=device, weights_only=True)
      )
      runner.policy_learner.actor.load_state_dict(
        torch.load(model_dir / "actor.pt", map_location=device, weights_only=True)
      )
      runner.policy_learner.critic.load_state_dict(
        torch.load(model_dir / "critic.pt", map_location=device, weights_only=True)
      )
      runner._dynamics.eval()
      runner.policy_learner.actor.eval()
      before = {
        key: value.detach().clone()
        for key, value in runner.policy_learner.actor.state_dict().items()
      }
      before_world = {
        key: value.detach().clone()
        for key, value in runner._dynamics.state_dict().items()
      }
      # Inference uses the same collection step, without normalization updates.
      runner._flush_normalization = runner._pending_norm_updates.clear
      original_step = env.step
      errors, forces = [], []

      def check_transition(
        action,
        env=env,
        runner=runner,
        original_step=original_step,
        errors=errors,
        forces=forces,
      ):
        state = env.get_observations()["full_state"].clone()
        ctx = runner._sim_context.read(env.unwrapped).clone()
        times = runner.motion_cmd.time_steps.clone()
        output = original_step(action)
        obs, reward, _, extras = output
        wrench = extras["external_disturbance"]["external_wrench_w"]
        aux, rep = runner._dynamics(
          runner._representation.represent(state),
          action,
          runner._representation.extract_aux_state(state),
          ctx,
          wrench,
        )
        assert torch.isfinite(action).all() and torch.isfinite(reward).all()
        assert torch.isfinite(aux).all() and torch.isfinite(rep).all()
        valid = runner.motion_cmd.time_steps == times + 1
        if valid.any():
          errors.append(
            float(
              (rep[valid] - runner._representation.represent(obs["full_state"][valid]))
              .square()
              .mean()
            )
          )
        forces.append(float((wrench.abs().sum(-1) > 0).float().mean()))
        return output

      env.step = check_transition
      env.reset()
      runner._seed_sim_history()
      for _ in range(32):
        runner.step()
      assert errors
      for key, value in runner.policy_learner.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
      for key, value in runner._dynamics.state_dict().items():
        torch.testing.assert_close(value, before_world[key], rtol=0, atol=0)
      fraction = sum(forces) / len(forces)
      assert (fraction > 0) if forced else (fraction == 0)
      results.append(
        {
          "forced": forced,
          "steps": 32,
          "force_fraction": fraction,
          "rep_state_mse": sum(errors) / len(errors),
          "actor_unchanged": True,
        }
      )
    finally:
      env.close()
  (root / "reload.json").write_text(
    json.dumps(results, indent=2, allow_nan=False) + "\n"
  )
