"""Matched-budget force-input versus blind WM pilot on one fixed dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.phy.data import sample_windows, valid_window_starts
from supermimic.phy.dynamics import ForceDynamics, ForceWorldModelLearner
from supermimic.representations import create_representation
from supermimic.scripts.train_utils import setup_device


def make_learner(bundle, args, use_wrench):
  meta = bundle["metadata"]
  torch.manual_seed(args.seed)
  rep = create_representation(
    "gencoord",
    meta["full_state_dim"],
    meta["num_actions"],
    meta["dt"],
    meta["decimation"],
    bundle["default_joint_pos"].to(args.device),
    actor_obs_mode="rep_state",
  ).to(args.device)
  dynamics = ForceDynamics(
    rep,
    meta["num_actions"],
    meta["sim_ctx_dim"],
    bundle["action_scale"],
    bundle["action_offset"],
    tuple(args.hidden_dims),
    num_bodies=len(meta["disturbance"]["body_names"]),
    use_wrench=use_wrench,
  ).to(args.device)
  cfg = WorldModelLearnerCfg(
    batch_size=args.batch_size, num_batches=1, horizon=args.horizon
  )
  return ForceWorldModelLearner(dynamics, rep, cfg, device=args.device)


@torch.no_grad()
def evaluate(learner, td):
  learner.dynamics.eval()
  rep = learner.representation
  predicted, aux = learner._rollout(
    rep.represent(td["state"][:, 0]),
    td["action"],
    rep.extract_aux_state(td["state"][:, 0]),
    td["sim_ctx"],
    td["external_wrench_w"],
  )
  target = td["next_state"][:, :-1]
  loss, mse = learner._compute_losses(
    predicted, aux, rep.represent(target), rep.extract_aux_state(target)
  )
  if not torch.isfinite(loss) or not torch.isfinite(mse).all():
    raise ValueError("nonfinite held-out predictions")
  return {
    "weighted_state_loss": float(loss),
    **{
      name: float(value)
      for name, value in zip(learner.loss_group_weights, mse, strict=True)
    },
  }


def fit(args):
  if args.iterations < 1 or args.horizon < 2:
    raise ValueError("positive iterations and horizon >= 2 required")
  if args.output_dir.exists():
    raise FileExistsError(args.output_dir)
  args.device = setup_device(args.device)
  bundle = torch.load(args.dataset, map_location="cpu", weights_only=True)
  raw = bundle["data"]
  n, steps = raw["state"].shape[:2]
  split = int(n * 0.75)
  if split < 1 or split >= n:
    raise ValueError("dataset needs separate train/validation environments")
  train_envs, val_envs = torch.arange(split), torch.arange(split, n)
  starts = valid_window_starts(
    raw["terminated"], raw["truncated"], train_envs, args.horizon
  )
  data = TensorDict(raw, batch_size=[n, steps]).to(args.device)
  learners = {
    "conditioned": make_learner(bundle, args, True),
    "blind": make_learner(bundle, args, False),
  }
  # Identical weights and all normalization samples. No validation leakage.
  for learner in learners.values():
    learner.update_normalization(
      data["state"][:split].flatten(0, 1), data["action"][:split].flatten(0, 1)
    )
  args.output_dir.mkdir(parents=True)
  generator = torch.Generator().manual_seed(args.seed)
  curve = []
  for step in range(args.iterations):
    td = sample_windows(data, starts, args.horizon, args.batch_size, generator)
    row = {"iteration": step}
    for name, learner in learners.items():
      logs = learner.update([td])
      if step % 25 == 0 or step == args.iterations - 1:
        row[name] = float(logs["Loss/state"])
        if not torch.isfinite(logs["Loss/state"]):
          raise ValueError(f"nonfinite {name} training loss")
    if len(row) > 1:
      curve.append(row)
      print(json.dumps(row), flush=True)
  report = {
    "scope": "phase1 small functional pilot; no robustness or paper reproduction claim",
    "source_commit": subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
    "dataset": str(args.dataset),
    "seed": args.seed,
    "split": {
      "train_env_ids": train_envs.tolist(),
      "validation_env_ids": val_envs.tolist(),
    },
    "split_scope": "whole environment trajectories; same motion and force distribution",
    "learner_config": asdict(next(iter(learners.values())).cfg),
    "hidden_dims": args.hidden_dims,
    "training_curve": curve,
    "evaluation": {},
  }
  eval_rng = torch.Generator().manual_seed(args.seed + 100000)
  for horizon in (1, 8, 32):
    length = horizon + 1
    if length > steps:
      continue
    valid = valid_window_starts(raw["terminated"], raw["truncated"], val_envs, length)
    eval_td = sample_windows(data, valid, length, args.eval_batch_size, eval_rng)
    forced = eval_td["external_wrench_w"][:, :horizon].abs().sum((-1, -2)) > 0
    groups = {"all": torch.ones_like(forced), "forced": forced, "zero": ~forced}
    metrics = {}
    for group, mask in groups.items():
      count = int(mask.sum())
      metrics[group] = {"windows": count}
      if count:
        selected = eval_td[mask]
        for name, learner in learners.items():
          metrics[group][name] = evaluate(learner, selected)
    # Wrong alignment is a diagnostic, not a third trained model.
    shuffled = eval_td.clone()
    shuffled["external_wrench_w"] = shuffled["external_wrench_w"].roll(1, 0)
    metrics["shuffled_force_conditioned"] = evaluate(learners["conditioned"], shuffled)
    report["evaluation"][str(horizon)] = metrics
  for name, learner in learners.items():
    for value in learner.dynamics.state_dict().values():
      if not torch.isfinite(value).all():
        raise ValueError(f"nonfinite {name} checkpoint")
    torch.save(
      {
        "dynamics": learner.dynamics.state_dict(),
        "metadata": bundle["metadata"],
        "use_wrench": name == "conditioned",
        "hidden_dims": args.hidden_dims,
        "learner_config": asdict(learner.cfg),
        "source_commit": report["source_commit"],
        "dataset_sha256": report["dataset_sha256"],
        "sim_context": bundle["sim_context"],
      },
      args.output_dir / f"{name}.pt",
    )
  (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report["evaluation"], indent=2), flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--dataset", type=Path, required=True)
  p.add_argument("--output-dir", type=Path, required=True)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--iterations", type=int, default=200)
  p.add_argument("--horizon", type=int, default=8)
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--eval-batch-size", type=int, default=64)
  p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])
  fit(p.parse_args())


if __name__ == "__main__":
  main()
