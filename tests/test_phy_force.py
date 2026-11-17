"""Force timing, frame convention, reset boundaries and information ablation."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.phy.data import valid_window_starts
from supermimic.phy.disturbances import ExternalWrenchCfg, ExternalWrenchController
from supermimic.phy.dynamics import (
  ForceDynamics,
  ForceWorldModelLearner,
  root_frame_wrench,
)
from supermimic.representations import create_representation


class FakeAsset:
  body_names = ["pelvis", "torso"]

  def __init__(self, n):
    self.data = SimpleNamespace(body_external_wrench=torch.zeros(n, 2, 6))

  def write_external_wrench_to_sim(self, force, torque, body_ids, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids[:, None]
    self.data.body_external_wrench[ids, body_ids] = torch.cat([force, torque], -1)


def test_force_persists_for_physics_then_clears_next_step_and_partial_reset():
  asset = FakeAsset(4)
  cfg = ExternalWrenchCfg(
    body_names=("pelvis", "torso"),
    force_range=(7, 7),
    torque_range=(2, 2),
    duration_steps=(2, 2),
    cooldown_steps=(1, 1),
    zero_probability=0,
  )
  controller = ExternalWrenchController(asset, 4, "cpu", cfg)
  assert controller.prepare()["external_wrench_w"].count_nonzero() == 0
  controller.advance()
  first = controller.prepare()
  controller.advance()
  second = controller.prepare()
  controller.advance()  # expiry must NOT erase the current simulator buffer
  torch.testing.assert_close(first["external_wrench_w"], second["external_wrench_w"])
  torch.testing.assert_close(
    asset.data.body_external_wrench.flatten(1), second["external_wrench_w"]
  )
  controller.reset(torch.tensor([0, 2]))
  assert asset.data.body_external_wrench[[0, 2]].count_nonzero() == 0
  assert asset.data.body_external_wrench[[1, 3]].count_nonzero() > 0
  assert controller.prepare()["external_wrench_w"].count_nonzero() == 0


def test_wrench_rotates_force_and_torque_without_merging_slots():
  aux = torch.tensor([[0, 0, 1, 2**-0.5, 0, 0, 2**-0.5]])
  force = torch.tensor([[1.0, 0, 0, 0, 1, 0, 0, 0, 3, 4, 0, 0]])
  actual = root_frame_wrench(force, aux)
  torch.testing.assert_close(
    actual,
    torch.tensor([[0.0, -1, 0, 1, 0, 0, 0, 0, 3, 0, -4, 0]]),
    atol=1e-6,
    rtol=1e-6,
  )


def test_wrench_changes_conditioned_prediction_but_blind_ignores_it():
  rep = create_representation(
    "gencoord", 21, 4, 0.02, 4, torch.zeros(4), actor_obs_mode="rep_state"
  )
  models = []
  for enabled in (True, False):
    torch.manual_seed(2)
    models.append(
      ForceDynamics(
        rep,
        4,
        1,
        torch.ones(4),
        torch.zeros(4),
        (16, 16),
        use_wrench=enabled,
      ).eval()
    )
  for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True):
    torch.testing.assert_close(a, b)
  state = torch.zeros(2, rep.rep_state_dim)
  aux = torch.tensor([[0.0, 0, 1, 1, 0, 0, 0]]).repeat(2, 1)
  action = torch.zeros(2, 4, requires_grad=True)
  ctx = torch.zeros(2, 1)
  wrench = torch.randn(2, 6, requires_grad=True)
  for enabled, model in zip((True, False), models, strict=True):
    output = model(state, action, aux, ctx, wrench)[1]
    zero = model(state, action, aux, ctx, torch.zeros_like(wrench))[1]
    if enabled:
      assert not torch.allclose(output, zero)
      da, dw = torch.autograd.grad(output.square().sum(), (action, wrench))
      assert torch.isfinite(da).all() and da.abs().sum() > 0
      assert torch.isfinite(dw).all() and dw.abs().sum() > 0
    else:
      torch.testing.assert_close(output, zero)


def test_rollout_consumes_matching_force_at_every_step():
  learner = object.__new__(ForceWorldModelLearner)
  learner.dynamics = lambda rep, action, aux, ctx, wrench: (aux, rep + wrench)
  actions = torch.zeros(1, 4, 1)
  force = torch.tensor([[[1.0], [2.0], [4.0], [100.0]]])
  predicted, _ = learner._rollout(
    torch.zeros(1, 1), actions, torch.zeros(1, 7), actions, force
  )
  torch.testing.assert_close(predicted, torch.tensor([[[1.0], [3.0], [7.0]]]))


def test_windows_do_not_cross_reset_and_holdout_environments_are_separate():
  terminated = torch.zeros(4, 9, dtype=torch.bool)
  terminated[:, 3] = True
  truncated = torch.zeros_like(terminated)
  for envs in (torch.tensor([0, 1, 2]), torch.tensor([3])):
    valid = valid_window_starts(terminated, truncated, envs, 4)
    assert set(valid[:, 0].tolist()) == set(envs.tolist())
    for env, start in valid.tolist():
      assert not terminated[env, start : start + 2].any()
    # Terminal at the last scored action is allowed (explicit terminal target).
    assert (valid[:, 1] == 1).any()
    assert not (valid[:, 1] == 2).any()
  with pytest.raises(ValueError, match="window length"):
    valid_window_starts(terminated, truncated, torch.tensor([0]), 10)


def test_command_teleport_invalidates_even_the_last_scored_target():
  done = torch.zeros(1, 8, dtype=torch.bool)
  valid_transition = torch.ones_like(done)
  valid_transition[:, 3] = False
  starts = valid_window_starts(done, done, torch.tensor([0]), 4, valid_transition)
  assert set(starts[:, 1].tolist()) == {0, 4}


@pytest.mark.parametrize(
  "change",
  [
    {"force_range": (2, -2)},
    {"duration_steps": (0, 2)},
    {"body_names": ("pelvis", "pelvis")},
    {"zero_probability": 1.1},
  ],
)
def test_invalid_disturbance_rejected(change):
  with pytest.raises(ValueError):
    ExternalWrenchCfg(**change).validate()


@pytest.mark.skipif(
  not os.environ.get("SUPERMIMIC_PHY_GPU_MOTION"),
  reason="set SUPERMIMIC_PHY_GPU_MOTION to a local motion for the G1 GPU probe",
)
def test_g1_force_changes_physics_with_identical_initial_state_and_action():
  import mjlab.tasks  # noqa: F401
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg

  import supermimic.tasks  # noqa: F401
  from supermimic.phy.environment import ForceTrackingEnv
  from supermimic.scripts.train_utils import setup_device

  device = setup_device("cuda:0")
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
  cfg.seed = 0
  cfg.scene.num_envs = 3
  cfg.events = {}  # Identical physical parameters in every world.
  cfg.auto_reset = False
  cfg.commands["motion"].motion_file = os.environ["SUPERMIMIC_PHY_GPU_MOTION"]
  cfg.commands["motion"].joint_position_range = (0.0, 0.0)
  disturbance = ExternalWrenchCfg(
    enabled=True,
    force_range=(0, 0),
    cooldown_steps=(0, 0),
    zero_probability=0,
  )
  env = RslRlVecEnvWrapper(ForceTrackingEnv(cfg, device, disturbance))
  try:
    obs, _ = env.reset()
    state = obs["full_state"].clone()
    torch.testing.assert_close(state, state[:1].expand_as(state), atol=1e-5, rtol=1e-5)
    controller = env.unwrapped.external_disturbance
    controller.wrench.zero_()
    controller.wrench[1, 0, 0] = 100.0
    controller.wrench[2, 0, 0] = -100.0
    controller.remaining.fill_(2)  # Keep these prescribed pulses in prepare().
    next_obs, _, _, extras = env.step(torch.zeros(3, env.num_actions, device=device))
    applied = extras["external_disturbance"]["external_wrench_w"]
    torch.testing.assert_close(
      applied[:, 0], torch.tensor([0.0, 100, -100], device=device)
    )
    assert applied[:, 1:].count_nonzero() == 0
    next_state = next_obs["full_state"]
    assert torch.isfinite(next_state).all()
    root_vx = next_state[:, 7 + env.num_actions]
    assert root_vx[1] > root_vx[0] + 1e-4
    assert root_vx[2] < root_vx[0] - 1e-4
    report = {
      "physical_response_check_passed": True,
      "comparison": "same batch, identical initial full states and zero actions; DR off",
      "force_x_N": applied[:, 0].tolist(),
      "next_root_vx_m_s": root_vx.tolist(),
      "control_dt": env.unwrapped.step_dt,
    }
    if path := os.environ.get("SUPERMIMIC_PHY_FORCE_REPORT"):
      Path(path).write_text(json.dumps(report, indent=2) + "\n")
  finally:
    env.close()
