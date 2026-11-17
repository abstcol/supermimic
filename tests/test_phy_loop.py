"""Pulse continuation, force isolation, gradients, and causal online replay."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
from supermimic.phy.disturbances import ExternalWrenchCfg, ExternalWrenchController
from supermimic.phy.replay import ForceReplayBuffer
from supermimic.phy.rollout import (
  ForcePolicyLearner,
  ForceWorldModelEnv,
  WrenchPlanSampler,
)
from supermimic.replay_buffer import FieldSpec
from supermimic.tasks.task_spec import TaskSpec
from tests.test_phy_force import FakeAsset


def test_plan_continues_current_pulse_and_obeys_cooldown_and_duration():
  cfg = ExternalWrenchCfg(
    force_range=(5, 5),
    duration_steps=(2, 2),
    cooldown_steps=(1, 1),
    zero_probability=0,
  )
  sampler = WrenchPlanSampler(cfg, "cpu", 3)
  initial = torch.tensor([[3.0, 0, 0, 0, 0, 0]], requires_grad=True)
  force, timers = sampler.sample(initial, torch.tensor([[1, 0]]), 5)
  torch.testing.assert_close(force[0, :, 0], torch.tensor([3.0, 0, 5, 5, 0, 5]))
  assert timers[0, :, 0].tolist() == [1, 0, 2, 1, 0, 2]
  assert timers[0, :, 1].tolist() == [0, 1, 0, 0, 1, 0]
  assert not force.requires_grad and force.shape == (1, 6, 6)


def test_disabled_disturbance_is_zero_in_real_controller_and_model_plan():
  cfg = ExternalWrenchCfg(
    enabled=False, force_range=(10, 10), cooldown_steps=(0, 0), zero_probability=0
  )
  controller = ExternalWrenchController(FakeAsset(2), 2, "cpu", cfg)
  assert controller.prepare()["external_wrench_w"].count_nonzero() == 0
  force, _ = WrenchPlanSampler(cfg, "cpu", 1).sample(
    torch.ones(2, 6), torch.tensor([[2, 0], [0, 0]]), 6
  )
  assert force.count_nonzero() == 0


def _buffer():
  schema = {
    "state": FieldSpec(1),
    "actor_obs": FieldSpec(1),
    "transition_valid": FieldSpec(dtype=torch.bool),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
  }
  buffer = ForceReplayBuffer(schema, buffer_size=8, num_envs=1)
  for i in range(12):
    buffer.insert(
      TensorDict(
        {
          "state": torch.tensor([[float(i)]]),
          "actor_obs": torch.tensor([[float(i)]]),
          "transition_valid": torch.tensor([i != 7]),
          "terminated": torch.tensor([i == 8]),
          "truncated": torch.tensor([False]),
        },
        batch_size=[1],
      )
    )
  return buffer


def test_online_wm_windows_never_cross_reset_or_score_a_teleport():
  batch = _buffer().sample(4, 128)
  for row in batch["state"].squeeze(-1).tolist():
    # The buffer wrapped: logical ages are [4..11]; only start 4 is valid.
    assert row == [4.0, 5, 6, 7]
    scored = row[:-1]
    assert 7.0 not in scored  # invalid physics target
    assert 8.0 not in scored[:-1]  # reset cannot be in scored-window interior
    assert all(b == a + 1 for a, b in zip(row[:-1], row[1:], strict=True))


def test_force_windows_preserve_original_uniform_environment_sampling():
  schema = {
    "state": FieldSpec(1),
    "transition_valid": FieldSpec(dtype=torch.bool),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
  }
  buffer = ForceReplayBuffer(schema, 8, 2)
  for step in range(8):
    buffer.insert(
      TensorDict(
        {
          "state": torch.tensor([[0.0], [1.0]]),
          "transition_valid": torch.tensor([True, step == 0]),
          "terminated": torch.zeros(2, dtype=torch.bool),
          "truncated": torch.zeros(2, dtype=torch.bool),
        },
        batch_size=[2],
      )
    )
  with torch.random.fork_rng():
    torch.manual_seed(3)
    fraction = float(buffer.sample(2, 2048)["state"][:, 0, 0].mean())
  assert 0.45 < fraction < 0.55  # 7 windows vs 1 must not give a 7:1 env ratio.


def test_online_sensor_history_pads_at_teleport_and_reset_without_changing_flags():
  buffer = _buffer()
  truncated_before = buffer._storage["truncated"].clone()
  batch, padded = buffer.sample_causal(4, 256)
  frames = batch["state"].squeeze(-1)
  torch.testing.assert_close(buffer._storage["truncated"], truncated_before)
  for row, pad in zip(frames.tolist(), padded.tolist(), strict=True):
    assert not pad[-1]
    if row[-1] == 8:
      assert row == [8.0, 8, 8, 8]  # preceding transition teleported
    if row[-1] == 9:
      assert row == [9.0, 9, 9, 9]  # preceding transition terminated
    if row[-1] in (8, 9):
      assert all(pad[:-1])
    assert all(b == a or b == a + 1 for a, b in zip(row[:-1], row[1:], strict=True))


class ToyRepresentation(torch.nn.Module):
  dt = 0.02

  def represent(self, state):
    return state[..., 7:]

  def extract_aux_state(self, state):
    return state[..., :7]

  def joint_state(self, state):
    return state

  def compute_actor_sensor(self, rep_state, *args, **kwargs):
    return rep_state

  def compute_wm_goal(self, motion, times, aux):
    goal = torch.zeros(times.shape[0], 1)
    return goal, goal

  def compute_wm_reward_from_reference(self, state, *args):
    return -state.square().squeeze(-1)


class ToyDynamics(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.gain = torch.nn.Parameter(torch.tensor(0.5))
    self.register_buffer("wrench_scale", torch.ones(6))

  def forward(self, rep, action, aux, ctx, wrench):
    return aux, rep + self.gain * action + 0.01 * wrench[:, :1]


class ToyActor(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(1, 1)
    torch.nn.init.constant_(self.linear.weight, 0.1)
    torch.nn.init.zeros_(self.linear.bias)

  def forward(self, obs):
    return self.linear(obs["history"][:, -1])


def _toy_env():
  n = 8
  state = torch.zeros(n, 8)
  state[:, 3] = 1
  state[:, 7] = 1
  start = TensorDict(
    {
      "state": state,
      "prev_action": torch.zeros(n, 1),
      "sim_ctx": torch.zeros(n, 1),
      "motion_time": torch.zeros(n, dtype=torch.long),
      "external_wrench_w": torch.zeros(n, 6),
      "disturbance_state": torch.tensor([[0, 1]]).repeat(n, 1),
    },
    batch_size=[n],
  )

  def body_state(times):
    pos = torch.zeros(n, 1, 3)
    quat = torch.tensor([1.0, 0, 0, 0]).reshape(1, 1, 4).repeat(n, 1, 1)
    return pos, quat, pos, pos

  motion = SimpleNamespace(motion_ends=torch.tensor([1000]), body_state=body_state)
  command = SimpleNamespace(motion=motion, motion_anchor_body_index=0)
  real_env = SimpleNamespace(
    num_actions=1,
    unwrapped=SimpleNamespace(
      step_dt=0.02, command_manager=SimpleNamespace(get_term=lambda _: command)
    ),
    cfg=SimpleNamespace(
      rewards={"tracking": SimpleNamespace(weight=1.0)},
      terminations={
        "anchor_pos": SimpleNamespace(params={"threshold": 10.0}),
        "anchor_ori": SimpleNamespace(params={"threshold": 10.0}),
      },
    ),
  )
  env = ForceWorldModelEnv(
    dynamics=ToyDynamics(),
    representation=ToyRepresentation(),
    env=real_env,
    batch_size=n,
    replay_buffer=SimpleNamespace(
      sample=lambda length, count: start.unsqueeze(1).clone()
    ),
    task_specs=[
      TaskSpec(
        name="motion", goal_dim=1, extract_training_goal=lambda *a: torch.zeros(n, 1)
      )
    ],
    history_length=1,
    termination_softness=0.05,
    device="cpu",
    horizon=2,
    plan_seed=1,
    disturbance=ExternalWrenchCfg(
      force_range=(5, 5),
      duration_steps=(2, 2),
      cooldown_steps=(1, 1),
      zero_probability=0,
    ),
  )
  return env


def test_actor_inputs_exclude_current_and_future_wrench_and_bootstrap_uses_next_force():
  env = _toy_env()
  batch = env.sample_initial_states()
  batch["force_plan"] = torch.zeros(8, 3, 6)
  obs_zero = env.reset(batch)
  batch["force_plan"] = torch.arange(3.0).reshape(1, 3, 1).repeat(8, 1, 6) + 10
  obs_force = env.reset(batch)
  assert set(obs_force.keys()) == {"history", "goal_motion", "critic"}
  for key in ("history", "goal_motion"):
    torch.testing.assert_close(obs_zero[key], obs_force[key])
  assert not torch.equal(obs_zero["critic"], obs_force["critic"])
  next_obs, _, _, _ = env.step(torch.zeros(8, 1))
  torch.testing.assert_close(env._force_plan[:, 0], torch.full((8, 6), 11.0))
  torch.testing.assert_close(next_obs["critic"][:, -8:-2], torch.full((8, 6), 11.0))
  env.step(torch.zeros(8, 1))
  torch.testing.assert_close(env._force_plan[:, 0], torch.full((8, 6), 12.0))


def test_policy_gradient_crosses_forced_model_without_updating_model_weights():
  env = _toy_env()
  actor, critic = ToyActor(), torch.nn.Linear(11, 1)
  torch.nn.init.zeros_(critic.weight)
  torch.nn.init.zeros_(critic.bias)
  learner = ForcePolicyLearner(
    actor,
    critic,
    env,
    ActorCriticPolicyLearnerCfg(horizon=2, batch_size=8),
    device="cpu",
  )
  old_actor = actor.linear.weight.detach().clone()
  old_model = env.dynamics.gain.detach().clone()
  stats = learner.update()
  assert all(torch.isfinite(value).all() for value in stats.values())
  assert stats["Policy/grad_norm"] > 0
  assert not torch.equal(actor.linear.weight, old_actor)
  torch.testing.assert_close(env.dynamics.gain, old_model)
  assert not env.dynamics.gain.requires_grad and env.dynamics.gain.grad is None


def test_force_environment_is_opt_in_and_default_auto_reset_is_preserved(monkeypatch):
  from supermimic.scripts import train_utils

  monkeypatch.setattr(train_utils, "RslRlVecEnvWrapper", lambda env: env)
  monkeypatch.setattr(train_utils, "ManagerBasedRlEnv", lambda **kw: ("official", kw))
  monkeypatch.setattr(train_utils, "ForceTrackingEnv", lambda **kw: ("force", kw))
  cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=0), auto_reset=True)
  kind, _ = train_utils.create_env(cfg, 8, "cpu")
  assert kind == "official" and cfg.auto_reset is True
  kind, kwargs = train_utils.create_env(cfg, 8, "cpu", ExternalWrenchCfg())
  assert kind == "force" and cfg.auto_reset is False
  assert kwargs["disturbance"].enabled
