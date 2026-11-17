"""Exogenous wrench plans for differentiable actor-critic rollouts."""

from __future__ import annotations

import torch

from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.learners.policy_learner import ActorCriticPolicyLearner
from supermimic.phy.dynamics import root_frame_wrench


class WrenchPlanSampler:
  """Continue a replay snapshot's pulse/timer, then sample future pulses.

  Sampling happens outside compiled/captured policy updates. All H+1 inputs
  are detached, including the bootstrap state's current force and timer.
  The plan is privileged environment state, never an actor observation.
  """

  def __init__(self, cfg, device, seed):
    cfg.validate()
    self.cfg = cfg
    self.device = device
    self.generator = torch.Generator(device=device).manual_seed(seed)

  def _counts(self, bounds, n):
    return torch.randint(
      bounds[0], bounds[1] + 1, (n,), generator=self.generator, device=self.device
    )

  def sample(self, wrench, state, horizon):
    if horizon < 1:
      raise ValueError("positive rollout horizon required")
    cfg = self.cfg
    n, width = wrench.shape
    bodies = width // 6
    if width != len(cfg.body_names) * 6 or state.shape != (n, 2):
      raise ValueError("wrench slots/timer shape does not match disturbance config")
    current = wrench.detach().clone()
    remaining, cooldown = state[:, 0].clone(), state[:, 1].clone()
    forces, timers = [], []
    for _ in range(horizon + 1):
      forces.append(current if cfg.enabled else torch.zeros_like(current))
      timers.append(torch.stack([remaining, cooldown], -1))
      active, expired = remaining > 0, remaining == 1
      remaining = remaining - active.long()
      cooldown = torch.where(
        expired,
        self._counts(cfg.cooldown_steps, n),
        (cooldown - (~active).long()).clamp(min=0),
      )
      current = torch.where(expired[:, None], 0.0, current)
      ready = (remaining == 0) & (cooldown == 0)
      vectors = [
        lo
        + (hi - lo)
        * torch.rand(n, bodies, 3, generator=self.generator, device=self.device)
        for lo, hi in (cfg.force_range, cfg.torque_range)
      ]
      candidate = torch.cat(vectors, -1).flatten(1)
      keep = (
        torch.rand(n, generator=self.generator, device=self.device)
        >= cfg.zero_probability
      )
      candidate = candidate * keep[:, None]
      current = torch.where(ready[:, None], candidate, current)
      remaining = torch.where(ready, self._counts(cfg.duration_steps, n), remaining)
    return torch.stack(forces, 1).detach(), torch.stack(timers, 1).detach()


class ForceWorldModelEnv(MultiTaskWorldModelEnv):
  """Official tracking reward/termination/history with forced dynamics.

  Only the critic receives current wrench and pulse timers. No future-force
  information is exposed to actor or critic; the future plan belongs to the
  transition generator. After advancing H times the critic sees plan[H].
  """

  def __init__(self, *args, disturbance, horizon, plan_seed, **kwargs):
    super().__init__(*args, **kwargs)
    self.horizon = horizon
    self.plan_sampler = WrenchPlanSampler(disturbance, self.device, plan_seed)
    self.timer_scale = torch.tensor(
      [max(1, disturbance.duration_steps[1]), max(1, disturbance.cooldown_steps[1])],
      device=self.device,
    )
    self.last_plan_nonzero_fraction = torch.zeros((), device=self.device)

  def sample_initial_states(self):
    batch = super().sample_initial_states()
    batch["force_plan"], batch["timer_plan"] = self.plan_sampler.sample(
      batch["external_wrench_w"], batch["disturbance_state"], self.horizon
    )
    self.last_plan_nonzero_fraction = (
      (batch["force_plan"][:, : self.horizon].abs().sum(-1) > 0).float().mean()
    )
    return batch

  def reset(self, batch=None):
    if batch is None:
      batch = self.sample_initial_states()
    self._force_plan, self._timer_plan = batch["force_plan"], batch["timer_plan"]
    return super().reset(batch)

  def _critic_obs(self):
    return torch.cat(
      [
        super()._critic_obs(),
        root_frame_wrench(self._force_plan[:, 0], self.aux_state)
        / self.dynamics.wrench_scale,
        self._timer_plan[:, 0] / self.timer_scale,
      ],
      -1,
    )

  def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
    next_aux, next_rep = self.dynamics(
      rep_state, action, aux_state, sim_ctx, self._force_plan[:, 0]
    )
    # Fixed shapes avoid Python step counters and recompilation per horizon step.
    self._force_plan = torch.cat([self._force_plan[:, 1:], self._force_plan[:, -1:]], 1)
    self._timer_plan = torch.cat([self._timer_plan[:, 1:], self._timer_plan[:, -1:]], 1)
    joint_state = self.representation.joint_state(next_rep)
    ref_pos, ref_quat, ref_vel, ref_avel = self._ref_body_state
    done = self._compute_term_prob(next_aux, ref_pos, ref_quat)
    reward = self.representation.compute_wm_reward_from_reference(
      next_rep,
      next_aux,
      joint_state,
      action,
      last_action,
      ref_pos,
      ref_quat,
      ref_vel,
      ref_avel,
    )
    return next_aux, next_rep, joint_state, done, reward


class ForcePolicyLearner(ActorCriticPolicyLearner):
  def _graph_state(self):
    return super()._graph_state() + [self.wm_env.timer_scale]
