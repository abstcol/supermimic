"""Physical force injection before ManagerBasedRlEnv's physics substeps."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

from supermimic.phy.disturbances import ExternalWrenchCfg, ExternalWrenchController


class ForceTrackingEnv(ManagerBasedRlEnv):
  def __init__(
    self, cfg: ManagerBasedRlEnvCfg, device: str, disturbance: ExternalWrenchCfg
  ):
    self.external_disturbance: ExternalWrenchController | None = None
    # One owner per force slot; the controller must not compete with events
    # firing after physics or with nonphysical velocity kicks.
    for term in cfg.events.values():
      if getattr(term.func, "__name__", "") in {
        "apply_external_force_torque",
        "apply_body_impulse",
        "push_by_setting_velocity",
      }:
        raise ValueError(
          "external disturbance conflicts with an existing force/push event"
        )
    super().__init__(cfg=cfg, device=device)
    self.external_disturbance = ExternalWrenchController(
      self.scene["robot"], self.num_envs, device, disturbance
    )

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    super()._reset_idx(env_ids)
    if self.external_disturbance is not None:
      self.external_disturbance.reset(env_ids)

  def step(self, action: torch.Tensor):
    assert self.external_disturbance is not None
    applied = self.external_disturbance.prepare()
    # Advance BEFORE super.step: automatic resets then retain their fresh timers.
    # This changes only controller tensors, not the already-written sim wrench.
    self.external_disturbance.advance()
    obs, reward, terminated, truncated, extras = super().step(action)
    extras["external_disturbance"] = applied
    return obs, reward, terminated, truncated, extras
