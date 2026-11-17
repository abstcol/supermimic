"""Control-step external wrenches; never part of the actor's observation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass
class ExternalWrenchCfg:
  enabled: bool = False
  body_names: tuple[str, ...] = ("pelvis",)
  force_range: tuple[float, float] = (-10.0, 10.0)  # N, per world-frame axis
  torque_range: tuple[float, float] = (0.0, 0.0)  # Nm, about each body's CoM
  duration_steps: tuple[int, int] = (2, 8)
  cooldown_steps: tuple[int, int] = (10, 40)
  zero_probability: float = 0.5
  seed: int = 0

  def validate(self) -> None:
    if not self.body_names or len(set(self.body_names)) != len(self.body_names):
      raise ValueError("body_names must be nonempty, unique, exact body names")
    for name in ("force_range", "torque_range"):
      lo, hi = getattr(self, name)
      if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError(f"invalid {name}")
    for name, minimum in (("duration_steps", 1), ("cooldown_steps", 0)):
      lo, hi = getattr(self, name)
      if not (isinstance(lo, int) and isinstance(hi, int) and minimum <= lo <= hi):
        raise ValueError(f"invalid {name}")
    if not 0.0 <= self.zero_probability <= 1.0:
      raise ValueError("zero_probability must be in [0, 1]")


class ExternalWrenchController:
  """Owns selected body slots. Writes once BEFORE the entire decimation loop.

  MuJoCo holds a world-frame wrench constant until overwritten. ``advance``
  changes the scheduler only, so the force already written remains in the sim
  throughout the current control step. Automatic resets clear only reset rows.
  """

  state_dim = 2  # remaining active steps, remaining cooldown steps (pre-step)

  def __init__(self, asset, num_envs: int, device: str, cfg: ExternalWrenchCfg):
    cfg.validate()
    self.cfg = cfg
    self.asset = asset
    self.num_envs = num_envs
    self.device = device
    self.body_ids = [asset.body_names.index(name) for name in cfg.body_names]
    self.wrench_dim = 6 * len(self.body_ids)
    self.generator = torch.Generator(device=device).manual_seed(cfg.seed)
    self.wrench = torch.zeros(num_envs, len(self.body_ids), 6, device=device)
    self.remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.cooldown = torch.zeros_like(self.remaining)
    self.reset()

  def _counts(self, bounds: tuple[int, int], count: int) -> torch.Tensor:
    return torch.randint(
      bounds[0],
      bounds[1] + 1,
      (count,),
      generator=self.generator,
      device=self.device,
    )

  def _vectors(self, bounds: tuple[float, float]) -> torch.Tensor:
    return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
      self.num_envs,
      len(self.body_ids),
      3,
      generator=self.generator,
      device=self.device,
    )

  def prepare(self) -> dict[str, torch.Tensor]:
    ready = (self.remaining == 0) & (self.cooldown == 0)
    candidate = torch.cat(
      [self._vectors(self.cfg.force_range), self._vectors(self.cfg.torque_range)],
      -1,
    )
    nonzero = (
      torch.rand(self.num_envs, generator=self.generator, device=self.device)
      >= self.cfg.zero_probability
    )
    candidate = candidate * nonzero[:, None, None]
    self.wrench.copy_(torch.where(ready[:, None, None], candidate, self.wrench))
    self.remaining.copy_(
      torch.where(
        ready,
        self._counts(self.cfg.duration_steps, self.num_envs),
        self.remaining,
      )
    )
    self.asset.write_external_wrench_to_sim(
      self.wrench[..., :3], self.wrench[..., 3:], body_ids=self.body_ids
    )
    # Log the actual simulator buffer, not the random candidate or next schedule.
    actual = self.asset.data.body_external_wrench[:, self.body_ids]
    return {
      "external_wrench_w": actual.reshape(self.num_envs, -1).clone(),
      "disturbance_state": torch.stack([self.remaining, self.cooldown], -1).clone(),
    }

  def advance(self) -> None:
    active = self.remaining > 0
    expired = self.remaining == 1
    self.remaining.sub_(active.long())
    self.cooldown.copy_(
      torch.where(
        expired,
        self._counts(self.cfg.cooldown_steps, self.num_envs),
        (self.cooldown - (~active).long()).clamp(min=0),
      )
    )
    self.wrench.masked_fill_(expired[:, None, None], 0.0)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    count = self.num_envs if env_ids is None else env_ids.numel()
    self.wrench[ids] = 0.0
    self.remaining[ids] = 0
    self.cooldown[ids] = self._counts(self.cfg.cooldown_steps, count)
    zeros = self.wrench.new_zeros(count, len(self.body_ids), 3)
    self.asset.write_external_wrench_to_sim(
      zeros, zeros, body_ids=self.body_ids, env_ids=env_ids
    )

  def metadata(self, step_dt: float) -> dict:
    return {
      "schema_version": 1,
      "config": asdict(self.cfg),
      "body_names": list(self.cfg.body_names),
      "frame": "world",
      "reference_point": "each body's center of mass",
      "layout": "body-major [Fx,Fy,Fz,Tx,Ty,Tz]",
      "units": ["N", "Nm"],
      "control_dt": step_dt,
      "fixed_within_control_step": True,
      "disturbance_state": ["remaining_active_steps", "remaining_cooldown_steps"],
    }
