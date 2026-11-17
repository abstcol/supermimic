"""External wrenches condition dynamics, never the actor."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.models import Dynamics


def root_frame_wrench(wrench_w: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
  """Rotate world-frame force AND CoM torque to the current root frame.

  Each slot retains its own body's CoM as reference point. This is a rotation,
  not a transport to the root CoM (which would need an extra moment arm).
  """
  vectors = wrench_w.reshape(*wrench_w.shape[:-1], -1, 3)
  quat = aux[..., None, 3:7].expand(*vectors.shape[:-1], 4)
  return quat_apply_inverse(quat, vectors).reshape_as(wrench_w)


class ForceDynamics(Dynamics):
  """Official integrator and MLP, with an additional fixed-slot wrench input.

  The blind ablation uses exactly the same parameter count, initialization and
  data, but replaces its wrench channels by zero. Fixed unit scaling requires
  no estimated wrench statistics and preserves a physical zero.
  """

  def __init__(
    self,
    representation,
    num_actions,
    sim_ctx_dim,
    action_scale,
    action_offset,
    hidden_dims,
    *,
    num_bodies=1,
    use_wrench=True,
    force_scale=10.0,
    torque_scale=1.0,
  ):
    if num_bodies < 1 or force_scale <= 0 or torque_scale <= 0:
      raise ValueError("positive slot count and wrench scales required")
    super().__init__(
      representation,
      num_actions,
      sim_ctx_dim + 6 * num_bodies,
      action_scale,
      action_offset,
      hidden_dims,
    )
    self.use_wrench = use_wrench
    self.register_buffer(
      "wrench_scale",
      torch.tensor([force_scale] * 3 + [torque_scale] * 3).repeat(num_bodies),
    )

  def forward(self, rep_state, action, aux_state, sim_ctx, wrench_w=None):
    if wrench_w is None:
      # Existing force-free playback can use the same trained model.
      wrench_w = action.new_zeros(*action.shape[:-1], self.wrench_scale.numel())
    wrench = root_frame_wrench(wrench_w, aux_state) / self.wrench_scale
    if not self.use_wrench:
      wrench = torch.zeros_like(wrench)
    return super().forward(
      rep_state, action, aux_state, torch.cat([sim_ctx, wrench], dim=-1)
    )


class ForceWorldModelLearner(WorldModelLearner):
  """Keep the official loss, optimizer, normalization and captured-update path.

  An H-frame window scores H-1 transitions, matching the released learner.
  Explicit next_state contains the true pre-reset terminal state, not a reset
  observation. The dataset sampler excludes resets from the window interior.
  """

  def update(self, data):
    self.dynamics.train()
    for p in self._all_params:
      p.requires_grad_(True)
    batches = torch.stack(
      [
        td.select("state", "next_state", "action", "sim_ctx", "external_wrench_w")
        for td in data
      ]
    )
    return self._captured_update(batches)

  def _update_from(self, batches):
    stats = []
    for td in batches.unbind(0):
      states = td["state"]
      target = td["next_state"][:, :-1]
      pred_rep, pred_aux = self._compiled_rollout(
        self._compiled_represent(states[:, 0]),
        td["action"],
        self.representation.extract_aux_state(states[:, 0]),
        td["sim_ctx"],
        td["external_wrench_w"],
      )
      loss, mse = self._compiled_compute_losses(
        pred_rep,
        pred_aux,
        self._compiled_represent(target),
        self.representation.extract_aux_state(target),
      )
      from supermimic.learners.learner_utils import _optimize_step

      _optimize_step(self.optimizers, loss, self._all_params, max_norm=1.0)
      stats.append(torch.cat([loss.unsqueeze(0), mse]).detach())
    names = ["state", *self.loss_group_weights]
    return dict(
      zip((f"Loss/{n}" for n in names), torch.stack(stats).mean(0), strict=True)
    )

  def _rollout(self, init_rep_state, actions, init_aux, sim_ctx, wrench_w):
    rep, aux = init_rep_state, init_aux
    reps, auxs = [], []
    for k in range(actions.shape[1] - 1):
      aux, rep = self.dynamics(rep, actions[:, k], aux, sim_ctx[:, k], wrench_w[:, k])
      reps.append(rep)
      auxs.append(aux)
    return torch.stack(reps, dim=1), torch.stack(auxs, dim=1)
