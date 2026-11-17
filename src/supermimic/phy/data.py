"""Fixed-data windows, split by entire environment trajectories."""

from __future__ import annotations

import torch


def valid_window_starts(terminated, truncated, env_ids, length, transition_valid=None):
  """Return (env, start) pairs whose scored H-1 transitions never reset.

  A terminal transition may be the last scored transition: its explicit target
  is pre-reset. Its following frame is unused, as in the released WM learner.
  """
  if length < 2 or length > terminated.shape[1]:
    raise ValueError("window length must be between 2 and collected steps")
  done = (terminated | truncated)[env_ids]
  interior = done.unfold(1, length, 1)[..., : length - 2]
  valid = ~interior.any(-1)
  if transition_valid is not None:
    valid &= transition_valid[env_ids].unfold(1, length, 1)[..., : length - 1].all(-1)
  rows, starts = valid.nonzero(as_tuple=True)
  if rows.numel() == 0:
    raise ValueError("no reset-free windows")
  return torch.stack([env_ids[rows], starts], -1)


def sample_windows(data, starts, length, count, generator):
  """Draw CPU indices once; both learners consume the identical TensorDict."""
  choices = torch.randint(starts.shape[0], (count,), generator=generator)
  pairs = starts[choices].to(data.device)
  return data[pairs[:, :1], pairs[:, 1:] + torch.arange(length, device=data.device)]
