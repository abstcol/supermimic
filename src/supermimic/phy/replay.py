"""Online force replay with explicit targets and reset/teleport boundaries."""

from __future__ import annotations

import copy

import torch

from supermimic.phy.data import valid_window_starts
from supermimic.replay_buffer import ReplayBuffer


class ForceReplayBuffer(ReplayBuffer):
  def sample(self, sequence_length, batch_size):
    if sequence_length == 1:
      return super().sample(sequence_length, batch_size)
    n, length = self.num_transitions, sequence_length
    if length > self.buffer_size or n < length:
      raise ValueError("not enough stored frames for the requested window")
    base = self._step if n == self.buffer_size else 0
    slots = (base + torch.arange(n, device=self.device)) % self.buffer_size
    chronological = self._storage[:, slots]
    starts = valid_window_starts(
      chronological["terminated"],
      chronological["truncated"],
      torch.arange(self.num_envs, device=self.device),
      length,
      chronological["transition_valid"],
    )
    # Keep the original env-stratified distribution, even when failures leave
    # very different numbers of valid windows in each environment.
    counts = torch.bincount(starts[:, 0], minlength=self.num_envs)
    envs = torch.multinomial((counts > 0).float(), batch_size, replacement=True)
    offsets = counts.cumsum(0) - counts
    choices = (
      offsets[envs] + (torch.rand(batch_size, device=self.device) * counts[envs]).long()
    )
    pairs = starts[choices]
    storage_slots = (
      base + pairs[:, 1:] + torch.arange(length, device=self.device)
    ) % self.buffer_size
    return self._storage[pairs[:, :1], storage_slots]

  def sample_causal(self, sequence_length, batch_size):
    # Reuse the official history padding algorithm without mutating true flags.
    proxy = copy.copy(self)
    proxy._storage = self._storage.select(*self._storage.keys())
    proxy._storage["truncated"] = (
      self._storage["truncated"] | ~self._storage["transition_valid"]
    )
    return ReplayBuffer.sample_causal(proxy, sequence_length, batch_size)
