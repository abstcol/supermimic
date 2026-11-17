"""Online force replay with explicit targets and reset/teleport boundaries."""

from __future__ import annotations

import copy

import torch

from supermimic.phy.data import valid_window_starts
from supermimic.replay_buffer import ReplayBuffer


class ForceReplayBuffer(ReplayBuffer):
  def snapshot(self):
    base = self._step if self.num_transitions == self.buffer_size else 0
    slots = (
      base + torch.arange(self.num_transitions, device=self.device)
    ) % self.buffer_size
    return self._storage[:, slots].to("cpu").to_dict()

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
    choices = torch.randint(starts.shape[0], (batch_size,), device=self.device)
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
