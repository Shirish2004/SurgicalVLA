"""Resumable shuffling for map-style datasets."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[int]):
    """Endless stream of dataset indices: epoch ``e`` is a permutation seeded by ``seed + e``.

    The stream is a pure function of ``(seed, position)``, so a run resumed with
    ``set_position(samples_consumed)`` continues exactly where it stopped, without loading the
    skipped samples.
    """

    def __init__(self, num_samples: int, seed: int = 0):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = num_samples
        self.seed = seed
        self.position = 0  # samples drawn so far

    def set_position(self, samples_consumed: int) -> None:
        self.position = int(samples_consumed)

    def _perm(self, epoch: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed + epoch)
        return torch.randperm(self.num_samples, generator=g)

    def __iter__(self) -> Iterator[int]:
        epoch, offset = divmod(self.position, self.num_samples)
        while True:
            for i in self._perm(epoch)[offset:].tolist():
                self.position += 1
                yield i
            epoch, offset = epoch + 1, 0

    def __len__(self) -> int:
        return self.num_samples
