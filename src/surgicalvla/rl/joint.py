"""Joint / interleaved SFT + RL (``training.mode``).

* ``sequential`` (default): RL from an SFT checkpoint, no demo loss.
* ``joint``: every update minimises ``L_GRPO (+ KL to reference) + lambda_t * L_FM``,
  where ``L_FM`` is the flow-matching loss on a separate demonstration batch streamed
  from the Hub. ``lambda_t`` follows :class:`LambdaSchedule`.
* ``interleaved``: ``sft_steps`` pure flow-matching updates every ``interleave_every``
  RL updates.

Why log gradient statistics: the demo term is a regulariser against forgetting the
SFT distribution; whether it helps or fights the policy gradient shows up directly
in ``cos(g_RL, g_FM)`` (negative = conflict) and in the ratio of the two norms,
which also tells whether ``lambda`` is in a sensible range.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

MODES = ("sequential", "joint", "interleaved")


@dataclass
class LambdaSchedule:
    start: float = 1.0
    end: float = 0.1
    kind: str = "cosine"  # constant | linear | cosine | exp
    steps: int = 200

    def __post_init__(self):
        if self.kind not in ("constant", "linear", "cosine", "exp"):
            raise ValueError(f"unknown lambda schedule {self.kind!r}")
        if self.kind == "exp" and (self.start <= 0 or self.end <= 0):
            raise ValueError("exp schedule needs positive start and end")

    def __call__(self, step: int) -> float:
        if self.kind == "constant" or self.steps <= 0:
            return float(self.start)
        frac = min(max(step / self.steps, 0.0), 1.0)
        if self.kind == "linear":
            return float(self.start + (self.end - self.start) * frac)
        if self.kind == "cosine":
            return float(self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * frac)))
        return float(self.start * (self.end / self.start) ** frac)


def combine_gradients(params: list[torch.nn.Parameter], g_rl: list[torch.Tensor | None], g_fm: list[torch.Tensor | None], lam: float) -> dict[str, float]:
    """Write ``g_rl + lam * g_fm`` into ``.grad`` and return the per-term statistics."""
    dot = n_rl = n_fm = 0.0
    for p, a, b in zip(params, g_rl, g_fm, strict=True):
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b
        dot += float((a.float() * b.float()).sum())
        n_rl += float(a.float().pow(2).sum())
        n_fm += float(b.float().pow(2).sum())
        p.grad = (a + lam * b).to(p.dtype)
    n_rl, n_fm = math.sqrt(n_rl), math.sqrt(n_fm)
    return {
        "grad_norm_rl": n_rl,
        "grad_norm_fm": n_fm,
        "grad_norm_fm_weighted": lam * n_fm,
        "grad_cosine_rl_fm": dot / (n_rl * n_fm) if n_rl > 0 and n_fm > 0 else 0.0,
        "lambda": lam,
    }


def take_grads(params: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    """Move the accumulated ``.grad`` out of the parameters (and clear them)."""
    out = []
    for p in params:
        out.append(None if p.grad is None else p.grad.detach().clone())
        p.grad = None
    return out


class DemoStream:
    """Endless batches of demonstrations from a (Hub-streamed) LeRobot dataset, resumable.

    ``dataset`` is any iterable of sample dicts exposing ``state_dict`` /
    ``load_state_dict`` (LeRobot's ``StreamingLeRobotDataset``), iterated in-process so
    the saved position is exact. ``rename`` maps dataset keys to policy keys.
    """

    def __init__(self, dataset: Iterable[dict], batch_size: int, rename: dict[str, str] | None = None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rename = dict(rename or {})
        self._it: Iterator[dict] | None = None
        self.epoch = 0

    def _next_sample(self) -> dict:
        if self._it is None:
            self._it = iter(self.dataset)
        try:
            return next(self._it)
        except StopIteration:
            self.epoch += 1
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(self.epoch)
            self._it = iter(self.dataset)
            return next(self._it)

    def next_batch(self) -> dict:
        samples = [self._next_sample() for _ in range(self.batch_size)]
        batch: dict = {}
        for key in samples[0]:
            name = self.rename.get(key, key)
            values = [s[key] for s in samples]
            batch[name] = torch.stack(values) if torch.is_tensor(values[0]) else values
        return batch

    def state_dict(self) -> dict:
        inner = self.dataset.state_dict() if hasattr(self.dataset, "state_dict") else {}
        return {"dataset": inner, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state.get("epoch", 0))
        if hasattr(self.dataset, "load_state_dict") and state.get("dataset"):
            self.dataset.load_state_dict(state["dataset"])
        self._it = None


def make_demo_stream(policy_config, demo_cfg: dict, seed: int) -> DemoStream:
    """Hub-streamed demonstrations with the action chunk the policy trains on."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

    repo = demo_cfg["repo_id"]
    meta = LeRobotDatasetMetadata(repo, revision=demo_cfg.get("revision"))
    episodes = demo_cfg.get("episodes")
    blocks = demo_cfg.get("episode_blocks")
    if blocks:  # RoboTwin demo_clean: first `take` of every `block` episodes
        from surgicalvla.utils.splits import block_episodes

        episodes = block_episodes(meta.total_episodes, int(blocks["block"]), int(blocks["take"]))
    dataset = StreamingLeRobotDataset(
        repo,
        episodes=episodes,
        revision=demo_cfg.get("revision"),
        delta_timestamps=resolve_delta_timestamps(policy_config, meta),
        streaming=True,
        buffer_size=int(demo_cfg.get("buffer_size", 1000)),
        seed=int(seed),
        shuffle=True,
        repeat=True,
    )
    return DemoStream(dataset, int(demo_cfg.get("batch_size", 8)), demo_cfg.get("rename_map"))


__all__ = ["MODES", "DemoStream", "LambdaSchedule", "combine_gradients", "make_demo_stream", "take_grads"]
