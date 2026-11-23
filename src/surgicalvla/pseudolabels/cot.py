"""Explicit reasoning traces from pseudo-labels, filtered by action-likelihood gain.

Candidates are short structured traces written from an episode's pseudo-labels (no human
annotation): the phase (from gripper events), the target phrase, and the target's motion.
Each candidate is scored by how much it lowers the policy's own flow-matching loss on the
demonstrated action chunk, compared with no trace, using identical noise and flow time for
both (a paired comparison). Only traces with positive gain are kept. This is the
action-predictiveness filter of R&B-EnCoRe, with the flow-matching loss standing in for the
action log-likelihood.
"""

from __future__ import annotations

import numpy as np

PHASES = ("approach", "grasp", "transport", "release")


def phase_name(frame: int, phase_frames: np.ndarray, phase_kinds: np.ndarray) -> str:
    """Name the segment containing ``frame`` from the gripper events (kind 0) before it."""
    events = [int(f) for f, k in zip(phase_frames, phase_kinds, strict=True) if k == 0]
    n_before = sum(f <= frame for f in events)
    if n_before == 0:
        return "approach"
    if n_before == 1:
        close = events[0]
        return "grasp" if frame - close < 10 else "transport"
    return "release"


def motion_word(motion: np.ndarray, eps: float = 0.003) -> str:
    dx, dy = float(motion[0]), float(motion[1])
    if max(abs(dx), abs(dy)) < eps:
        return "still"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def candidates(labels: dict, frame: int) -> list[str]:
    """Candidate traces for ``frame`` of one episode's sidecar ``labels``."""
    idx = labels["frame_index"]
    j = int(np.clip(np.searchsorted(idx, frame, side="right") - 1, 0, len(idx) - 1))
    obj = str(labels.get("target_phrase", "")) or "object"
    phase = phase_name(frame, labels["phase_frames"], labels["phase_kinds"])
    move = motion_word(labels["obj_motion"][j]) if "obj_motion" in labels else "still"
    return [
        f"<cot> phase: {phase} | object: {obj} </cot>",
        f"<cot> phase: {phase} | object: {obj} | motion: {move} </cot>",
        f"<cot> object: {obj} </cot>",
    ]


def score(policy, batch: dict, traces: list[str], tokenizer, max_tokens: int, n_samples: int = 4, seed: int = 0) -> list[float]:
    """Flow-matching loss gain of each trace over no trace (higher = more action-predictive)."""
    import torch

    device = next(policy.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    actions = batch["action"]
    shape = (actions.shape[0], policy.config.chunk_size, policy.config.max_action_dim)
    noises = [torch.randn(shape, generator=gen).to(device) for _ in range(n_samples)]
    times = [policy.model.sample_time(actions.shape[0], "cpu").to(device) for _ in range(n_samples)]

    def loss(extra: dict) -> float:
        total = 0.0
        with torch.no_grad():
            for noise, t in zip(noises, times, strict=True):
                total += float(policy.forward({**batch, **extra}, noise=noise, time=t)[1]["fm_loss"])
        return total / n_samples

    base = loss({})
    gains = []
    for trace in traces:
        enc = tokenizer([trace] * actions.shape[0], return_tensors="pt", padding="max_length", max_length=max_tokens, truncation=True, add_special_tokens=False)
        extra = {"cot_input_ids": enc.input_ids.to(device), "cot_attention_mask": enc.attention_mask.bool().to(device)}
        gains.append(base - loss(extra))
    return gains


__all__ = ["PHASES", "candidates", "motion_word", "phase_name", "score"]
