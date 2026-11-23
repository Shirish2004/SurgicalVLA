"""Phase boundaries from proprioception and actions only.

Boundaries are the frames where the gripper command changes state (open -> close or
close -> open) plus the local minima of the arm's speed between them. Both signals
exist in every LeRobot manipulation dataset; which action dimensions are grippers is
a dataset property passed in by the caller, not looked up from any benchmark API.

Each boundary carries a confidence in [0, 1]: for gripper events, how cleanly the
command switches (a step is 1, a slow ramp or chatter is lower); for speed minima,
how deep the minimum is relative to the surrounding speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRIPPER_EVENT = 0
SPEED_MINIMUM = 1


@dataclass
class PhaseBoundaries:
    frames: np.ndarray  # (K,) int, sorted
    kinds: np.ndarray  # (K,) int, GRIPPER_EVENT | SPEED_MINIMUM
    confidence: np.ndarray  # (K,) float in [0, 1]

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"phase_frames": self.frames, "phase_kinds": self.kinds, "phase_confidence": self.confidence}


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x.astype(np.float64)
    kernel = np.ones(window) / window
    pad = window // 2
    return np.convolve(np.pad(x.astype(np.float64), pad, mode="edge"), kernel, mode="valid")[: len(x)]


def gripper_events(gripper: np.ndarray, min_gap: int = 5, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Frames where the (normalised) gripper command crosses its mid-level, with confidence.

    ``gripper`` is a 1-D command or opening signal. It is rescaled to [0, 1] per episode,
    so the sign convention (open = +1 or -1) does not matter. Crossings closer than
    ``min_gap`` frames to the previous one are treated as chatter and dropped.
    """
    g = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if len(g) < 3 or np.ptp(g) < 1e-6:
        return np.zeros(0, int), np.zeros(0)
    g = (g - g.min()) / np.ptp(g)
    above = g > 0.5
    crossings = np.flatnonzero(above[1:] != above[:-1]) + 1
    frames, conf = [], []
    for c in crossings:
        if frames and c - frames[-1] < min_gap:
            continue
        lo, hi = max(0, c - window), min(len(g), c + window)
        before, after = g[lo:c].mean(), g[c:hi].mean()
        frames.append(int(c))
        conf.append(float(np.clip(abs(after - before), 0.0, 1.0)))  # 1 for a clean step
    return np.asarray(frames, int), np.asarray(conf)


def speed_minima(
    positions: np.ndarray, window: int = 5, min_separation: int = 10, min_depth: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Local minima of smoothed speed, with depth-based confidence.

    ``positions`` is ``(T, D)``: any continuous proprioceptive coordinates (joint angles or
    end-effector pose, gripper dimensions excluded). Depth of a minimum is
    ``1 - v_min / max(v_left_peak, v_right_peak)``; minima shallower than ``min_depth``
    are discarded.
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    if len(p) < 2 * window + 3:
        return np.zeros(0, int), np.zeros(0)
    speed = _smooth(np.linalg.norm(np.diff(p, axis=0), axis=1), window)
    frames, conf = [], []
    for t in range(1, len(speed) - 1):
        if not (speed[t] <= speed[t - 1] and speed[t] < speed[t + 1]):
            continue
        left = speed[max(0, t - min_separation) : t].max(initial=speed[t])
        right = speed[t + 1 : t + 1 + min_separation].max(initial=speed[t])
        peak = max(left, right)
        depth = 0.0 if peak <= 1e-9 else 1.0 - speed[t] / peak
        if depth < min_depth:
            continue
        if frames and t + 1 - frames[-1] < min_separation:
            if depth > conf[-1]:
                frames[-1], conf[-1] = t + 1, depth
            continue
        frames.append(t + 1)  # speed[t] is between frames t and t+1
        conf.append(float(depth))
    return np.asarray(frames, int), np.asarray(conf)


def phase_boundaries(
    state: np.ndarray,
    action: np.ndarray,
    gripper_dims: list[int],
    *,
    window: int = 5,
    min_separation: int = 10,
    merge_radius: int = 3,
) -> PhaseBoundaries:
    """Gripper events (from the action's gripper dims) plus arm-speed minima (from the state)."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    frames, kinds, conf = [], [], []
    for d in gripper_dims:
        f, c = gripper_events(action[:, d], window=window)
        frames += list(f)
        kinds += [GRIPPER_EVENT] * len(f)
        conf += list(c)
    arm = [i for i in range(state.shape[1]) if i not in set(gripper_dims)]
    f, c = speed_minima(state[:, arm], window=window, min_separation=min_separation)
    for frame, cf in zip(f, c, strict=True):
        # a speed minimum at a gripper event is the same boundary
        if any(abs(frame - g) <= merge_radius for g, k in zip(frames, kinds, strict=True) if k == GRIPPER_EVENT):
            continue
        frames.append(int(frame))
        kinds.append(SPEED_MINIMUM)
        conf.append(float(cf))
    order = np.argsort(frames, kind="stable")
    return PhaseBoundaries(np.asarray(frames, int)[order], np.asarray(kinds, int)[order], np.asarray(conf)[order])


def phase_ids(n_frames: int, boundaries: np.ndarray) -> np.ndarray:
    """Per-frame segment index (0 before the first boundary, +1 after each)."""
    ids = np.zeros(n_frames, dtype=np.int16)
    for b in sorted(int(x) for x in boundaries):
        if 0 <= b < n_frames:
            ids[b:] += 1
    return ids


__all__ = ["GRIPPER_EVENT", "SPEED_MINIMUM", "PhaseBoundaries", "gripper_events", "phase_boundaries", "phase_ids", "speed_minima"]
