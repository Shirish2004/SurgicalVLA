"""Point tracks on the target object with TAPNext++ (Apache-2.0, vendored in ``third_party``).

Query points are sampled on a regular grid inside the target mask at its first confidently
segmented frame and tracked online through the episode (forward from that frame and, for
the frames before it, backward on the reversed clip). Coordinates are normalised to [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_CKPT = Path.home() / ".cache" / "surgicalvla" / "tapnextpp_ckpt.pt"


def grid_points_in_mask(mask: np.ndarray, n_points: int) -> np.ndarray:
    """Up to ``n_points`` ``[x, y]`` pixel coordinates on a grid covering ``mask``."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(1, int(np.ceil(np.sqrt(n_points * 2))))
    gx, gy = np.meshgrid(np.linspace(x0, x1, side), np.linspace(y0, y1, side))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inside = mask[pts[:, 1].round().astype(int), pts[:, 0].round().astype(int)]
    pts = pts[inside]
    if len(pts) > n_points:
        pts = pts[np.linspace(0, len(pts) - 1, n_points).round().astype(int)]
    return pts.astype(np.float32)


class PointTracker:
    def __init__(self, checkpoint: str | Path = DEFAULT_CKPT, device: str = "cuda", n_points: int = 32):
        from surgicalvla.third_party.tapnet.tapnextpp.votsp2026.model import TAPNextPP

        self.model = TAPNextPP.from_checkpoint(str(checkpoint), device=device, input_resolution=256)
        self.n_points = int(n_points)

    def _run(self, frames_bgr: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, vis, state = [], [], None
        for i, frame in enumerate(frames_bgr):
            p, v, state = self.model.track_frame(frame, query_points_xy=queries if i == 0 else None, state=state)
            pos.append(p)
            vis.append(v)
        return np.stack(pos, axis=1), np.stack(vis, axis=1)  # (P, F, 2), (P, F)

    def track(self, frames: np.ndarray, masks: np.ndarray, scores: np.ndarray, min_score: float = 0.5) -> dict[str, np.ndarray]:
        """``frames`` ``(F, H, W, 3)`` RGB uint8, target ``masks`` ``(F, H, W)`` and ``scores`` ``(F,)``."""
        n, h, w = frames.shape[:3]
        good = np.flatnonzero((scores >= min_score) & masks.reshape(n, -1).any(axis=1))
        if len(good) == 0:
            return {"tracks": np.zeros((0, n, 2), np.float16), "track_visible": np.zeros((0, n), bool)}
        t0 = int(good[0])
        queries = grid_points_in_mask(masks[t0], self.n_points)
        if len(queries) == 0:
            return {"tracks": np.zeros((0, n, 2), np.float16), "track_visible": np.zeros((0, n), bool)}
        bgr = np.ascontiguousarray(frames[..., ::-1])
        fwd_pos, fwd_vis = self._run(bgr[t0:], queries)
        pos = np.zeros((len(queries), n, 2), np.float32)
        vis = np.zeros((len(queries), n), bool)
        pos[:, t0:], vis[:, t0:] = fwd_pos, fwd_vis
        if t0 > 0:
            back_pos, back_vis = self._run(bgr[: t0 + 1][::-1], queries)
            pos[:, :t0] = back_pos[:, 1:][:, ::-1]
            vis[:, :t0] = back_vis[:, 1:][:, ::-1]
        pos /= np.array([w, h], np.float32)
        return {"tracks": pos.astype(np.float16), "track_visible": vis}


__all__ = ["DEFAULT_CKPT", "PointTracker", "grid_points_in_mask"]
