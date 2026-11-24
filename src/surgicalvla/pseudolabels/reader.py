"""Training-time access to sidecar pseudo-labels, keyed by ``(episode_index, frame_index)``.

For a dataset frame ``f`` the nearest labelled frame at or before it is used. Each target
comes with its confidence; episodes without a sidecar get zero confidence, so they add
nothing to the auxiliary losses. Keys follow the ``aux.*`` convention the model reads, plus
``subtask_phase_id`` (pseudo phase segment, clipped to ``num_phases - 1``).
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from surgicalvla.pseudolabels.phases import phase_ids
from surgicalvla.pseudolabels.sidecar import SidecarStore


class SidecarTargets:
    def __init__(
        self,
        root: str,
        dataset_id: str,
        *,
        feat_dim: int = 384,
        track_horizon: int = 5,
        num_phases: int = 4,
        cache_episodes: int = 512,
        tokenizer=None,
        cot_max_tokens: int = 48,
    ):
        self.store = SidecarStore(root, dataset_id)
        self.feat_dim = int(feat_dim)
        self.track_horizon = int(track_horizon)
        self.num_phases = int(num_phases)
        self._cache: OrderedDict[int, dict | None] = OrderedDict()
        self._cache_size = int(cache_episodes)
        self.tokenizer = tokenizer  # set to feed explicit CoT traces (cot_<episode>.json) to training
        self.cot_max_tokens = int(cot_max_tokens)
        self._cot: dict[int, list] = {}

    def trace(self, ep: int, frame: int) -> str:
        """Stored reasoning trace of the phase segment containing ``frame`` ('' if none)."""
        if ep not in self._cot:
            path = self.store.dir / f"cot_{ep:06d}.json"
            import json

            self._cot[ep] = json.loads(path.read_text()) if path.exists() else []
        rows = self._cot[ep]
        d = self._episode(ep)
        if not rows or d is None:
            return ""
        seg = int(phase_ids(max(int(frame) + 1, 1), d["phase_frames"])[-1])
        for r in rows:
            if r["segment"] == seg:
                return r["trace"] or ""
        return ""

    def _episode(self, ep: int) -> dict | None:
        if ep in self._cache:
            self._cache.move_to_end(ep)
            return self._cache[ep]
        data = self.store.read(ep) if self.store.done(ep) and self.store.path(ep).exists() else None
        self._cache[ep] = data
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def frame_targets(self, ep: int, frame: int) -> dict[str, np.ndarray]:
        out = {
            "aux.obj_feat": np.zeros(self.feat_dim, np.float32), "aux.obj_feat_conf": np.float32(0),
            "aux.obj_motion": np.zeros(2, np.float32), "aux.obj_motion_conf": np.float32(0),
            "aux.track_disp": np.zeros(2, np.float32), "aux.track_disp_conf": np.float32(0),
            "aux.rel_disp": np.zeros(3, np.float32), "aux.rel_disp_conf": np.float32(0),
            "subtask_phase_id": np.int64(0),
        }
        d = self._episode(ep)
        if d is None or "obj_score" not in d:
            return out
        idx = d["frame_index"]
        j = int(np.clip(np.searchsorted(idx, frame, side="right") - 1, 0, len(idx) - 1))
        score = float(d["obj_score"][j])
        out["aux.obj_feat"] = d["obj_feat"][j].astype(np.float32)
        out["aux.obj_feat_conf"] = np.float32(score)
        out["aux.obj_motion"] = d["obj_motion"][j].astype(np.float32)
        out["aux.obj_motion_conf"] = np.float32(d["motion_confidence"][j])
        out["aux.rel_disp"] = d["rel_disp"][j].astype(np.float32)
        out["aux.rel_disp_conf"] = np.float32(d["rel_confidence"][j])
        if "tracks" in d and len(d["tracks"]):
            k = min(j + self.track_horizon, len(idx) - 1)
            vis = d["track_visible"][:, j] & d["track_visible"][:, k]
            if vis.any():
                disp = d["tracks"][vis, k].astype(np.float32) - d["tracks"][vis, j].astype(np.float32)
                out["aux.track_disp"] = disp.mean(axis=0)
                out["aux.track_disp_conf"] = np.float32(vis.mean() * score)
        seg = phase_ids(int(idx[-1]) + 1, d["phase_frames"])[min(int(frame), int(idx[-1]))]
        out["subtask_phase_id"] = np.int64(min(int(seg), self.num_phases - 1))
        return out

    def overlay(self, ep: int, frame: int, image: np.ndarray) -> np.ndarray | None:
        """``image`` (H, W, 3) uint8 of ``frame`` with its pseudo-labels drawn (None if unlabelled).

        Labels come from the nearest labelled frame at or before ``frame``.
        """
        from surgicalvla.pseudolabels.viz import draw_frame

        d = self._episode(int(ep))
        if d is None or "obj_score" not in d:
            return None
        idx = d["frame_index"]
        j = int(np.clip(np.searchsorted(idx, frame, side="right") - 1, 0, len(idx) - 1))
        phase = int(phase_ids(int(idx[-1]) + 1, d["phase_frames"])[min(int(frame), int(idx[-1]))])
        lines = [f"ep {int(ep)} frame {int(frame)} phase {phase}"]
        trace = self.trace(int(ep), int(frame))
        if trace:
            lines.append(trace.replace("<cot>", "").replace("</cot>", "").strip()[:60])
        return draw_frame(image, d, j, trail=self.track_horizon, lines=tuple(lines))

    def for_batch(self, episode_index, frame_index) -> dict[str, torch.Tensor]:
        eps = torch.as_tensor(episode_index).reshape(-1).tolist()
        frames = torch.as_tensor(frame_index).reshape(-1).tolist()
        rows = [self.frame_targets(int(e), int(f)) for e, f in zip(eps, frames, strict=True)]
        out = {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0]}
        if self.tokenizer is not None:
            texts = [self.trace(int(e), int(f)) for e, f in zip(eps, frames, strict=True)]
            if any(texts):
                enc = self.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=self.cot_max_tokens,
                                     truncation=True, add_special_tokens=False)
                out["cot_input_ids"] = enc.input_ids
                out["cot_attention_mask"] = enc.attention_mask.bool()
        return out


def split_future_frames(batch: dict, image_keys: list[str]) -> dict:
    """``(B, 2, ...)`` observations from delta indices ``[0, h]`` -> current frame, plus ``future.<key>`` images."""
    for key, value in list(batch.items()):
        if not (key.startswith("observation.") and torch.is_tensor(value)):
            continue
        if key in image_keys and value.dim() == 5 and value.shape[1] == 2:
            batch[f"future.{key}"] = value[:, 1]
            batch[key] = value[:, 0]
        elif value.dim() == 3 and value.shape[1] == 2 and key not in image_keys:
            batch[key] = value[:, 0]
    return batch


__all__ = ["SidecarTargets", "split_future_frames"]
