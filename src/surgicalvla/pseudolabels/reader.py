"""Training-time access to the feature cache, keyed by ``(episode_index, frame_index)``.

Targets: for a dataset frame ``f`` the nearest labelled frame at or before it is used. Each
target comes with its confidence; episodes without cached labels get zero confidence, so they
add nothing to the auxiliary losses. Keys follow the ``aux.*`` convention the model reads, plus
``subtask_phase_id`` (pseudo phase segment, clipped to ``num_phases - 1``).

Inputs (``object_input``): the object descriptors of the latest input frame at or before
``f``, never a later one, as ``obj_sam_feat (K, C)``, ``obj_pose_vec (K, 8)``,
``obj_valid_mask (K,)``. ``"objects"`` gives the target candidates and the end effector,
``"grid"`` the 2×2-grid control from the same encoders.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from surgicalvla.pseudolabels.phases import phase_ids
from surgicalvla.pseudolabels.cache import FeatureCache, read_traces, rle_decode
from surgicalvla.pseudolabels.sidecar import mask_to_bits


class SidecarTargets:
    """Reader of one cache directory ``<root>/<dataset-slug>/<name>`` (``FeatureCache``).

    ``expect``: manifest fields the training config requires (``check_manifest``); a
    mismatch raises before training starts.
    """

    def __init__(
        self,
        root: str,
        dataset_id: str,
        *,
        name: str | None = None,
        expect: dict | None = None,
        feat_dim: int = 384,
        track_horizon: int = 5,
        num_phases: int = 4,
        cache_episodes: int = 512,
        tokenizer=None,
        cot_max_tokens: int = 48,
        object_input: str = "none",
        upload_repo: str | None = None,
    ):
        self.cache = FeatureCache(root, dataset_id, name or latest_cache(root, dataset_id), expect=expect, upload_repo=upload_repo)
        self.feat_dim = int(feat_dim)
        self.track_horizon = int(track_horizon)
        self.num_phases = int(num_phases)
        self.object_input = object_input
        self._cache: OrderedDict[int, dict | None] = OrderedDict()
        self._cache_size = int(cache_episodes)
        self.tokenizer = tokenizer  # set to feed explicit CoT traces to training
        self.cot_max_tokens = int(cot_max_tokens)
        self._cot: dict[int, list] | None = None

    @property
    def manifest(self) -> dict:
        return self.cache.manifest

    def trace(self, ep: int, frame: int) -> str:
        """Kept reasoning trace of the phase segment containing ``frame`` ('' if none)."""
        if self._cot is None:
            self._cot = read_traces(self.cache)
        rows = self._cot.get(int(ep), [])
        d = self._episode(ep)
        if not rows or d is None:
            return ""
        seg = int(phase_ids(max(int(frame) + 1, 1), d["phase_frames"])[-1])
        for r in rows:
            if int(r["segment"]) == seg and r.get("kept", True):
                return r["trace"] or ""
        return ""

    def _episode(self, ep: int) -> dict | None:
        if ep in self._cache:
            self._cache.move_to_end(ep)
            return self._cache[ep]
        raw = self.cache.read(ep)
        data = None if raw is None else episode_view(raw)
        self._cache[ep] = data
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def frame_inputs(self, ep: int, frame: int) -> dict[str, np.ndarray] | None:
        """Object inputs of the latest input frame at or before ``frame`` (None if uncached)."""
        if self.object_input == "none":
            return None
        d = self._episode(ep)
        prefix = "grid" if self.object_input == "grid" else "obj"
        if d is None or not len(d["in_frame_index"]):
            return None
        idx = d["in_frame_index"]
        j = int(np.searchsorted(idx, frame, side="right") - 1)
        if j < 0:
            return None
        feat = d[f"in_{prefix}_feat"][j].astype(np.float32)
        pose = d[f"in_{prefix}_pose"][j].astype(np.float32)
        valid = d["in_obj_valid"][j] if prefix == "obj" else np.ones(len(feat), bool)
        return {"obj_sam_feat": feat, "obj_pose_vec": pose, "obj_valid_mask": valid.astype(bool)}

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
        if self.object_input != "none":
            ins = [self.frame_inputs(int(e), int(f)) for e, f in zip(eps, frames, strict=True)]
            ref = next((i for i in ins if i is not None), None)
            if ref is not None:
                blank = {k: np.zeros_like(v) for k, v in ref.items()}
                for k in ref:
                    out[k] = torch.from_numpy(np.stack([(i or blank)[k] for i in ins]))
        if self.tokenizer is not None:
            texts = [self.trace(int(e), int(f)) for e, f in zip(eps, frames, strict=True)]
            if any(texts):
                enc = self.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=self.cot_max_tokens,
                                     truncation=True, add_special_tokens=False)
                out["cot_input_ids"] = enc.input_ids
                out["cot_attention_mask"] = enc.attention_mask.bool()
        return out


def latest_cache(root: str, dataset_id: str) -> str:
    """Name of the most recently created cache for ``dataset_id`` under ``root``."""
    import json
    import re
    from pathlib import Path

    base = Path(root) / re.sub(r"[^A-Za-z0-9._-]+", "__", dataset_id)
    found = sorted((json.loads(p.read_text()).get("created", ""), p.parent.name) for p in base.glob("*/manifest.json"))
    if not found:
        raise FileNotFoundError(f"no feature cache under {base}; run scripts/make_pseudolabels.py")
    return found[-1][1]


def episode_view(raw: dict) -> dict:
    """Cache rows of one episode -> per-episode arrays (targets on target frames, inputs on input frames)."""
    t = raw["has_target"].astype(bool)
    i = raw["has_input"].astype(bool)
    h, w = int(raw["ep.height"]), int(raw["ep.width"])
    d = {
        "frame_index": raw["frame"][t], "in_frame_index": raw["frame"][i],
        "target_phrase": raw["ep.phrase"], "instruction": raw["ep.instruction"],
        "phase_frames": np.asarray(raw["ep.phase_frames"], np.int64),
        "phase_kinds": np.asarray(raw["ep.phase_kinds"], np.int64),
        "obj_select_confidence": raw["ep.obj_select_confidence"], "height": h, "width": w,
    }
    for k in ("obj_box", "obj_feat", "grip_box", "rel_disp", "obj_motion", "obj_score", "grip_score", "obj_depth",
              "grip_depth", "rel_confidence", "motion_confidence"):
        d[k] = raw[k][t]
    d["tracks"] = np.transpose(raw["track_xy"][t], (1, 0, 2))  # (P, F, 2)
    d["track_visible"] = raw["track_visible"][t].T
    rows = np.flatnonzero(t)
    d["obj_mask"] = np.stack([mask_to_bits(rle_decode(raw["obj_mask_rle"][r], (h, w))) for r in rows]) if len(rows) else np.zeros((0, 128), np.uint8)
    for k in ("in_obj_feat", "in_obj_pose", "in_obj_valid", "in_grid_feat", "in_grid_pose"):
        d[k] = raw[k][i]
    return d


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


__all__ = ["SidecarTargets", "episode_view", "latest_cache", "split_future_frames"]
