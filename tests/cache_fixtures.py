"""A small synthetic feature cache written through the real writer (scripts/make_pseudolabels.episode_rows)."""

from __future__ import annotations

import numpy as np

from scripts.make_pseudolabels import episode_rows
from surgicalvla.pseudolabels.cache import FeatureCache, write_traces

MANIFEST = {"labeler": {"stride": 2, "input_stride": 1, "causal_inputs": True, "object_inputs": {"num_candidates": 3}},
            "modules": {}, "config_hash": "test-hash"}


def make_cache(tmp_path, *, n: int = 20, feat: int = 8, episode: int = 3) -> FeatureCache:
    f = n // 2  # labelled frames (stride 2)
    h = w = 16
    obj_masks = np.zeros((f, h, w), bool)
    obj_masks[:, 4:8, 4:8] = True
    vis = {
        "obj_box": np.tile([0.25, 0.25, 0.5, 0.5], (f, 1)), "obj_feat": np.ones((f, feat)),
        "grip_box": np.tile([0.6, 0.6, 0.9, 0.9], (f, 1)), "rel_disp": np.zeros((f, 3)),
        "obj_motion": np.tile([0.01, 0.0], (f, 1)), "obj_score": np.full(f, 0.9), "grip_score": np.full(f, 0.7),
        "obj_depth": np.full(f, 0.4), "grip_depth": np.full(f, 0.5), "rel_confidence": np.full(f, 0.5),
        "motion_confidence": np.full(f, 0.8), "obj_select_confidence": 0.9,
        "_obj_masks_full": obj_masks, "_grip_masks_full": np.zeros((f, h, w), bool),
    }
    tracks = {"tracks": np.cumsum(np.full((4, f, 2), 0.01), axis=1), "track_visible": np.ones((4, f), bool)}
    k = 4
    inputs = {"obj_feat": np.arange(n)[:, None, None] * np.ones((n, k, feat)),  # feature = frame index
              "obj_pose": np.zeros((n, k, 8)), "obj_valid": np.ones((n, k), bool),
              "grid_feat": np.zeros((n, 4, feat)), "grid_pose": np.zeros((n, 4, 8))}
    cols = episode_rows(n, 2, 1, vis, tracks, inputs, k, feat, 32)
    cache = FeatureCache(tmp_path, "org/ds", "test", manifest=MANIFEST)
    cache.add({"episode": episode, "length": n, "instruction": "pick up the black bowl", "phrase": "black bowl",
               "phrase_confidence": 1.0, "height": h, "width": w, "obj_select_confidence": 0.9,
               "phase_frames": [6, 14], "phase_kinds": [0, 0], "phase_confidence": [1.0, 1.0]}, cols)
    cache.flush()
    write_traces(cache, [{"episode": episode, "segment": 1, "frame": 10, "trace": "<cot> phase: grasp </cot>",
                          "kept": True, "gain": 0.1, "token_ids": [1, 2]}])
    return cache
