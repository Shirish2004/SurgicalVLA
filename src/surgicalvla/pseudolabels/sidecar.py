"""Sidecar store for pseudo-labels: one compressed file per episode plus a manifest.

Layout::

    <root>/<dataset_id>/
        manifest.json            schema version, dataset, revision, labeler config, per-episode status
        ep_000123.npz            arrays for episode 123 (see SCHEMA)

The store is resumable (episodes recorded in the manifest are skipped), and its disk use
is bounded by ``max_gb``: when a write would exceed the cap, episodes already pushed to
the Hub are evicted, least recently written first; if none can be evicted the write
fails instead of growing past the cap. Writes are atomic (temporary file + rename).

Sidecars are keyed by ``episode_index`` and carry their own ``frame_index`` array, so
they align with any LeRobot dataset regardless of how it is stored or streamed.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

# name -> (description, per-frame?)
SCHEMA = {
    "frame_index": ("labelled frame indices within the episode, (F,) int32", True),
    "camera": ("camera key the vision targets come from", False),
    "instruction": ("episode instruction", False),
    "target_phrase": ("noun phrase of the manipulated object", False),
    "phrase_confidence": ("confidence of the noun phrase in [0, 1]", False),
    "obj_box": ("target box, normalised xyxy, (F, 4) float16", True),
    "obj_mask": ("target mask at 32x32, bit-packed, (F, 128) uint8", True),
    "obj_score": ("segmentation confidence of the target, (F,) float16", True),
    "obj_feat": ("mask-pooled DINOv2 feature of the target, (F, C) float16", True),
    "grip_box": ("gripper box, normalised xyxy, (F, 4) float16", True),
    "grip_score": ("segmentation confidence of the gripper, (F,) float16", True),
    "obj_depth": ("median relative depth inside the target mask, (F,) float16", True),
    "grip_depth": ("median relative depth inside the gripper mask, (F,) float16", True),
    "rel_disp": ("target minus gripper: (x, y) in normalised image units and relative depth, (F, 3) float16", True),
    "rel_confidence": ("confidence of rel_disp, (F,) float16", True),
    "obj_motion": ("target displacement to the next labelled frame, (F, 2) float16", True),
    "motion_confidence": ("confidence of obj_motion, (F,) float16", True),
    "tracks": ("point tracks on the target, (P, F, 2) float16 normalised xy", False),
    "track_visible": ("point visibility, (P, F) bool", False),
    "phase_frames": ("phase boundary frames, (K,) int32", False),
    "phase_kinds": ("0 = gripper event, 1 = speed minimum, (K,) int8", False),
    "phase_confidence": ("boundary confidence, (K,) float16", False),
}


def mask_to_bits(mask: np.ndarray, size: int = 32) -> np.ndarray:
    """Downsample a boolean mask to ``size x size`` (nearest) and bit-pack it."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    ys = (np.arange(size) * h / size + h / (2 * size)).astype(int).clip(0, h - 1)
    xs = (np.arange(size) * w / size + w / (2 * size)).astype(int).clip(0, w - 1)
    return np.packbits(mask[np.ix_(ys, xs)].reshape(-1))


def bits_to_mask(bits: np.ndarray, size: int = 32) -> np.ndarray:
    return np.unpackbits(np.asarray(bits, dtype=np.uint8))[: size * size].reshape(size, size).astype(bool)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name)


class SidecarStore:
    def __init__(self, root: str | Path, dataset_id: str, *, revision: str | None = None, config: dict | None = None, max_gb: float = 20.0):
        self.dir = Path(root) / _safe(dataset_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * 1024**3)
        self.manifest_path = self.dir / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{self.manifest_path}: schema {self.manifest.get('schema_version')} != {SCHEMA_VERSION}")
            if config is not None and self.manifest.get("config") != config:
                raise ValueError(f"{self.manifest_path} was written with a different labeler config; use a new root")
        else:
            self.manifest = {
                "schema_version": SCHEMA_VERSION,
                "dataset": dataset_id,
                "revision": revision,
                "config": config,
                "episodes": {},
            }
            self._save_manifest()

    # ------------------------------------------------------------------ manifest
    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1))
        os.replace(tmp, self.manifest_path)

    def done(self, episode: int) -> bool:
        return str(int(episode)) in self.manifest["episodes"]

    def path(self, episode: int) -> Path:
        return self.dir / f"ep_{int(episode):06d}.npz"

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob("ep_*.npz"))

    # ------------------------------------------------------------------ write / read
    def write(self, episode: int, arrays: dict, stats: dict | None = None) -> int:
        """Atomically store one episode; returns its size in bytes."""
        tmp = self.dir / f".tmp_ep_{int(episode):06d}.npz"
        np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
        size = tmp.stat().st_size
        self._make_room(size)
        os.replace(tmp, self.path(episode))
        self.manifest["episodes"][str(int(episode))] = {"bytes": size, "written": time.time(), "uploaded": False, **(stats or {})}
        self._save_manifest()
        return size

    def read(self, episode: int) -> dict[str, np.ndarray]:
        with np.load(self.path(episode), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def mark_uploaded(self, episode: int) -> None:
        self.manifest["episodes"][str(int(episode))]["uploaded"] = True
        self._save_manifest()

    def _make_room(self, incoming: int) -> None:
        used = self.disk_bytes()
        if used + incoming <= self.max_bytes:
            return
        evictable = sorted(
            ((int(e), info) for e, info in self.manifest["episodes"].items() if info.get("uploaded") and self.path(int(e)).exists()),
            key=lambda x: x[1]["written"],
        )
        for episode, _ in evictable:
            used -= self.path(episode).stat().st_size
            self.path(episode).unlink()
            if used + incoming <= self.max_bytes:
                return
        raise OSError(
            f"pseudo-label store {self.dir} would exceed its {self.max_bytes / 1024**3:.1f} GB cap "
            "and has no uploaded episodes to evict; raise --max-gb or push to the Hub"
        )


__all__ = ["SCHEMA", "SCHEMA_VERSION", "SidecarStore", "bits_to_mask", "mask_to_bits"]
