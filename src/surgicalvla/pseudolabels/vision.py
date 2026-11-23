"""Object-centric vision targets from frozen models: SAM 3 tracks, DINOv2 features, relative depth.

For one camera stream of an episode (frames already subsampled by the caller):

1. SAM 3 video tracking with text prompts for the target noun phrase, its head noun, and the
   robot's end effector (``GRIPPER_PROMPTS``), with low detection thresholds so that weakly
   matching objects still become candidate tracks (simulator renderings often do not match
   the instruction's colour words). The gripper is the end-effector track with the highest
   mean score. Among the object candidates that do not mostly overlap the gripper, the target
   is the one maximising ``0.5 * text score + 0.5 * carried score``, where the carried score
   is how consistently the candidate moves with the gripper after the gripper closes: the
   manipulated object is the one that gets moved.
2. DINOv2 patch features, average-pooled inside the target mask.
3. Depth Anything V2 relative depth (larger = closer), normalised per frame to its
   5th-95th percentile range; medians inside the target and gripper masks.
4. Gripper-relative displacement of the target: image-plane centroid offset (normalised
   coordinates) and relative-depth difference.
5. Target motion: centroid displacement to the next labelled frame.

All models are frozen; nothing here reads simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from surgicalvla.pseudolabels.sidecar import mask_to_bits

# "gripper" / "robot gripper" are never detected by SAM 3 on LIBERO or RoboTwin frames, while
# these are (scores 0.5-0.8 on both); the best-scoring track per episode is used.
GRIPPER_PROMPTS = ("robot end effector", "robot hand")


@dataclass
class VisionConfig:
    sam_id: str = "facebook/sam3"
    dino_id: str = "facebook/dinov2-small"
    depth_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    gripper_prompts: tuple[str, ...] = GRIPPER_PROMPTS
    min_score: float = 0.1  # below this a frame's detection counts as missing
    det_threshold: float = 0.15  # SAM 3 video score_threshold_detection (model default 0.5)
    new_det_threshold: float = 0.2  # SAM 3 video new_det_thresh (model default 0.7)
    gripper_overlap: float = 0.5  # target candidates overlapping the gripper more than this are rejected
    dtype: str = "float16"
    # gripper reference point for the displacement target: "centroid" of the end-effector mask,
    # the mask pixel "nearest" to the target centroid, or the "lowest" mask pixels (image space)
    gripper_point: str = "nearest"


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return np.array([xs.mean() / w, ys.mean() / h])


def _carried_score(obj_c: list, grip_c: list, grasp_frame: int | None, min_step: float = 2e-3) -> float:
    """Mean cosine between object and gripper centroid steps after ``grasp_frame``, clipped to [0, 1].

    Steps where the gripper barely moves are ignored. Returns 0 without a grasp frame or
    without enough overlapping observations.
    """
    if grasp_frame is None:
        return 0.0
    cos = []
    for f in range(max(grasp_frame, 0), len(obj_c) - 1):
        if any(x is None for x in (obj_c[f], obj_c[f + 1], grip_c[f], grip_c[f + 1])):
            continue
        g = grip_c[f + 1] - grip_c[f]
        o = obj_c[f + 1] - obj_c[f]
        if np.linalg.norm(g) < min_step:
            continue
        cos.append(float(np.dot(o, g) / (np.linalg.norm(o) * np.linalg.norm(g) + 1e-9)))
    if len(cos) < 3:
        return 0.0
    return float(np.clip(np.mean(cos), 0.0, 1.0))


def gripper_point(mask: np.ndarray, target: np.ndarray | None, mode: str) -> np.ndarray | None:
    """Normalised ``[x, y]`` reference point of the end-effector mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    pts = np.stack([xs / w, ys / h], axis=1)
    if mode == "nearest" and target is not None:
        return pts[np.argmin(np.linalg.norm(pts - target, axis=1))]
    if mode == "lowest":
        return pts[ys >= np.quantile(ys, 0.95)].mean(axis=0)
    return pts.mean(axis=0)


def _box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(4)
    h, w = mask.shape
    return np.array([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h])


class VisionLabeler:
    def __init__(self, config: VisionConfig | None = None, device: str = "cuda"):
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForDepthEstimation,
            Sam3VideoModel,
            Sam3VideoProcessor,
        )

        self.cfg = config or VisionConfig()
        self.device = device
        self.dtype = getattr(torch, self.cfg.dtype)
        self.sam = Sam3VideoModel.from_pretrained(
            self.cfg.sam_id,
            torch_dtype=self.dtype,
            score_threshold_detection=self.cfg.det_threshold,
            new_det_thresh=self.cfg.new_det_threshold,
        ).to(device).eval()
        self.sam_proc = Sam3VideoProcessor.from_pretrained(self.cfg.sam_id)
        self.dino = AutoModel.from_pretrained(self.cfg.dino_id, torch_dtype=self.dtype).to(device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(self.cfg.dino_id)
        self.depth = AutoModelForDepthEstimation.from_pretrained(self.cfg.depth_id, torch_dtype=self.dtype).to(device).eval()
        self.depth_proc = AutoImageProcessor.from_pretrained(self.cfg.depth_id)

    # ------------------------------------------------------------------ SAM 3
    @torch.no_grad()
    def track(
        self, frames: np.ndarray, phrase: str, grasp_frame: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Target and gripper masks ``(F, H, W)``, per-frame scores ``(F,)`` and the target's selection confidence.

        ``grasp_frame`` is the first labelled frame after the gripper closes, if any; it enables
        the carried score.
        """
        n, h, w = frames.shape[:3]
        head = phrase.split()[-1]
        object_prompts = [phrase] if head == phrase else [phrase, head]
        session = self.sam_proc.init_video_session(video=list(frames), inference_device=self.device, dtype=self.dtype)
        self.sam_proc.add_text_prompt(session, [*object_prompts, *self.cfg.gripper_prompts])
        per_frame: list[dict] = [dict() for _ in range(n)]
        prompt_ids: dict[str, set[int]] = {}
        for out in self.sam.propagate_in_video_iterator(session):
            res = self.sam_proc.postprocess_outputs(session, out)
            f = int(out.frame_idx)
            for p, ids in res["prompt_to_obj_ids"].items():
                prompt_ids.setdefault(p, set()).update(int(i) for i in ids)
            for oid, score, mask in zip(res["object_ids"].tolist(), res["scores"].tolist(), res["masks"], strict=True):
                per_frame[f][int(oid)] = (float(score), mask.bool().cpu().numpy())

        def gather(oid: int | None) -> tuple[np.ndarray, np.ndarray]:
            masks = np.zeros((n, h, w), bool)
            scores = np.zeros(n, np.float32)
            if oid is not None:
                for f in range(n):
                    if oid in per_frame[f]:
                        scores[f], masks[f] = per_frame[f][oid]
            return masks, scores

        def mean_score(oid: int) -> float:
            return float(np.mean([per_frame[f][oid][0] if oid in per_frame[f] else 0.0 for f in range(n)]))

        object_ids = set().union(*(prompt_ids.get(p, set()) for p in object_prompts))
        grip_ids = set().union(*(prompt_ids.get(p, set()) for p in self.cfg.gripper_prompts)) - object_ids
        grip_id = max(grip_ids, key=mean_score, default=None)
        grip_masks, grip_scores = gather(grip_id)
        grip_c = [_centroid(m) for m in grip_masks]

        best, best_value = None, -1.0
        for oid in object_ids:
            masks, scores = gather(oid)
            overlap = [(m & g).sum() / m.sum() for m, g in zip(masks, grip_masks, strict=True) if m.any()]
            if overlap and float(np.median(overlap)) > self.cfg.gripper_overlap:
                continue  # this candidate is the gripper itself
            text = float(scores.mean())
            carried = _carried_score([_centroid(m) for m in masks], grip_c, grasp_frame)
            value = 0.5 * text + 0.5 * carried
            if value > best_value:
                best, best_value = oid, value
        obj_masks, obj_scores = gather(best)
        return obj_masks, obj_scores, grip_masks, grip_scores, max(best_value, 0.0)

    # ------------------------------------------------------------------ features / depth
    @torch.no_grad()
    def pooled_features(self, frames: np.ndarray, masks: np.ndarray, batch: int = 16) -> np.ndarray:
        feats = []
        for i in range(0, len(frames), batch):
            inputs = self.dino_proc(images=list(frames[i : i + batch]), return_tensors="pt").to(self.device)
            tokens = self.dino(pixel_values=inputs["pixel_values"].to(self.dtype)).last_hidden_state[:, 1:]  # drop CLS
            side = int(round(tokens.shape[1] ** 0.5))
            grid = tokens.reshape(tokens.shape[0], side, side, -1)
            m = torch.from_numpy(masks[i : i + batch]).to(self.device, torch.float32)[:, None]
            m = F.interpolate(m, size=(side, side), mode="area")[:, 0]  # fraction of each patch inside the mask
            denom = m.sum(dim=(1, 2)).clamp_min(1e-6)[:, None]
            feats.append(((grid.float() * m[..., None]).sum(dim=(1, 2)) / denom).cpu())
        return torch.cat(feats).numpy()

    @torch.no_grad()
    def relative_depth(self, frames: np.ndarray, batch: int = 16) -> np.ndarray:
        h, w = frames.shape[1:3]
        maps = []
        for i in range(0, len(frames), batch):
            inputs = self.depth_proc(images=list(frames[i : i + batch]), return_tensors="pt").to(self.device)
            pred = self.depth(pixel_values=inputs["pixel_values"].to(self.dtype)).predicted_depth.float()
            pred = F.interpolate(pred[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
            lo = torch.quantile(pred.flatten(1), 0.05, dim=1)[:, None, None]
            hi = torch.quantile(pred.flatten(1), 0.95, dim=1)[:, None, None]
            maps.append(((pred - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1).cpu())
        return torch.cat(maps).numpy()

    # ------------------------------------------------------------------ all targets
    def label(self, frames: np.ndarray, phrase: str, grasp_frame: int | None = None) -> dict[str, np.ndarray]:
        """All vision targets for ``frames`` ``(F, H, W, 3)`` uint8 and the target ``phrase``.

        ``grasp_frame``: first labelled frame after the gripper closes (enables motion-based
        target selection).
        """
        obj_m, obj_s, grip_m, grip_s, select_conf = self.track(frames, phrase, grasp_frame)
        obj_s[obj_s < self.cfg.min_score] = 0.0
        grip_s[grip_s < self.cfg.min_score] = 0.0
        obj_m[obj_s == 0] = False
        grip_m[grip_s == 0] = False
        depth = self.relative_depth(frames)
        feats = self.pooled_features(frames, obj_m)
        n = len(frames)
        obj_c = [_centroid(m) for m in obj_m]
        grip_c = [gripper_point(m, c, self.cfg.gripper_point) for m, c in zip(grip_m, obj_c, strict=True)]
        obj_d = np.array([np.median(depth[f][obj_m[f]]) if obj_m[f].any() else 0.0 for f in range(n)])
        grip_d = np.array([np.median(depth[f][grip_m[f]]) if grip_m[f].any() else 0.0 for f in range(n)])
        rel = np.zeros((n, 3))
        rel_conf = np.zeros(n)
        for f in range(n):
            if obj_c[f] is not None and grip_c[f] is not None:
                rel[f, :2] = obj_c[f] - grip_c[f]
                rel[f, 2] = obj_d[f] - grip_d[f]
                rel_conf[f] = min(obj_s[f], grip_s[f])
        motion = np.zeros((n, 2))
        motion_conf = np.zeros(n)
        for f in range(n - 1):
            if obj_c[f] is not None and obj_c[f + 1] is not None:
                motion[f] = obj_c[f + 1] - obj_c[f]
                motion_conf[f] = min(obj_s[f], obj_s[f + 1])
        f16 = np.float16
        return {
            "obj_box": np.stack([_box(m) for m in obj_m]).astype(f16),
            "obj_mask": np.stack([mask_to_bits(m) for m in obj_m]),
            "obj_score": obj_s.astype(f16),
            "obj_select_confidence": np.array(select_conf, f16),
            "obj_feat": feats.astype(f16),
            "grip_box": np.stack([_box(m) for m in grip_m]).astype(f16),
            "grip_score": grip_s.astype(f16),
            "obj_depth": obj_d.astype(f16),
            "grip_depth": grip_d.astype(f16),
            "rel_disp": rel.astype(f16),
            "rel_confidence": rel_conf.astype(f16),
            "obj_motion": motion.astype(f16),
            "motion_confidence": motion_conf.astype(f16),
            "_obj_masks_full": obj_m,  # consumed by the point tracker, not stored
            "_grip_masks_full": grip_m,  # used by validation, not stored
        }


__all__ = ["GRIPPER_PROMPTS", "VisionConfig", "VisionLabeler"]
