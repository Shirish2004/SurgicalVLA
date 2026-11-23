"""Object-centric vision targets from frozen models: SAM 3 tracks, DINOv2 features, relative depth.

For one camera stream of an episode (frames already subsampled by the caller):

1. SAM 3 video tracking with text prompts for the target noun phrase and for the robot's
   end effector (``GRIPPER_PROMPTS``). The gripper is the end-effector track with the highest
   mean score; the target is the phrase's track with the highest mean score whose masks do
   not mostly overlap the gripper (SAM 3 sometimes returns the gripper for object phrases).
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
    min_score: float = 0.5  # below this a frame's detection counts as missing
    gripper_overlap: float = 0.5  # target candidates overlapping the gripper more than this are rejected
    dtype: str = "float16"


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return np.array([xs.mean() / w, ys.mean() / h])


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
        self.sam = Sam3VideoModel.from_pretrained(self.cfg.sam_id, torch_dtype=self.dtype).to(device).eval()
        self.sam_proc = Sam3VideoProcessor.from_pretrained(self.cfg.sam_id)
        self.dino = AutoModel.from_pretrained(self.cfg.dino_id, torch_dtype=self.dtype).to(device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(self.cfg.dino_id)
        self.depth = AutoModelForDepthEstimation.from_pretrained(self.cfg.depth_id, torch_dtype=self.dtype).to(device).eval()
        self.depth_proc = AutoImageProcessor.from_pretrained(self.cfg.depth_id)

    # ------------------------------------------------------------------ SAM 3
    @torch.no_grad()
    def track(self, frames: np.ndarray, phrase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Target and gripper masks ``(F, H, W)`` with per-frame scores ``(F,)``."""
        n, h, w = frames.shape[:3]
        session = self.sam_proc.init_video_session(video=list(frames), inference_device=self.device, dtype=self.dtype)
        self.sam_proc.add_text_prompt(session, [phrase, *self.cfg.gripper_prompts])
        per_frame: list[dict] = [dict() for _ in range(n)]
        prompt_ids: dict[str, set[int]] = {}
        for out in self.sam.propagate_in_video_iterator(session):
            res = self.sam_proc.postprocess_outputs(session, out)
            f = int(out.frame_idx)
            for p, ids in res["prompt_to_obj_ids"].items():
                prompt_ids.setdefault(p, set()).update(int(i) for i in ids)
            for oid, score, mask in zip(res["object_ids"].tolist(), res["scores"].tolist(), res["masks"], strict=True):
                per_frame[f][int(oid)] = (float(score), mask.bool().cpu().numpy())

        def best_track(ids: set[int], reject: np.ndarray | None = None) -> int | None:
            best, best_score = None, -1.0
            for oid in ids:
                scores = []
                for f in range(n):
                    if oid not in per_frame[f]:
                        scores.append(0.0)
                        continue
                    s, m = per_frame[f][oid]
                    if reject is not None and m.any() and (m & reject[f]).sum() / m.sum() > self.cfg.gripper_overlap:
                        s = 0.0
                    scores.append(s)
                mean = float(np.mean(scores))
                if mean > best_score:
                    best, best_score = oid, mean
            return best

        def gather(oid: int | None) -> tuple[np.ndarray, np.ndarray]:
            masks = np.zeros((n, h, w), bool)
            scores = np.zeros(n, np.float32)
            if oid is None:
                return masks, scores
            for f in range(n):
                if oid in per_frame[f]:
                    scores[f], masks[f] = per_frame[f][oid]
            return masks, scores

        grip_ids = set().union(*(prompt_ids.get(p, set()) for p in self.cfg.gripper_prompts))
        grip_masks, grip_scores = gather(best_track(grip_ids - prompt_ids.get(phrase, set())))
        obj_masks, obj_scores = gather(best_track(prompt_ids.get(phrase, set()), reject=grip_masks))
        return obj_masks, obj_scores, grip_masks, grip_scores

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
    def label(self, frames: np.ndarray, phrase: str) -> dict[str, np.ndarray]:
        """All vision targets for ``frames`` ``(F, H, W, 3)`` uint8 and the target ``phrase``."""
        obj_m, obj_s, grip_m, grip_s = self.track(frames, phrase)
        obj_s[obj_s < self.cfg.min_score] = 0.0
        grip_s[grip_s < self.cfg.min_score] = 0.0
        obj_m[obj_s == 0] = False
        grip_m[grip_s == 0] = False
        depth = self.relative_depth(frames)
        feats = self.pooled_features(frames, obj_m)
        n = len(frames)
        obj_c = [_centroid(m) for m in obj_m]
        grip_c = [_centroid(m) for m in grip_m]
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
        }


__all__ = ["GRIPPER_PROMPTS", "VisionConfig", "VisionLabeler"]
