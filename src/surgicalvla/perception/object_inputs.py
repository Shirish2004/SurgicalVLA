"""Past-only object inputs for the policy: one frame in, object descriptors out.

Each frame is processed on its own, with no memory and no look-ahead, so the features for
frame ``t`` depend on frame ``t`` only. The same code builds the offline training cache and
runs live inside the policy at evaluation, which keeps training and test inputs identical.

Per frame, SAM 3 (image mode) detects instances of the target noun phrase (and its head noun)
and of the end effector. The ``num_candidates`` best target instances that no end effector covers, plus
the ``num_grippers`` best distinct end-effector instances, become objects. Choosing among several
same-class candidates (two black bowls) is left to the policy, which sees the instruction.

Object descriptor:

* ``feat``: DINOv2-S patch features pooled inside the instance mask, ``(C,)``;
* ``pose``: ``[x1, y1, x2, y2, cx, cy, depth, score]``: box and mask centroid in normalised
  image coordinates, median relative depth (Depth Anything V2 Small, per-frame 5–95 %
  normalisation) inside the mask, and the SAM 3 score.

The grid control (``layout="grid"``) uses the same encoders without any object structure:
the four cells of a 2×2 grid, pooled the same way, with score 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

POSE_DIM = 8
GRIPPER_PROMPTS = ("robot end effector", "robot hand")


@dataclass(frozen=True)
class ObjectInputConfig:
    sam_id: str = "facebook/sam3"
    dino_id: str = "facebook/dinov2-small"
    depth_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    gripper_prompts: tuple[str, ...] = GRIPPER_PROMPTS
    num_candidates: int = 3  # target-phrase instances kept per frame
    num_grippers: int = 1  # end-effector instances kept per frame (2 for bimanual robots)
    score_threshold: float = 0.15  # SAM 3 instance score below which an instance is ignored
    gripper_overlap: float = 0.5  # target instances covered by the end effector more than this are dropped
    dtype: str = "float16"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["gripper_prompts"] = list(self.gripper_prompts)
        return d


# --------------------------------------------------------------------------- selection (pure)
def select_objects(instances: dict[str, list[tuple[float, np.ndarray]]], target_prompts: list[str],
                   cfg: ObjectInputConfig) -> tuple[list[tuple[float, np.ndarray]], list[tuple[float, np.ndarray]]]:
    """``instances``: prompt -> [(score, mask)] for ONE frame. Returns (targets, grippers).

    Grippers: the ``num_grippers`` highest-scoring end-effector instances, duplicates (found by
    both end-effector prompts, IoU > 0.5) merged. Targets: the ``num_candidates``
    highest-scoring phrase instances that no gripper covers, duplicates found by both the
    phrase and its head noun (IoU > 0.7) merged.
    """
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        return float((a & b).sum()) / max(float((a | b).sum()), 1.0)

    grips: list[tuple[float, np.ndarray]] = []
    for score, mask in sorted((i for p in cfg.gripper_prompts for i in instances.get(p, []) if i[0] >= cfg.score_threshold),
                              key=lambda i: -i[0]):
        if mask.any() and all(iou(mask, m) <= 0.5 for _, m in grips):
            grips.append((score, mask))
        if len(grips) == cfg.num_grippers:
            break
    cands = sorted((i for p in target_prompts for i in instances.get(p, []) if i[0] >= cfg.score_threshold),
                   key=lambda i: -i[0])
    kept: list[tuple[float, np.ndarray]] = []
    for score, mask in cands:
        area = mask.sum()
        if area == 0:
            continue
        if any((mask & g).sum() / area > cfg.gripper_overlap for _, g in grips):
            continue
        if any(iou(mask, m) > 0.7 for _, m in kept):
            continue
        kept.append((score, mask))
        if len(kept) == cfg.num_candidates:
            break
    return kept, grips


def describe(mask: np.ndarray, score: float, depth: np.ndarray) -> np.ndarray:
    """``[x1, y1, x2, y2, cx, cy, depth, score]`` of one instance (normalised coordinates)."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return np.zeros(POSE_DIM, np.float32)
    return np.array([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h,
                     xs.mean() / w, ys.mean() / h, float(np.median(depth[mask])), score], np.float32)


def grid_masks(h: int, w: int) -> list[np.ndarray]:
    out = []
    for gy in range(2):
        for gx in range(2):
            m = np.zeros((h, w), bool)
            m[gy * h // 2 : (gy + 1) * h // 2, gx * w // 2 : (gx + 1) * w // 2] = True
            out.append(m)
    return out


# --------------------------------------------------------------------------- extractor
class ObjectInputExtractor:
    """Frozen SAM 3 + DINOv2 + Depth Anything; ``__call__(frames, phrase)`` per frame, no state."""

    def __init__(self, config: ObjectInputConfig | None = None, device: str = "cuda", shared: dict | None = None):
        from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation, Sam3Model, Sam3Processor

        self.cfg = config or ObjectInputConfig()
        self.device = device
        self.dtype = getattr(torch, self.cfg.dtype)
        shared = shared or {}
        self.sam = Sam3Model.from_pretrained(self.cfg.sam_id, dtype=self.dtype).to(device).eval()
        self.sam_proc = Sam3Processor.from_pretrained(self.cfg.sam_id)
        self.dino = shared.get("dino") or AutoModel.from_pretrained(self.cfg.dino_id, dtype=self.dtype).to(device).eval()
        self.dino_proc = shared.get("dino_proc") or AutoImageProcessor.from_pretrained(self.cfg.dino_id)
        self.depth = shared.get("depth") or AutoModelForDepthEstimation.from_pretrained(self.cfg.depth_id, dtype=self.dtype).to(device).eval()
        self.depth_proc = shared.get("depth_proc") or AutoImageProcessor.from_pretrained(self.cfg.depth_id)
        self._text_cache: dict[tuple[str, ...], tuple] = {}

    @property
    def feat_dim(self) -> int:
        return int(self.dino.config.hidden_size)

    @torch.no_grad()
    def _text(self, prompts: tuple[str, ...]):
        if prompts not in self._text_cache:
            txt = self.sam_proc(text=list(prompts), return_tensors="pt").to(self.device)
            emb = self.sam.get_text_features(input_ids=txt["input_ids"], attention_mask=txt["attention_mask"])
            self._text_cache = {prompts: (emb, txt["attention_mask"])}  # one instruction at a time
        return self._text_cache[prompts]

    @torch.no_grad()
    def _detect(self, frame: np.ndarray, prompts: tuple[str, ...]) -> dict[str, list[tuple[float, np.ndarray]]]:
        emb, att = self._text(prompts)
        img = self.sam_proc(images=[frame], return_tensors="pt").to(self.device)
        ve = self.sam.get_vision_features(pixel_values=img["pixel_values"].to(self.dtype))
        n = len(prompts)

        def rep(x):
            if torch.is_tensor(x):
                return x.expand(n, *x.shape[1:])
            if isinstance(x, (tuple, list)):
                return type(x)(rep(v) for v in x)
            return x

        out = self.sam(vision_embeds=ve.__class__(**{k: rep(v) for k, v in ve.items()}), text_embeds=emb, attention_mask=att)
        h, w = frame.shape[:2]
        res = self.sam_proc.post_process_instance_segmentation(
            out, threshold=self.cfg.score_threshold, mask_threshold=0.5, target_sizes=[[h, w]] * n)
        return {p: [(float(s), m.bool().cpu().numpy()) for s, m in zip(r["scores"].float().tolist(), r["masks"], strict=True)]
                for p, r in zip(prompts, res, strict=True)}

    @torch.no_grad()
    def _dense(self, frames: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """DINOv2 patch grids ``(F, S, S, C)`` and relative depth maps ``(F, H, W)``."""
        h, w = frames.shape[1:3]
        inputs = self.dino_proc(images=list(frames), return_tensors="pt").to(self.device)
        tokens = self.dino(pixel_values=inputs["pixel_values"].to(self.dtype)).last_hidden_state[:, 1:].float()
        side = int(round(tokens.shape[1] ** 0.5))
        grids = tokens.reshape(len(frames), side, side, -1)
        inputs = self.depth_proc(images=list(frames), return_tensors="pt").to(self.device)
        pred = self.depth(pixel_values=inputs["pixel_values"].to(self.dtype)).predicted_depth.float()
        pred = F.interpolate(pred[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        lo = torch.quantile(pred.flatten(1), 0.05, dim=1)[:, None, None]
        hi = torch.quantile(pred.flatten(1), 0.95, dim=1)[:, None, None]
        depth = ((pred - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1).cpu().numpy()
        return grids, depth

    @staticmethod
    def _pool(grid: torch.Tensor, masks: list[np.ndarray]) -> np.ndarray:
        if not masks:
            return np.zeros((0, grid.shape[-1]), np.float32)
        side = grid.shape[0]
        m = torch.from_numpy(np.stack(masks)).to(grid.device, torch.float32)[:, None]
        m = F.interpolate(m, size=(side, side), mode="area")[:, 0]
        return ((grid[None] * m[..., None]).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1e-6)[:, None]).cpu().numpy()

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, phrase: str | None, batch: int = 8) -> dict[str, np.ndarray]:
        """``frames`` ``(F, H, W, 3)`` uint8 -> per-frame arrays (each frame independent).

        ``obj_feat (F, K, C)``, ``obj_pose (F, K, 8)``, ``obj_valid (F, K)`` with
        ``K = num_candidates + num_grippers`` (candidates first, end effectors last), and the grid control
        ``grid_feat (F, 4, C)``, ``grid_pose (F, 4, 8)``.
        """
        k = self.cfg.num_candidates + self.cfg.num_grippers
        phrase = (phrase or "").strip()
        head = phrase.split()[-1] if phrase else ""
        target_prompts = [p for p in dict.fromkeys([phrase, head]) if p]
        prompts = tuple([*target_prompts, *self.cfg.gripper_prompts])
        n, h, w = frames.shape[:3]
        c = self.feat_dim
        out = {"obj_feat": np.zeros((n, k, c), np.float16), "obj_pose": np.zeros((n, k, POSE_DIM), np.float16),
               "obj_valid": np.zeros((n, k), bool), "grid_feat": np.zeros((n, 4, c), np.float16),
               "grid_pose": np.zeros((n, 4, POSE_DIM), np.float16)}
        cells = grid_masks(h, w)
        for i in range(0, n, batch):
            chunk = frames[i : i + batch]
            grids, depth = self._dense(chunk)
            for j, frame in enumerate(chunk):
                f = i + j
                targets, grippers = select_objects(self._detect(frame, prompts), target_prompts, self.cfg)
                slots: list[tuple[int, float, np.ndarray]] = [(s, sc, m) for s, (sc, m) in enumerate(targets)]
                slots += [(self.cfg.num_candidates + g, sc, m) for g, (sc, m) in enumerate(grippers)]
                feats = self._pool(grids[j], [m for _, _, m in slots])
                for (slot, sc, m), ft in zip(slots, feats, strict=True):
                    out["obj_feat"][f, slot] = ft
                    out["obj_pose"][f, slot] = describe(m, sc, depth[j])
                    out["obj_valid"][f, slot] = True
                out["grid_feat"][f] = self._pool(grids[j], cells)
                out["grid_pose"][f] = np.stack([describe(m, 1.0, depth[j]) for m in cells])
        return out


__all__ = ["GRIPPER_PROMPTS", "POSE_DIM", "ObjectInputConfig", "ObjectInputExtractor", "describe", "grid_masks", "select_objects"]
