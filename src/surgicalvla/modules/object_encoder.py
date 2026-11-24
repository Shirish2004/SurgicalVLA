"""Object-centric 6-D perception interface (frozen tracker + trainable projector)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# SO(3) helpers (shared by the pose head, the RL reward and the tests)
# ---------------------------------------------------------------------------


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """6-D rotation -> SO(3) matrix (Gram-Schmidt)."""
    if d6.shape[-1] != 6:
        raise ValueError(f"expected 6-D rotations, got last dim {d6.shape[-1]}")
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    # Remove the component of a2 along b1, then normalise.
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_rotation_angle(r_a: Tensor, r_b: Tensor) -> Tensor:
    """SO(3) geodesic distance in radians."""
    rel = r_a.transpose(-1, -2) @ r_b
    cos = (rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    return torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def geodesic_rotation_loss(pred_6d: Tensor, target_6d: Tensor) -> Tensor:
    """Mean geodesic error between predicted and target 6-D rotations."""
    return geodesic_rotation_angle(rotation_6d_to_matrix(pred_6d), rotation_6d_to_matrix(target_6d)).mean()


# ---------------------------------------------------------------------------
# Trainable projector
# ---------------------------------------------------------------------------


def _fourier_features(x: Tensor, bands: int = 16) -> Tensor:
    """Sinusoidal encoding of every input scalar."""
    if bands < 1:
        raise ValueError("bands must be >= 1")
    freqs = torch.arange(bands, device=x.device, dtype=x.dtype)
    freqs = 2.0**freqs  # 1, 2, 4, ... rad
    angles = x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)


@dataclass
class ObjectFeatures:
    """Projector output."""

    tokens: Tensor  # (B, K_obj, D_vlm)
    token_mask: Tensor  # (B, K_obj) bool
    context: Tensor  # (B, D_exp)
    valid: Tensor  # (B, N_obj) bool


class ObjectCentric6DProjector(nn.Module):
    """Fuse SAM ROI features + 13-D pose into VLM-width tokens."""

    def __init__(
        self,
        hidden_size_vlm: int,
        hidden_size_expert: int,
        sam_feat_dim: int = 256,
        pose_dim: int = 13,
        num_tokens: int = 4,
        fourier_bands: int = 16,
        dropout: float = 0.15,
        jitter_translation: float = 0.01,
        jitter_rotation: float = 0.02,
        layout: str = "pooled",
    ):
        """``layout="pooled"``: all objects are averaged into ``num_tokens`` shared slots.
        ``layout="per_object"``: one token per object (``num_tokens`` = number of objects), so
        the policy can tell candidates apart; slot embeddings mark the object position.
        """
        super().__init__()
        if layout not in ("pooled", "per_object"):
            raise ValueError(f"unknown layout {layout!r}")
        self.layout = layout
        self.hidden_size_vlm = hidden_size_vlm
        self.hidden_size_expert = hidden_size_expert
        self.pose_dim = pose_dim
        self.num_tokens = num_tokens
        self.dropout_p = float(dropout)
        self.jitter_t = float(jitter_translation)
        self.jitter_r = float(jitter_rotation)

        hidden = hidden_size_expert
        self.sam_proj = nn.Sequential(
            nn.Linear(sam_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        fourier_dim = pose_dim * 2 * fourier_bands
        self.pose_proj = nn.Sequential(
            nn.Linear(fourier_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())
        self.token_head = nn.Linear(hidden, (1 if layout == "per_object" else num_tokens) * hidden_size_vlm)
        self.slot_emb = nn.Parameter(torch.zeros(num_tokens, hidden_size_vlm))
        self.context_head = nn.Linear(hidden, hidden_size_expert)
        nn.init.normal_(self.slot_emb, std=0.02)

    def forward(
        self,
        sam_roi_feat: Tensor | None,
        obj_pose_vec: Tensor | None,
        obj_valid_mask: Tensor | None = None,
    ) -> ObjectFeatures:
        if sam_roi_feat is None or obj_pose_vec is None:
            if sam_roi_feat is not None or obj_pose_vec is not None:
                raise ValueError("sam_roi_feat and obj_pose_vec must be provided together")
            return self.empty(sam_roi_feat, obj_pose_vec, obj_valid_mask)

        if sam_roi_feat.ndim != 3 or obj_pose_vec.ndim != 3:
            raise ValueError(
                "expected sam_roi_feat (B, N, C) and obj_pose_vec (B, N, 13), got "
                f"{tuple(sam_roi_feat.shape)} and {tuple(obj_pose_vec.shape)}"
            )
        if obj_pose_vec.shape[-1] != self.pose_dim:
            raise ValueError(f"expected {self.pose_dim}-D poses, got {obj_pose_vec.shape[-1]}")
        bsize, n_obj = obj_pose_vec.shape[:2]
        if obj_valid_mask is None:
            obj_valid_mask = torch.ones(bsize, n_obj, dtype=torch.bool, device=obj_pose_vec.device)
        else:
            obj_valid_mask = obj_valid_mask.to(device=obj_pose_vec.device, dtype=torch.bool)

        pose = obj_pose_vec
        if self.training:
            pose = self._jitter(pose, obj_valid_mask)
            obj_valid_mask = self._token_dropout(obj_valid_mask)

        sam_feat = self.sam_proj(sam_roi_feat.to(dtype=pose.dtype))
        pose_feat = self.pose_proj(_fourier_features(pose, self._bands()))
        fused = self.fuse(torch.cat([sam_feat, pose_feat], dim=-1))

        valid_f = obj_valid_mask.to(dtype=fused.dtype)
        fused = fused * valid_f[..., None]
        denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        any_valid = obj_valid_mask.any(dim=1, keepdim=True)
        context = self.context_head(fused)
        context = (context * valid_f[..., None]).sum(dim=1) / denom
        context = torch.where(any_valid, context, torch.zeros_like(context))
        if self.layout == "per_object":
            if n_obj != self.num_tokens:
                raise ValueError(f"per_object layout expects {self.num_tokens} objects, got {n_obj}")
            tokens = self.token_head(fused) + self.slot_emb  # (B, N, D)
            tokens = tokens * valid_f[..., None]
            return ObjectFeatures(tokens=tokens, token_mask=obj_valid_mask, context=context, valid=obj_valid_mask)
        tokens = self.token_head(fused).view(bsize, n_obj, self.num_tokens, self.hidden_size_vlm)
        tokens = tokens * valid_f[..., None, None]

        tokens = (tokens.sum(dim=1) / denom[..., None]).add(self.slot_emb)
        tokens = torch.where(any_valid[..., None], tokens, torch.zeros_like(tokens))
        token_mask = any_valid.expand(-1, self.num_tokens)
        return ObjectFeatures(tokens=tokens, token_mask=token_mask, context=context, valid=obj_valid_mask)

    # ------------------------------------------------------------------ internals
    def _bands(self) -> int:
        # ``fourier_dim = pose_dim * 2 * bands`` was used to size the first layer.
        return self.pose_proj[0].in_features // (self.pose_dim * 2)

    def _jitter(self, pose: Tensor, valid: Tensor) -> Tensor:
        if self.training is False or (self.jitter_t <= 0 and self.jitter_r <= 0):
            return pose
        noise = torch.zeros_like(pose)
        if self.pose_dim == 13:  # [box, translation, 6-D rotation]
            noise[..., 4:7] = torch.randn_like(pose[..., 4:7]) * self.jitter_t
            noise[..., 7:13] = torch.randn_like(pose[..., 7:13]) * self.jitter_r
        else:  # [box, centroid, depth, score]: jitter the geometry, never the score
            noise[..., :-1] = torch.randn_like(pose[..., :-1]) * self.jitter_t
        noise = noise * valid[..., None].to(noise.dtype)
        return pose + noise

    def _token_dropout(self, valid: Tensor) -> Tensor:
        if not self.training or self.dropout_p <= 0:
            return valid
        keep = torch.rand(valid.shape, device=valid.device) >= self.dropout_p
        # Always keep at least one object whenever one was valid, so a batch never
        # loses its only object by chance.
        dropped = valid & ~keep
        all_dropped = dropped.all(dim=1, keepdim=True) & valid.any(dim=1, keepdim=True)
        dropped = torch.where(all_dropped.expand_as(dropped), torch.zeros_like(dropped), dropped)
        return valid & ~dropped

    def empty(
        self,
        sam_roi_feat: Tensor | None = None,
        obj_pose_vec: Tensor | None = None,
        obj_valid_mask: Tensor | None = None,
    ) -> ObjectFeatures:
        """All-masked output for batches without objects."""
        ref = sam_roi_feat if sam_roi_feat is not None else obj_pose_vec
        if obj_valid_mask is not None:
            bsize = obj_valid_mask.shape[0]
            device = obj_valid_mask.device
            dtype = self.slot_emb.dtype
        elif ref is not None:
            bsize, device = ref.shape[0], ref.device
            dtype = self.slot_emb.dtype
        else:
            # Parameters define the device/dtype even with no tensor in the batch.
            bsize, device, dtype = 1, self.slot_emb.device, self.slot_emb.dtype
        return ObjectFeatures(
            tokens=torch.zeros(bsize, self.num_tokens, self.hidden_size_vlm, device=device, dtype=dtype),
            token_mask=torch.zeros(bsize, self.num_tokens, dtype=torch.bool, device=device),
            context=torch.zeros(bsize, self.hidden_size_expert, device=device, dtype=dtype),
            valid=torch.zeros(bsize, 1, dtype=torch.bool, device=device),
        )


# ---------------------------------------------------------------------------
# Frozen perception wrapper
# ---------------------------------------------------------------------------


@dataclass
class ObjectTrackOutput:
    """Tracker output passed to the projector."""

    sam_roi_feat: Tensor
    obj_pose_vec: Tensor
    obj_valid_mask: Tensor


class ZeroShotObjectTrackerWrapper(nn.Module):
    """Adapter from cached/mock/live object features to :class:`ObjectTrackOutput`."""

    MODES = ("passthrough", "cached", "mock", "live")

    def __init__(
        self,
        mode: str = "passthrough",
        sam_feat_dim: int = 256,
        pose_dim: int = 13,
        num_objects: int = 2,
        cache: Mapping[Any, ObjectTrackOutput] | None = None,
        detector: Callable[..., Any] | None = None,
        mask_tracker: Callable[..., Any] | None = None,
        pose_estimator: Callable[..., Any] | None = None,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.sam_feat_dim = int(sam_feat_dim)
        self.pose_dim = int(pose_dim)
        self.num_objects = int(num_objects)
        self.cache = dict(cache or {})
        # Live components are deliberately plain attributes (not submodules): they
        # are frozen external models and must not appear in the policy state dict.
        object.__setattr__(self, "detector", detector)
        object.__setattr__(self, "mask_tracker", mask_tracker)
        object.__setattr__(self, "pose_estimator", pose_estimator)

    def attach_live(
        self,
        detector: Callable[..., Any],
        mask_tracker: Callable[..., Any],
        pose_estimator: Callable[..., Any],
    ) -> None:
        object.__setattr__(self, "detector", detector)
        object.__setattr__(self, "mask_tracker", mask_tracker)
        object.__setattr__(self, "pose_estimator", pose_estimator)
        self.mode = "live"

    def set_cache(self, cache: Mapping[Any, ObjectTrackOutput]) -> None:
        self.cache = dict(cache)

    def forward(self, batch: dict[str, Any]) -> ObjectTrackOutput:
        if self._has_batch_features(batch):
            return self._from_batch(batch)
        if self.mode == "passthrough":
            return self.mock(batch)
        if self.mode == "cached":
            key = batch.get("obj_cache_key")
            if key is None:
                raise KeyError("cached object mode needs batch['obj_cache_key']")
            keys = key if isinstance(key, (list, tuple)) else [key]
            return self._stack([self.cache[k] for k in keys])
        if self.mode == "mock":
            return self.mock(batch)
        return self._live(batch)

    # ------------------------------------------------------------------ sources
    @staticmethod
    def _has_batch_features(batch: Mapping[str, Any]) -> bool:
        return "obj_sam_feat" in batch and "obj_pose_vec" in batch

    def _from_batch(self, batch: Mapping[str, Any]) -> ObjectTrackOutput:
        sam = batch["obj_sam_feat"]
        pose = batch["obj_pose_vec"]
        valid = batch.get("obj_valid_mask")
        if valid is None:
            valid = torch.ones(pose.shape[:2], dtype=torch.bool, device=pose.device)
        return ObjectTrackOutput(sam, pose, valid)

    def mock(self, batch: Mapping[str, Any]) -> ObjectTrackOutput:
        ref = next((v for v in batch.values() if isinstance(v, Tensor) and v.ndim >= 1), None)
        if ref is None:
            raise ValueError("mock object mode needs at least one tensor in the batch to infer batch size")
        bsize, device = ref.shape[0], ref.device
        return ObjectTrackOutput(
            sam_roi_feat=torch.zeros(bsize, self.num_objects, self.sam_feat_dim, device=device),
            obj_pose_vec=torch.zeros(bsize, self.num_objects, self.pose_dim, device=device),
            obj_valid_mask=torch.zeros(bsize, self.num_objects, dtype=torch.bool, device=device),
        )

    def _live(self, batch: Mapping[str, Any]) -> ObjectTrackOutput:
        if self.detector is None or self.mask_tracker is None or self.pose_estimator is None:
            raise RuntimeError(
                "live object mode requires detector/mask_tracker/pose_estimator; "
                "call attach_live() or use cached/mock/passthrough for offline training"
            )
        rgb = batch.get("observation.images.agentview")
        if rgb is None:
            rgb = next((v for k, v in batch.items() if "image" in k and isinstance(v, Tensor)), None)
        if rgb is None:
            raise KeyError("live object mode needs an RGB observation")
        instruction = batch.get("task", "")
        feats, poses, valids = [], [], []
        for i in range(rgb.shape[0]):
            det = self.detector(rgb[i], instruction)
            tracked = self.mask_tracker(rgb[i], det)
            pose = self.pose_estimator(rgb[i], tracked)
            feats.append(tracked.roi_features)
            poses.append(pose.as_pose13())
            valids.append(True)
        return ObjectTrackOutput(
            sam_roi_feat=torch.stack(feats),
            obj_pose_vec=torch.stack(poses),
            obj_valid_mask=torch.tensor(valids, device=rgb.device)[:, None]
            .expand(len(valids), self.num_objects)
            .clone(),
        )

    def _stack(self, outputs: list[ObjectTrackOutput]) -> ObjectTrackOutput:
        return ObjectTrackOutput(
            sam_roi_feat=torch.stack([o.sam_roi_feat for o in outputs]),
            obj_pose_vec=torch.stack([o.obj_pose_vec for o in outputs]),
            obj_valid_mask=torch.stack([o.obj_valid_mask for o in outputs]),
        )

    # ------------------------------------------------------------------ helpers
    def normalize_pose_vec(self, pose: Tensor) -> Tensor:
        """Convert a detector 13-D pose into the projector layout."""
        if pose.shape[-1] != self.pose_dim:
            raise ValueError(f"expected {self.pose_dim}-D pose, got {pose.shape[-1]}")
        out = pose.clone()
        xyxy = pose[..., :4]
        out[..., :4] = xyxy / xyxy.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        rot = rotation_6d_to_matrix(pose[..., 7:13])
        out[..., 7:13] = torch.cat([rot[..., 0], rot[..., 1]], dim=-1)
        return out


def build_object_tracker(config) -> ZeroShotObjectTrackerWrapper:
    """Build the tracker described by the config."""
    return ZeroShotObjectTrackerWrapper(
        mode=getattr(config, "obj_tracker_mode", "passthrough"),
        sam_feat_dim=getattr(config, "obj_sam_feat_dim", 256),
        pose_dim=getattr(config, "obj_pose_dim", 13),
        num_objects=getattr(config, "num_objects", 2),
    )


__all__ = [
    "ObjectCentric6DProjector",
    "ObjectFeatures",
    "ObjectTrackOutput",
    "ZeroShotObjectTrackerWrapper",
    "build_object_tracker",
    "geodesic_rotation_angle",
    "geodesic_rotation_loss",
    "rotation_6d_to_matrix",
]
