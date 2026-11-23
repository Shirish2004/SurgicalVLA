"""Frozen zero-shot perception stack used by the live object stream."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from surgicalvla.modules.object_encoder import ObjectTrackOutput, rotation_6d_to_matrix

DEFAULT_QUERIES = ("object", "handle", "instrument", "tool", "gripper")


@dataclass
class Detection:
    boxes: Tensor  # (N, 4) xyxy in pixels
    scores: Tensor  # (N,)
    labels: list[str] = field(default_factory=list)


@dataclass
class Track:
    masks: Tensor  # (N, H, W) bool
    roi_features: Tensor  # (N, C_sam)
    boxes: Tensor  # (N, 4)
    labels: list[str] = field(default_factory=list)


@dataclass
class ObjectPose:
    bbox_xyxy: Tensor  # (N, 4) normalised
    translation: Tensor  # (N, 3)
    rotation_6d: Tensor  # (N, 6)

    def as_pose13(self) -> Tensor:
        return torch.cat([self.bbox_xyxy, self.translation, self.rotation_6d], dim=-1)


class PoseBackend(Protocol):
    def __call__(self, rgb: Tensor, depth: Tensor | None, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(translation (3,), rotation_6d (6,))`` for one mask."""


def to_pil(image: Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    x = image.detach().float().cpu()
    if x.min() < 0:
        x = (x.clamp(-1, 1) + 1) / 2
    return Image.fromarray((x.clamp(0, 1) * 255).byte().numpy())


def to_uint8(image: Tensor) -> np.ndarray:
    return np.array(to_pil(image))


def instruction_to_queries(instruction: str, extra: tuple[str, ...] = ()) -> list[str]:
    words = re.findall(r"[a-zA-Z]{3,}", instruction.lower())
    stop = {"the", "and", "with", "into", "from", "that", "this", "for", "left", "right", "then", "open", "close"}
    nouns = [w for w in words if w not in stop]
    queries = list(dict.fromkeys([*extra, *nouns[:4], *DEFAULT_QUERIES[:2]]))
    return queries


class OpenVocabDetector(nn.Module):
    """OWLv2 by default; any HF zero-shot detector (e.g. GroundingDINO) via ``model_name``."""

    def __init__(self, model_name: str = "google/owlv2-large-patch14-ensemble", device: str = "cuda", threshold: float = 0.15):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.threshold = float(threshold)
        self._kind = "grounding" if "grounding" in model_name.lower() else "owl"
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        if self._kind == "owl":
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            self._processor = Owlv2Processor.from_pretrained(self.model_name)
            object.__setattr__(self, "_model", Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device).eval())
        else:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            object.__setattr__(self, "_model", AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device).eval())

    @torch.inference_mode()
    def forward(self, image: Tensor, instruction: str, queries: tuple[str, ...] = ()) -> Detection:
        self._load()
        img = to_pil(image)
        text = instruction_to_queries(instruction, queries)
        inputs = self._processor(text=[text] if self._kind == "owl" else text, images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
        outputs = self._model(**inputs)
        if self._kind == "owl":
            result = self._processor.post_process_grounded_object_detection(
                outputs, threshold=self.threshold, target_sizes=[img.size[::-1]], text_labels=[text]
            )[0]
            boxes, scores, labels = result["boxes"], result["scores"], result.get("text_labels", result.get("labels", []))
        else:
            result = self._processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], box_threshold=self.threshold, text_threshold=0.2, target_sizes=[img.size[::-1]]
            )[0]
            boxes, scores, labels = result["boxes"], result["scores"], result.get("text_labels", [])
        order = scores.argsort(descending=True)
        return Detection(boxes[order].float(), scores[order].float(), [str(x) for x in np.asarray(labels)[order.cpu().numpy()].tolist()])


class Sam2MaskTracker(nn.Module):
    """Box-prompted SAM 2.1; the previous mask can seed the next frame (memory-bank stand-in)."""

    def __init__(self, model_name: str = "facebook/sam2.1-hiera-large", device: str = "cuda", multimask: bool = False):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.multimask = bool(multimask)
        self._model = None
        self._processor = None
        self._prev_masks: Tensor | None = None
        self._prev_embeddings: Any = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import Sam2Model, Sam2Processor

        self._processor = Sam2Processor.from_pretrained(self.model_name)
        object.__setattr__(self, "_model", Sam2Model.from_pretrained(self.model_name).to(self.device).eval())

    def reset(self) -> None:
        self._prev_masks = None
        self._prev_embeddings = None

    @torch.inference_mode()
    def forward(self, image: Tensor, det: Detection) -> Track:
        self._load()
        img = to_pil(image)
        boxes = det.boxes[:1] if det.boxes.numel() else torch.tensor([[0.0, 0.0, img.width - 1.0, img.height - 1.0]])
        boxes = boxes.to(self.device)
        inputs = self._processor(
            images=img,
            input_boxes=[[b.tolist() for b in boxes]],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
        outputs = self._model(**inputs, multimask_output=self.multimask)
        masks = self._processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )[0]
        if masks.ndim == 4:
            masks = masks[:, 0]
        masks = masks.bool()
        self._prev_masks = masks.detach().cpu()

        pixel_values = inputs.get("pixel_values")
        embeddings = self._model.get_image_embeddings(pixel_values) if pixel_values is not None else None
        self._prev_embeddings = embeddings
        feats = self._pool_features(embeddings, masks) if embeddings is not None else self._pixel_pool(img, masks)
        return Track(masks=masks, roi_features=feats, boxes=det.boxes[: masks.shape[0]], labels=det.labels[: masks.shape[0]])

    def _pool_features(self, embeddings: Any, masks: Tensor) -> Tensor:
        feat = embeddings[-1] if isinstance(embeddings, (list, tuple)) else embeddings
        feat = feat[0] if feat.ndim == 4 else feat
        c, h, w = feat.shape
        pooled = []
        for mask in masks:
            m = F.interpolate(mask[None, None].float(), size=(h, w), mode="nearest")[0, 0]
            denom = m.sum().clamp_min(1.0)
            pooled.append((feat * m[None]).sum(dim=(1, 2)) / denom)
        return torch.stack(pooled)

    @staticmethod
    def _pixel_pool(img: Image.Image, masks: Tensor) -> Tensor:
        x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        h, w = masks.shape[-2:]
        x = F.interpolate(x[None], size=(h, w), mode="bilinear", align_corners=False)[0]
        out = []
        for mask in masks:
            m = mask.float()
            out.append((x * m[None]).sum(dim=(1, 2)) / m.sum().clamp_min(1.0))
        return torch.stack(out) if out else torch.zeros(0, 3)


class DepthPcaPoseBackend:
    """Depth + mask centroid and PCA frame; a real 6-D estimate with no CAD model."""

    def __init__(self, fov_deg: float = 60.0, depth_scale: float = 1.0):
        self.fov = float(fov_deg)
        self.depth_scale = float(depth_scale)

    def __call__(self, rgb: Tensor, depth: Tensor | None, mask: Tensor) -> tuple[Tensor, Tensor]:
        h, w = mask.shape
        if depth is None:
            depth = torch.ones(h, w)
        depth = depth.detach().float().cpu().reshape(h, w) * self.depth_scale
        ys, xs = mask.nonzero(as_tuple=True)
        if ys.numel() < 8:
            return torch.zeros(3), torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        fx = fy = w / (2.0 * np.tan(np.deg2rad(self.fov) / 2.0))
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        z = depth[ys, xs]
        points = torch.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], dim=-1)
        center = points.mean(0)
        centered = points - center
        cov = centered.T @ centered / max(1, points.shape[0])
        _, vecs = torch.linalg.eigh(cov)
        r = vecs.flip(-1)
        if torch.det(r) < 0:
            r[:, -1] *= -1
        rot6 = torch.cat([r[:, 0], r[:, 1]])
        return center, rot6


class DepthEstimator(nn.Module):
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        object.__setattr__(self, "_model", DepthAnythingForDepthEstimation.from_pretrained(self.model_name).to(self.device).eval())

    @torch.inference_mode()
    def forward(self, image: Tensor) -> Tensor:
        self._load()
        img = to_pil(image)
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        depth = self._model(**inputs).predicted_depth
        return F.interpolate(depth[:, None], size=(img.height, img.width), mode="bicubic", align_corners=False)[0, 0]


class SurgicalPerceptionStack(nn.Module):
    """Detector + mask tracker + 6-D pose, exposed as the three live-wrapper callables."""

    def __init__(
        self,
        detector: OpenVocabDetector | None = None,
        tracker: Sam2MaskTracker | None = None,
        pose_backend: PoseBackend | None = None,
        depth_estimator: DepthEstimator | None = None,
        device: str = "cuda",
        num_objects: int = 2,
        sam_feat_dim: int = 256,
        keyframe_interval: int = 30,
        queries: tuple[str, ...] = (),
    ):
        super().__init__()
        self.detector = detector or OpenVocabDetector(device=device)
        self.tracker = tracker or Sam2MaskTracker(device=device)
        self.pose_backend = pose_backend or DepthPcaPoseBackend()
        self.depth_estimator = depth_estimator or DepthEstimator(device=device)
        self.device = device
        self.num_objects = num_objects
        self.sam_feat_dim = sam_feat_dim
        self.keyframe_interval = keyframe_interval
        self.queries = queries
        self._frame = 0
        self._cached_det: Detection | None = None

    def attach_to(self, wrapper: nn.Module) -> None:
        wrapper.attach_live(self.detect, self.track, self.pose)

    def reset(self) -> None:
        self._frame = 0
        self._cached_det = None
        self.tracker.reset()

    def detach(self) -> None:
        object.__setattr__(self.detector, "_model", None)
        object.__setattr__(self.tracker, "_model", None)
        if self.depth_estimator is not None:
            object.__setattr__(self.depth_estimator, "_model", None)
        torch.cuda.empty_cache()

    def detect(self, rgb: Tensor, instruction: str) -> Detection:
        if self._cached_det is None or self._frame % max(1, self.keyframe_interval) == 0:
            self._cached_det = self.detector(rgb, instruction, self.queries)
        self._frame += 1
        return self._cached_det

    def track(self, rgb: Tensor, det: Detection) -> Track:
        return self.tracker(rgb, det)

    def pose(self, rgb: Tensor, tracked: Track) -> ObjectPose:
        depth = None
        try:
            depth = self.depth_estimator(rgb)
        except Exception:
            depth = None
        t, r = [], []
        for mask in tracked.masks:
            ti, ri = self.pose_backend(rgb, depth, mask)
            t.append(ti)
            r.append(ri)
        bboxes = []
        h, w = tracked.masks.shape[-2:]
        for mask in tracked.masks:
            ys, xs = mask.nonzero(as_tuple=True)
            if ys.numel() == 0:
                bboxes.append(torch.tensor([0.0, 0.0, 0.0, 0.0]))
            else:
                bboxes.append(torch.tensor([xs.min() / w, ys.min() / h, xs.max() / w, ys.max() / h]))
        return ObjectPose(
            bbox_xyxy=torch.stack(bboxes).float() if bboxes else torch.zeros(0, 4),
            translation=torch.stack(t).float() if t else torch.zeros(0, 3),
            rotation_6d=torch.stack(r).float() if r else torch.zeros(0, 6),
        )

    @torch.inference_mode()
    def predict(self, rgb: Tensor, instruction: str) -> ObjectTrackOutput:
        det = self.detect(rgb, instruction)
        tracked = self.track(rgb, det)
        poses = self.pose(rgb, tracked)
        n = min(self.num_objects, tracked.roi_features.shape[0])
        feats = tracked.roi_features[:n]
        pose13 = poses.as_pose13()[:n]
        if n < self.num_objects:
            feats = torch.cat([feats, feats.new_zeros(self.num_objects - n, self.sam_feat_dim)])
            pose13 = torch.cat([pose13, pose13.new_zeros(self.num_objects - n, 13)])
        valid = torch.zeros(self.num_objects, dtype=torch.bool)
        valid[:n] = True
        return ObjectTrackOutput(feats[None], pose13[None], valid[None])


def build_live_stack(device: str | None = None, **kwargs) -> SurgicalPerceptionStack:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SurgicalPerceptionStack(device=device, **kwargs)


__all__ = [
    "DEFAULT_QUERIES",
    "DepthEstimator",
    "DepthPcaPoseBackend",
    "Detection",
    "ObjectPose",
    "OpenVocabDetector",
    "PoseBackend",
    "Sam2MaskTracker",
    "SurgicalPerceptionStack",
    "Track",
    "build_live_stack",
    "instruction_to_queries",
    "rotation_6d_to_matrix",
]
