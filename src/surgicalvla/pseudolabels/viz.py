"""Overlay videos of pseudo-labels: target mask (green), gripper box (red), visible point tracks (yellow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from surgicalvla.pseudolabels.sidecar import bits_to_mask


def overlay_frames(frames: np.ndarray, labels: dict) -> list[np.ndarray]:
    """``frames``: the labelled frames ``(F, H, W, 3)``; ``labels``: one episode's sidecar arrays."""
    out = []
    h, w = frames.shape[1:3]
    phrase = str(labels.get("target_phrase", ""))
    for j, frame in enumerate(frames):
        img = frame.copy()
        if "obj_mask" in labels:
            m = np.array(Image.fromarray(bits_to_mask(labels["obj_mask"][j]).astype(np.uint8) * 255).resize((w, h))) > 0
            img[m] = (0.5 * img[m] + np.array([0, 127, 0])).astype(np.uint8)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        if "grip_box" in labels and float(labels["grip_score"][j]) > 0:
            draw.rectangle(list(labels["grip_box"][j].astype(float) * [w, h, w, h]), outline=(255, 0, 0), width=2)
        if "tracks" in labels and len(labels["tracks"]):
            for p, v in zip(labels["tracks"][:, j], labels["track_visible"][:, j], strict=True):
                if v:
                    x, y = float(p[0]) * w, float(p[1]) * h
                    draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 255, 0))
        score = float(labels["obj_score"][j]) if "obj_score" in labels else 0.0
        draw.text((3, 3), f"{phrase} {score:.2f}", fill=(255, 255, 255))
        out.append(np.array(pil))
    return out


def write_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 10) -> Path:
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = w - w % 2, h - h % 2
        stream.pix_fmt = "yuv420p"
        for f in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[: stream.height, : stream.width]), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


__all__ = ["overlay_frames", "write_mp4"]
