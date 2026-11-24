"""Overlay videos of pseudo-labels: target mask (green), gripper box (red), visible point tracks (yellow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from surgicalvla.pseudolabels.sidecar import bits_to_mask


def draw_frame(frame: np.ndarray, labels: dict, j: int, *, trail: int = 0, lines: tuple[str, ...] = ()) -> np.ndarray:
    """Draw labelled frame ``j`` of one episode's sidecar arrays onto ``frame`` (H, W, 3) uint8.

    Target mask (green fill) and box (green), gripper box (red), the line from gripper to target
    (magenta), visible TAPNext++ points (yellow) with their motion over the next ``trail``
    labelled frames (cyan), and text: phrase, SAM 3 scores, relative depths, ``lines``.
    """
    h, w = frame.shape[:2]
    img = frame.copy()
    scale = np.array([w, h, w, h], dtype=float)
    obj_score = float(labels["obj_score"][j]) if "obj_score" in labels else 0.0
    if "obj_mask" in labels and obj_score > 0:
        m = np.array(Image.fromarray(bits_to_mask(labels["obj_mask"][j]).astype(np.uint8) * 255).resize((w, h))) > 0
        img[m] = (0.5 * img[m] + np.array([0, 127, 0])).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    grip_ok = "grip_box" in labels and float(labels["grip_score"][j]) > 0
    if "obj_box" in labels and obj_score > 0:
        draw.rectangle(list(labels["obj_box"][j].astype(float) * scale), outline=(0, 255, 0), width=1)
    if grip_ok:
        draw.rectangle(list(labels["grip_box"][j].astype(float) * scale), outline=(255, 0, 0), width=2)
    if grip_ok and obj_score > 0:
        gb, ob = labels["grip_box"][j].astype(float) * scale, labels["obj_box"][j].astype(float) * scale
        draw.line([(gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2, (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2], fill=(255, 0, 255), width=1)
    if "tracks" in labels and len(labels["tracks"]):
        k = min(j + trail, labels["tracks"].shape[1] - 1)
        for p in range(labels["tracks"].shape[0]):
            if not labels["track_visible"][p, j]:
                continue
            x, y = float(labels["tracks"][p, j, 0]) * w, float(labels["tracks"][p, j, 1]) * h
            if k > j and labels["track_visible"][p, k]:
                draw.line([x, y, float(labels["tracks"][p, k, 0]) * w, float(labels["tracks"][p, k, 1]) * h], fill=(0, 255, 255), width=1)
            draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 255, 0))
    text = [f"{labels.get('target_phrase', '')} obj {obj_score:.2f}"]
    if grip_ok:
        text.append(f"grip {float(labels['grip_score'][j]):.2f}")
    if "obj_depth" in labels:
        text.append(f"depth obj {float(labels['obj_depth'][j]):.2f} grip {float(labels['grip_depth'][j]):.2f}")
    y = 2
    for t in [*text, *lines]:
        draw.text((3, y), t, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
        y += 11
    return np.array(pil)


def overlay_frames(frames: np.ndarray, labels: dict) -> list[np.ndarray]:
    """``frames``: the labelled frames ``(F, H, W, 3)``; ``labels``: one episode's sidecar arrays."""
    return [draw_frame(frame, labels, j) for j, frame in enumerate(frames)]


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


__all__ = ["draw_frame", "overlay_frames", "write_mp4"]
