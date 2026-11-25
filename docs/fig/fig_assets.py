"""Real example images for Fig. 1 (docs/fig/assets/): one LIBERO-Spatial frame and its pseudo-labels.

    pixi run -e dev python docs/fig/fig_assets.py

Reads episode EPISODE of lerobot/libero and its feature-cache rows, and writes, at frame T:
the third-person and wrist images, the third-person image with the cached SAM 3 target mask,
the object-token boxes (target candidates and end effector) and the point tracks, the frame
T + HORIZON that the latent tokens predict, and example.json (instruction, target phrase,
reasoning trace written from the pseudo-labels).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

EPISODE, T, HORIZON = 1261, 36, 10  # LIBERO-Spatial; frame T approaches the grasp at frame 47
CACHE = ("pseudolabels", "lerobot/libero", "v3-dfd05ccb2f89")
OUT = Path(__file__).with_name("assets")
SIZE = 256
CAND = (255, 140, 30)   # target-phrase candidates
GRIP = (60, 140, 220)   # end effector
MASK = np.array([255, 120, 20], np.float32)


def frame(ds, i: int, key: str) -> Image.Image:
    x = ds[i][key].permute(1, 2, 0).numpy()
    return Image.fromarray((x * 255).round().astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)


def box(draw, b, h, w, color, width=3):
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    if max(x1, y1, x2, y2) <= 1.0:  # normalised
        x1, x2, y1, y2 = x1 * SIZE, x2 * SIZE, y1 * SIZE, y2 * SIZE
    else:
        x1, x2, y1, y2 = x1 * SIZE / w, x2 * SIZE / w, y1 * SIZE / h, y2 * SIZE / h
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from surgicalvla.pseudolabels.cache import FeatureCache, rle_decode
    from surgicalvla.pseudolabels.cot import candidates
    from surgicalvla.pseudolabels.reader import episode_view

    raw = FeatureCache(*CACHE).read(EPISODE)
    view = episode_view(raw)
    h, w = view["height"], view["width"]
    ds = LeRobotDataset("lerobot/libero", episodes=[EPISODE])
    base = int(np.flatnonzero(np.asarray(ds.hf_dataset["frame_index"]) == 0)[0])
    OUT.mkdir(exist_ok=True)

    third = frame(ds, base + T, "observation.images.image")
    third.save(OUT / "third.jpg", quality=88)
    frame(ds, base + T, "observation.images.image2").save(OUT / "wrist.jpg", quality=88)
    frame(ds, base + T + HORIZON, "observation.images.image").save(OUT / "future.jpg", quality=88)

    # perception: object-token boxes on the current frame (inputs are past-only)
    rows_in = np.flatnonzero(raw["has_input"].astype(bool) & (raw["frame"] == T))
    pose, valid = raw["in_obj_pose"][rows_in[0]], raw["in_obj_valid"][rows_in[0]].astype(bool)
    per = third.copy()
    d = ImageDraw.Draw(per)
    for k in range(len(pose)):
        if valid[k]:
            box(d, pose[k], h, w, GRIP if k == len(pose) - 1 else CAND, width=3 if k in (0, len(pose) - 1) else 2)
    per.save(OUT / "objects.jpg", quality=88)

    # targets: SAM 3 target mask and point tracks over the next labelled frames
    rows_t = np.flatnonzero(raw["has_target"].astype(bool))
    j = int(np.flatnonzero(raw["frame"][rows_t] == T)[0])
    mask = np.asarray(Image.fromarray(rle_decode(raw["obj_mask_rle"][rows_t[j]], (h, w)).astype(np.uint8) * 255).resize((SIZE, SIZE)))
    img = np.asarray(third, np.float32)
    m = mask[..., None] > 127
    tgt = Image.fromarray(np.where(m, 0.45 * img + 0.55 * MASK, img).astype(np.uint8))
    d = ImageDraw.Draw(tgt)
    tracks, vis = view["tracks"], view["track_visible"]  # (P, F, 2) on target frames
    for p in range(tracks.shape[0]):
        sx, sy = (SIZE, SIZE) if float(np.max(tracks)) <= 1.0 else (SIZE / w, SIZE / h)  # normalised or pixels
        pts = [(float(x) * sx, float(y) * sy) for (x, y), v in zip(tracks[p, j:j + 6], vis[p, j:j + 6]) if v]
        if len(pts) >= 2:
            d.line(pts, fill=(255, 255, 255), width=2)
            d.ellipse([pts[-1][0] - 2.5, pts[-1][1] - 2.5, pts[-1][0] + 2.5, pts[-1][1] + 2.5], fill=(40, 200, 120))
    box(d, view["grip_box"][j], h, w, GRIP, width=2)
    tgt.save(OUT / "targets.jpg", quality=88)

    example = {"episode": EPISODE, "frame": T, "horizon": HORIZON, "instruction": view["instruction"],
               "phrase": view["target_phrase"], "trace": candidates(view, T)[1],
               "candidates_valid": int(valid[:-1].sum())}
    (OUT / "example.json").write_text(json.dumps(example, indent=1))
    print(example)


if __name__ == "__main__":
    main()
