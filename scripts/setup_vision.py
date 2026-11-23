"""Verify (and optionally download) the frozen perception models.

Usage:
    pixi run python scripts/setup_vision.py            # verify offline cache
    pixi run python scripts/setup_vision.py --download # allow network
    pixi run python scripts/setup_vision.py --synthetic-check
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

MODELS = {
    "detector": "google/owlv2-large-patch14-ensemble",
    "sam2": "facebook/sam2.1-hiera-large",
    "depth": "depth-anything/Depth-Anything-V2-Small-hf",
    "features": "facebook/dinov2-small",
}


def _available(repo: str, download: bool) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_files_only=not download)
        return True
    except Exception as exc:
        print(f"  missing: {exc}")
        return False


def _imports() -> bool:
    try:
        from transformers import (  # noqa: F401
            DepthAnythingForDepthEstimation,
            Owlv2ForObjectDetection,
            Owlv2Processor,
            Sam2Model,
            Sam2Processor,
        )
    except Exception as exc:
        print(f"transformers vision imports failed: {exc}")
        return False
    return True


def _synthetic() -> bool:
    from PIL import Image

    from surgicalvla.vision import SurgicalPerceptionStack

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack = SurgicalPerceptionStack(device=device, num_objects=2)
    image = torch.randint(0, 255, (3, 240, 320), dtype=torch.uint8).float() / 255.0
    try:
        det = stack.detect(image, "pick up the handle")
        tracked = stack.track(image, det)
        pose = stack.pose(image, tracked)
        print(
            f"  synthetic check ok: boxes={tuple(det.boxes.shape)} masks={tuple(tracked.masks.shape)} "
            f"feats={tuple(tracked.roi_features.shape)} pose={tuple(pose.as_pose13().shape)}"
        )
        return True
    except Exception as exc:
        print(f"  synthetic check failed: {exc}")
        return False
    finally:
        stack.detach()
        del Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="allow network access")
    ap.add_argument("--synthetic-check", action="store_true", help="run one forward pass on a random image")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "0" if args.download else "1")
    ok = _imports()
    for name, repo in MODELS.items():
        status = "ok" if _available(repo, args.download) else "missing"
        print(f"{name:9s} {repo:48s} {status}")
        ok = ok and status == "ok"
    if args.synthetic_check and ok:
        ok = _synthetic()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
