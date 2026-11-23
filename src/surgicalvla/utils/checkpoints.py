"""Find the newest saved policy inside a training output directory."""

from __future__ import annotations

from pathlib import Path


def latest_pretrained(path: Path) -> Path:
    """A training output dir -> its newest saved policy (lerobot-train or scripts/train.py layout)."""
    if (path / "config.json").is_file():
        return path
    for candidate in (path / "checkpoints" / "last" / "pretrained_model", path / "final"):
        if (candidate / "config.json").is_file():
            return candidate
    steps = sorted(
        (p for p in list(path.glob("checkpoints/*")) + list(path.glob("step_*")) if p.is_dir()),
        key=lambda p: int("".join(c for c in p.name if c.isdigit()) or -1),
    )
    for step in reversed(steps):
        for candidate in (step / "pretrained_model", step):
            if (candidate / "config.json").is_file():
                return candidate
    raise FileNotFoundError(f"no saved policy under {path}")
