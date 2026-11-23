"""YAML + dotlist config helpers shared by the training entry points.

``python scripts/train_rl.py --config configs/rl.yaml run.lr=3e-6 rl.group_size=4``
has to mean the same thing as the same syntax on ``scripts/train.py``, so both
parse overrides through :func:`parse_dotlist` / :func:`merge_config`.
"""

from __future__ import annotations

import warnings
from typing import Any

import yaml


def parse_dotlist(pairs: list[str]) -> dict:
    """``["a.b=1", "a.c=[1,2]"]`` -> ``{"a": {"b": 1, "c": [1, 2]}}`` (values YAML-parsed)."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"override {pair!r} is not of the form key=value")
        node = overrides
        *path, leaf = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return overrides


def merge_config(base: dict, patch: dict) -> dict:
    """Recursive in-place merge of ``patch`` into ``base``; returns ``base``."""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge_config(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Read ``path`` and apply dotlist overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return merge_config(cfg, parse_dotlist(overrides or []))


def warn_unknown(keys, valid, where: str) -> None:
    """Warn (rather than crash) about keys a dataclass config cannot accept."""
    for key in keys:
        if key not in valid:
            warnings.warn(f"Unknown {where} config key '{key}', ignored.", stacklevel=2)
