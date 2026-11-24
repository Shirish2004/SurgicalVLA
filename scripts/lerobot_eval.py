"""``lerobot-eval`` that also knows SurgicalVLA's ``smolvla_moe`` policy.

    python scripts/lerobot_eval.py <any lerobot-eval arguments>

LeRobot resolves a policy type through its draccus-registered config class and the sibling
``modeling_*`` module, so importing our config module is enough; the evaluation loop is
LeRobot's. Two additions, neither of which changes an episode:

* Policies with ``object_input`` get the frozen perception attached
  (``attach_live_perception``): object tokens are computed from the current frame, exactly as
  in the training cache. ``--policy.object_token_ablation=zero|shuffle`` probes their use.
* The length of every episode (first done step + 1) is written to
  ``<output_dir>/episode_lengths.json`` in the order LeRobot runs them (tasks in order,
  rollouts in order, environments in order), next to ``eval_info.json``.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import torch

import surgicalvla.modules.configuration_smolvla_moe  # noqa: F401  (registers "smolvla_moe")
from lerobot.scripts import lerobot_eval as _le

_lengths: list[int] = []
_lock = threading.Lock()
_rollout = _le.rollout
_make_policy = _le.make_policy


def _recording_rollout(*args, **kwargs):
    out = _rollout(*args, **kwargs)
    done = out["done"].to(int)
    first = torch.argmax(done, dim=1)
    ended = done.any(dim=1)
    lengths = torch.where(ended, first + 1, torch.full_like(first, done.shape[1]))
    with _lock:
        _lengths.extend(int(x) for x in lengths.tolist())
    return out


def _make_policy_with_perception(*args, **kwargs):
    policy = _make_policy(*args, **kwargs)
    cfg = getattr(policy, "config", None)
    if getattr(cfg, "object_input", "none") != "none" and hasattr(policy, "attach_live_perception"):
        from surgicalvla.perception.object_inputs import ObjectInputConfig, ObjectInputExtractor

        policy.attach_live_perception(ObjectInputExtractor(ObjectInputConfig(num_candidates=cfg.obj_candidates),
                                                           device=str(cfg.device or "cuda")))
    return policy


def _output_dir(argv: list[str]) -> Path | None:
    for a in argv:
        if a.startswith("--output_dir="):
            return Path(a.split("=", 1)[1])
    return None


def main() -> None:
    _le.rollout = _recording_rollout
    _le.make_policy = _make_policy_with_perception
    try:
        _le.main()
    finally:
        out = _output_dir(sys.argv[1:])
        if out is not None and _lengths:
            out.mkdir(parents=True, exist_ok=True)
            (out / "episode_lengths.json").write_text(json.dumps(_lengths))


if __name__ == "__main__":
    main()
