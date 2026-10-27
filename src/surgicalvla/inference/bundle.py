"""Save, load, and run compact inference bundles.

Bundles contain model weights, configuration, normalization statistics, and
expert labels; optimizer state is excluded.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

BUNDLE_SUFFIX = ".svla"


def _config_from_dict(data: dict):
    """Rebuild a policy configuration from a stored object or mapping."""
    from surgicalvla.modules import SurgicalSmolVLAConfig

    if isinstance(data, SurgicalSmolVLAConfig):
        return data
    return SurgicalSmolVLAConfig(**data)


def _pack_safetensors(weights: dict) -> bytes:
    """Serialize tensors through a temporary safetensors file."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        staging = Path(tmp.name)
    try:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in weights.items()},
            str(staging),
            metadata={"format": "pt"},
        )
        return staging.read_bytes()
    finally:
        staging.unlink(missing_ok=True)


def _unpack_safetensors(blob: bytes, device: str) -> dict[str, torch.Tensor]:
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        staging = Path(tmp.name)
    try:
        staging.write_bytes(blob)
        return load_file(str(staging), device=device)
    finally:
        staging.unlink(missing_ok=True)


def save_inference_bundle(
    path: str | Path,
    policy,
    *,
    config: Any = None,
    dataset_stats: dict | None = None,
    step: int = 0,
    stage: str = "",
    expert_labels: list[str] | None = None,
    **meta: Any,
) -> Path:
    """Write weights, configuration, normalization statistics, and labels."""
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(BUNDLE_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(policy, dict):
        # bare state dict: an explicit `config` has to be supplied separately,
        # otherwise the bundle is weights-only and cannot rebuild a policy
        weights = policy
    else:
        weights, config = policy.state_dict(), policy.config

    names = expert_labels
    if names is None and not isinstance(policy, dict):
        names = getattr(policy, "expert_labels", None)
    if names is None and config is not None:
        names = getattr(config, "expert_labels", None)
    payload = {
        "weights": _pack_safetensors(weights),
        "config": config,
        "dataset_stats": dataset_stats,
        "expert_labels": list(names or []),
        "meta": {
            "step": step,
            "stage": stage,
            "created_at": datetime.now(UTC).isoformat(),
            **meta,
        },
    }
    torch.save(payload, path)
    logger.info(
        "wrote inference bundle %s (%.1f MB, %d tensors, step=%s)",
        path,
        path.stat().st_size / 1e6,
        len(weights),
        step,
    )
    return path


@dataclass
class InferenceBundle:
    """A loaded bundle. ``policy`` is only built on :meth:`load_policy`."""

    path: Path
    weights: dict[str, torch.Tensor] = field(repr=False)
    config: Any = None
    dataset_stats: dict | None = None
    expert_labels: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def step(self) -> int:
        return int(self.meta.get("step", 0))

    @property
    def stage(self) -> str:
        return str(self.meta.get("stage", ""))

    def load_policy(self, device: str = "cpu"):
        """Rebuild the policy and load the weights.

        Strict on purpose: at inference a missing key means the model is not the
        one that was trained, which is far better caught here than as silently
        degraded behaviour in the OR. Training-time loading stays lenient because
        it legitimately resumes across Stage A/B architecture changes.
        """
        if self.config is None:
            raise ValueError(f"bundle {self.path} has no config; cannot rebuild the policy")
        config = _config_from_dict(self.config)
        from surgicalvla.modules.policy import SurgicalSmolVLAPolicy

        policy = SurgicalSmolVLAPolicy(config)
        policy.load_state_dict(self.weights, strict=True)
        policy.to(device)
        policy.eval()
        return policy


def load_inference_bundle(path: str | Path, device: str = "cpu") -> InferenceBundle:
    path = Path(path)
    if not path.exists() and path.suffix == "":
        path = path.with_suffix(BUNDLE_SUFFIX)
    if not path.exists():
        raise FileNotFoundError(f"no inference bundle at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    weights = _unpack_safetensors(payload["weights"], device)
    # `expert_labels` is the canonical key; read the pre-rename `tool_names`
    # as a fallback so bundles written by earlier checkpoints still load.
    expert_labels = payload.get("expert_labels") or payload.get("tool_names") or []
    return InferenceBundle(
        path=path,
        weights=weights,
        config=payload.get("config"),
        dataset_stats=payload.get("dataset_stats"),
        expert_labels=list(expert_labels),
        meta=dict(payload.get("meta") or {}),
    )


@torch.no_grad()
def predict_actions(
    bundle: InferenceBundle,
    batch: dict,
    device: str = "cpu",
    policy=None,
) -> torch.Tensor:
    """Run one forward pass and return the ``(B, action_dim)`` action.

    This is a single committed action step, not the full action chunk:
    ``policy.select_action`` draws one sample from the chunk and holds the rest
    in an internal queue. Call it repeatedly (feeding each observation back in)
    to roll the policy out. For the whole chunk at once, forward the batch and
    take ``loss-free`` output from ``policy.forward`` instead.

    ``batch`` must already be in policy input space (images normalised to
    ``[0, 1]``); language tokens are added here. Use :func:`build_policy_batch`
    to assemble it.
    """
    from surgicalvla.training.trainer import _extract_current_frames, add_language_tokens

    policy = policy if policy is not None else bundle.load_policy(device)
    config = policy.config
    # Tokenise before the device move: add_language_tokens inserts freshly
    # tokenised CPU tensors, so moving first would leave observation.language.*
    # on the CPU and crash the embedding lookup on CUDA.
    batch = add_language_tokens(batch, config)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch = _extract_current_frames(batch, list(config.image_features))

    action = policy.select_action(batch)
    return action.detach().to(device)


def build_policy_batch(
    observations: dict[str, torch.Tensor],
    task: str | list[str],
    normalize: Callable | None = None,
) -> dict[str, torch.Tensor]:
    """Assemble a policy batch from a flat observation dict.

    ``observations`` maps camera keys (plus ``observation.state``) to ``(B, C, H, W)``
    / ``(B, state_dim)`` tensors in ``[0, 1]`` / raw units. ``normalize`` is an
    optional callable applied to the state tensor -- pass the normaliser that was
    fitted with the dataset stats stored in the bundle.
    """
    batch = dict(observations)
    batch["task"] = task
    if normalize is not None:
        state_key = "observation.state"
        if state_key in batch:
            batch[state_key] = normalize(batch[state_key])
    return batch


