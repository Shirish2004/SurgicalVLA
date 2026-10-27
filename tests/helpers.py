"""Construct a small SurgicalVLA policy for CPU/GPU smoke tests.

Loading the real SmolVLM2-500M backbone is slow, so tests that only need the
architecture (module wiring, state_dict contents, weight transfer) use a
narrow-layer config. Only the genuinely end-to-end forward test needs the full
backbone.
"""

from __future__ import annotations

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from surgicalvla.modules import SurgicalSmolVLAConfig


def tiny_surgical_config(
    *,
    cameras: tuple[str, ...] = ("observation.images.top", "observation.images.wrist"),
    action_dim: int = 6,
    state_dim: int = 6,
    enable_uncertainty_gate: bool = True,
    enable_visual_logging: bool = False,
    fixed_camera_keys: list[str] | None = None,
    num_experts: int = 4,
    moe_top_k: int = 2,
    n_obs_steps: int = 2,
    token_keep_threshold: float = 0.0,
    **overrides,
) -> SurgicalSmolVLAConfig:
    """A SurgicalSmolVLAConfig with a 2-layer VLM stack so construction is fast.

    The vision tower and action expert are still full size (they are what the
    prefix actually attends through); only the language stack is trimmed.

    ``n_obs_steps=2`` by default so the temporal domain-uncertainty loss has a
    previous frame to difference against; pass ``n_obs_steps=1`` explicitly for
    tests that want the single-frame shape.
    """
    config = SurgicalSmolVLAConfig(
        # trim the language stack only; the vision tower and action expert stay
        # full size since the prefix attends through them
        num_vlm_layers=2,
        num_expert_layers=2,
        # architecture under test
        enable_uncertainty_gate=enable_uncertainty_gate,
        enable_visual_logging=enable_visual_logging,
        chunk_size=4,
        n_action_steps=4,
        n_obs_steps=n_obs_steps,
        num_experts=num_experts,
        moe_top_k=moe_top_k,
        token_keep_threshold=token_keep_threshold,
        fixed_camera_keys=list(fixed_camera_keys) if fixed_camera_keys else [],
        **overrides,
    )

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(state_dim,)
        ),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
    }
    for key in cameras:
        config.input_features[key] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        )
    config.validate_features()
    return config


def dummy_batch(config: SurgicalSmolVLAConfig, batch_size: int = 2, seed: int = 0) -> dict:
    """A batch shaped exactly like what the policy's forward expects."""
    g = torch.Generator().manual_seed(seed)
    batch = {
        "observation.state": torch.randn(batch_size, config.n_obs_steps, config.max_state_dim, generator=g),
        "action": torch.randn(batch_size, config.chunk_size, config.max_action_dim, generator=g),
        "task": ["use the forceps to grasp the suture"] * batch_size,
    }
    for key in config.image_features:
        batch[key] = torch.rand(
            batch_size, config.n_obs_steps, 3, 480, 640, generator=g
        )
    return batch