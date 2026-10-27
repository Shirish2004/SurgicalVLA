"""Round-trip tests for the single-file inference bundle.

The bundle is the hand-off artifact between training and the robot, so these
check the properties that actually matter at that boundary: weights survive
byte-for-byte, the config survives the ``dataclasses.asdict`` round trip that
``torch.save`` cannot do on its own, and the expert labels survive so a consumer
can still interpret router output.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from surgicalvla.inference import load_inference_bundle, save_inference_bundle
from surgicalvla.inference.bundle import BUNDLE_SUFFIX


def _dummy_config() -> dict:
    return {
        "chunk_size": 4,
        "n_action_steps": 4,
        "num_experts": 3,
        "moe_top_k": 2,
    }


def test_config_survives_the_bundle_round_trip(tmp_path: Path) -> None:
    """The rebuilt policy must run with the hyperparameters it trained with, and
    with its ``PolicyFeature`` types intact -- ``image_features`` and the policy
    constructor both depend on them, so a lossy dict round trip would fail only
    at construction time on a consumer's machine."""
    from lerobot.configs.types import FeatureType, PolicyFeature

    from surgicalvla.inference.bundle import _config_from_dict
    from surgicalvla.modules import SurgicalSmolVLAConfig

    config = SurgicalSmolVLAConfig(**_dummy_config())
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.top": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        ),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config.validate_features()

    bundle = load_inference_bundle(
        save_inference_bundle(tmp_path / "cfg", {}, config=config)
    )
    assert bundle.config is not None
    rebuilt = _config_from_dict(bundle.config)

    assert isinstance(rebuilt.input_features["observation.state"], PolicyFeature)
    assert list(rebuilt.image_features) == ["observation.images.top"]
    for field in (
        "chunk_size",
        "num_experts",
        "moe_top_k",
        "max_state_dim",
        "max_action_dim",
        "n_obs_steps",
    ):
        assert getattr(rebuilt, field) == getattr(config, field), field


def test_bundle_round_trips_weights_config_stats_labels(tmp_path: Path) -> None:
    weights = {
        "model.camera_weighting.log_temperature": torch.tensor(0.1),
        "model.uncertainty_gate.p_dom.0.weight": torch.randn(4, 8),
        "model.scientific_moe.router.1.weight": torch.randn(3, 8),
    }
    stats = {"observation.state": {"mean": [0.0, 1.0], "std": [1.0, 2.0]}}

    path = save_inference_bundle(
        tmp_path / "model",  # no suffix on purpose: the helper should add it
        weights,
        dataset_stats=stats,
        step=4321,
        stage="surgical",
        expert_labels=["scalpel", "forceps"],
    )
    assert path.name == f"model{BUNDLE_SUFFIX}"
    assert path.exists()

    bundle = load_inference_bundle(path)
    assert bundle.step == 4321
    assert bundle.stage == "surgical"
    assert bundle.expert_labels == ["scalpel", "forceps"]
    assert bundle.dataset_stats == stats
    assert set(bundle.weights) == set(weights)
    for key, value in weights.items():
        assert torch.equal(bundle.weights[key], value), key


def test_bundle_expert_labels_fallback_to_legacy_tool_names(tmp_path: Path) -> None:
    """Old bundles stored the list as ``tool_names`` and no ``expert_labels``;
    loading them must surface the labels via the canonical ``expert_labels``."""
    path = save_inference_bundle(tmp_path / "legacy", {"w": torch.zeros(1)}, step=1)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["tool_names"] = ["scalpel"]
    payload.pop("expert_labels", None)
    torch.save(payload, path)

    bundle = load_inference_bundle(path)
    assert bundle.expert_labels == ["scalpel"]


def test_bundle_missing_config_is_an_explicit_error(tmp_path: Path) -> None:
    path = save_inference_bundle(tmp_path / "w", {"w": torch.zeros(1)})
    bundle = load_inference_bundle(path)
    with pytest.raises(ValueError, match="no config"):
        bundle.load_policy()


def test_bundle_missing_file_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_inference_bundle(tmp_path / "nope")


def test_bundle_append_suffix_on_load(tmp_path: Path) -> None:
    """Loading by bare stem should find the bundle, matching save's behaviour."""
    path = save_inference_bundle(tmp_path / "best", {"w": torch.zeros(1)}, step=7)
    assert load_inference_bundle(tmp_path / "best").step == 7
    assert load_inference_bundle(path).step == 7