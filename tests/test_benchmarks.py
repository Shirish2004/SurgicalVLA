"""Audit A10: RL setups must live entirely in one benchmark's state/action space."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from surgicalvla.rl.benchmarks import LIBERO, ROBOTWIN, BenchmarkMismatch, get_benchmark, validate_rl_setup

ROOT = Path(__file__).resolve().parents[1]
LIBERO_CAMS = ["observation.images.camera1", "observation.images.camera2"]
GOOD = dict(
    policy_state_dim=8,
    policy_action_dim=7,
    policy_cameras=LIBERO_CAMS,
    camera_map={"observation.images.camera1": "agentview", "observation.images.camera2": "eye_in_hand"},
    action_slice=[0, 7],
    stats_state_dim=8,
    stats_action_dim=7,
    dataset_id="lerobot/libero",
    env_gripper_dims=[6],
)


def test_valid_libero_setup_passes():
    validate_rl_setup(LIBERO, **GOOD)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (dict(policy_state_dim=14, policy_action_dim=14, stats_state_dim=14, stats_action_dim=14), "policy acts in 14-D"),
        (dict(action_slice=[0, 6]), "action_slice"),
        (dict(stats_state_dim=14), "stats describe a 14-D state"),
        (dict(dataset_id="lerobot/robotwin_unified"), "not a libero dataset"),
        (dict(camera_map={"observation.images.camera1": "agentview", "observation.images.camera2": "agentview"}), "feed several"),
        (dict(camera_map={"observation.images.camera1": "agentview"}), "does not map"),
        (dict(camera_map={"observation.images.camera1": "agentview", "observation.images.camera2": "head_camera"}), "does not have"),
        (dict(env_gripper_dims=[6, 13]), "gripper dims"),
    ],
)
def test_each_cross_benchmark_mismatch_is_rejected(override, message):
    with pytest.raises(BenchmarkMismatch, match=message):
        validate_rl_setup(LIBERO, **{**GOOD, **override})


def test_robotwin_stays_14d():
    validate_rl_setup(
        ROBOTWIN,
        policy_state_dim=14,
        policy_action_dim=14,
        policy_cameras=["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"],
        camera_map={
            "observation.images.cam_high": "head_camera",
            "observation.images.cam_left_wrist": "left_camera",
            "observation.images.cam_right_wrist": "right_camera",
        },
        action_slice=[0, 14],
        dataset_id="lerobot/robotwin_unified",
        env_gripper_dims=[6, 13],
    )
    with pytest.raises(BenchmarkMismatch):
        validate_rl_setup(ROBOTWIN, **GOOD)  # a LIBERO policy in RoboTwin


def test_pre_audit_config_is_rejected():
    """The exact pre-fix setup: RoboTwin aloha checkpoint + stats, sliced to 7-D, on LIBERO."""
    with pytest.raises(BenchmarkMismatch) as err:
        validate_rl_setup(
            LIBERO,
            policy_state_dim=14,
            policy_action_dim=14,
            policy_cameras=["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"],
            camera_map={
                "observation.images.cam_high": "agentview",
                "observation.images.cam_left_wrist": "eye_in_hand",
                "observation.images.cam_right_wrist": "agentview",
            },
            action_slice=[0, 7],
            stats_state_dim=14,
            stats_action_dim=14,
            dataset_id="lerobot/robotwin_unified",
            env_gripper_dims=[6],
        )
    text = str(err.value)
    for needle in ("14-D", "action_slice", "feed several", "not a libero dataset"):
        assert needle in text


def _load(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def _smolvla_libero_features():
    from huggingface_hub import try_to_load_from_cache

    cfg = yaml.safe_load((ROOT / "configs" / "reinflow_grpo.yaml").read_text())
    path = try_to_load_from_cache(cfg["checkpoint"], "config.json", revision=cfg["checkpoint_revision"])
    if not isinstance(path, str):
        pytest.skip("smolvla_libero config not cached")
    return json.loads(Path(path).read_text())


def test_reinflow_config_is_valid_for_its_checkpoint():
    cfg = _load("reinflow_grpo.yaml")
    assert cfg["benchmark"] == "libero"
    feats = _smolvla_libero_features()
    lc = cfg["libero"]
    validate_rl_setup(
        get_benchmark(cfg["benchmark"]),
        policy_state_dim=feats["input_features"]["observation.state"]["shape"][0],
        policy_action_dim=feats["output_features"]["action"]["shape"][0],
        policy_cameras=[k for k, v in feats["input_features"].items() if v["type"] == "VISUAL"],
        camera_map=lc["camera_map"],
        action_slice=lc["action_slice"],
        dataset_id="lerobot/libero",
        env_gripper_dims=lc["env_gripper_dims"],
    )
    assert lc["n_exec_steps"] == feats["n_action_steps"], "RL should execute chunks like the SFT eval"


def test_rl_configs_never_point_at_robotwin_checkpoints():
    for name in ("reinflow_grpo.yaml", "rl.yaml"):
        cfg = _load(name)
        ckpt = str(cfg.get("checkpoint") or "")
        assert "checkpoints/" not in ckpt, f"{name} still points at a RoboTwin checkpoint"
        stats = str((cfg.get("stats") or {}).get("path") or "")
        assert "robotwin" not in stats.lower(), f"{name} still uses RoboTwin stats"
        cams = list(cfg["libero"]["camera_map"].values())
        assert len(cams) == len(set(cams)), f"{name} duplicates a camera"


def test_reinflow_build_policy_refuses_robotwin_benchmark():
    import scripts.train_reinflow_grpo as trg

    with pytest.raises(BenchmarkMismatch, match="LIBERO simulator only"):
        trg.build_policy({"benchmark": "robotwin", "checkpoint": "unused"}, "cpu")
