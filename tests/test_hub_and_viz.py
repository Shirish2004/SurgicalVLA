"""Hub model names and pseudo-label overlays on logged frames."""

from __future__ import annotations

import numpy as np

from surgicalvla.pseudolabels.sidecar import mask_to_bits
from surgicalvla.pseudolabels.viz import draw_frame
from surgicalvla.utils.hub import model_name, repo_id

SFT = {"benchmark": "libero", "run": {"seed": 1}, "model": {"policy_type": "smolvla_moe"},
       "smolvla_moe": {"moe_variant": "matched", "aux_vision": True, "latent_cot": True, "use_cot": True},
       "aux": {"cot_traces": False}, "dataset": {"episodes": "configs/data/libero_libero_spatial_episodes.json"},
       "hub": {"push": True, "namespace": "Shirish24"}}


def test_sft_name_follows_the_run_settings():
    assert model_name(SFT, "sft") == "libero-sft-matched-aux-lcot-spatial-s1"
    assert repo_id(SFT, "sft") == "Shirish24/surgical-vla-libero-sft-matched-aux-lcot-spatial-s1"


def test_rl_name_has_mode_and_suite():
    cfg = {"benchmark": "libero", "run": {"seed": 0}, "training": {"mode": "joint"}, "env": {"suite": "libero_10"},
           "hub": {"push": True}}
    assert repo_id(cfg, "rl") == "Shirish24/surgical-vla-libero-rl-joint-libero10-s0"
    cfg["env"]["suite"] = "a,b"
    assert model_name(cfg, "rl") == "libero-rl-joint-all-s0"


def test_template_and_push_off():
    cfg = {**SFT, "hub": {"push": True, "model_name": "{benchmark}-{stage}-{variant}-Test_A-s{seed}"}}
    assert model_name(cfg, "sft") == "libero-sft-matched-test-a-s1"
    assert repo_id({**SFT, "hub": {"push": False}}, "sft") is None


def test_draw_frame_marks_mask_boxes_and_tracks():
    mask = np.zeros((64, 64), bool)
    mask[40:56, 40:56] = True
    labels = {
        "target_phrase": np.array("bowl"), "obj_score": np.array([0.9]), "obj_mask": mask_to_bits(mask)[None],
        "obj_box": np.array([[0.6, 0.6, 0.9, 0.9]]), "grip_box": np.array([[0.1, 0.6, 0.3, 0.95]]),
        "grip_score": np.array([0.8]), "obj_depth": np.array([0.4]), "grip_depth": np.array([0.5]),
        "tracks": np.array([[[0.2, 0.8]]]), "track_visible": np.array([[True]]),
    }
    frame = np.zeros((64, 64, 3), np.uint8)
    out = draw_frame(frame, labels, 0, lines=("phase 1",))
    assert out.shape == frame.shape and out[53, 53, 1] > 0  # green mask fill
    assert ((out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 0)).any()  # red gripper box
    assert ((out[..., 0] == 255) & (out[..., 1] == 255) & (out[..., 2] == 0)).any()  # yellow track point
