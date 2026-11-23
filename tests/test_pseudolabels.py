"""Pseudo-label building blocks that need no model weights."""

from __future__ import annotations

import numpy as np
import pytest

from surgicalvla.pseudolabels.nounphrase import extract_phrase, grounded, rule_phrase
from surgicalvla.pseudolabels.phases import GRIPPER_EVENT, SPEED_MINIMUM, gripper_events, phase_boundaries, phase_ids, speed_minima
from surgicalvla.pseudolabels.sidecar import SidecarStore, bits_to_mask, mask_to_bits
from surgicalvla.pseudolabels.tracks import grid_points_in_mask


# ----------------------------------------------------------------------------- phases
def test_gripper_events_find_clean_steps_with_full_confidence():
    g = np.r_[np.full(30, -1.0), np.full(40, 1.0), np.full(30, -1.0)]
    frames, conf = gripper_events(g)
    assert frames.tolist() == [30, 70]
    assert np.allclose(conf, 1.0)


def test_gripper_events_ignore_chatter_and_constant_signals():
    g = np.r_[np.full(30, 0.0), [1, 0, 1, 0], np.full(30, 1.0)]
    frames, _ = gripper_events(g, min_gap=5)
    assert len(frames) == 1
    assert len(gripper_events(np.ones(50))[0]) == 0


def test_speed_minima_at_pauses():
    # move, stop at t=50, move again
    t = np.arange(100)
    x = np.where(t < 50, t, 50 + (t - 50)).astype(float)
    x[45:56] = x[45]  # pause
    frames, conf = speed_minima(np.stack([x, np.zeros_like(x)], 1), window=3, min_separation=10)
    assert len(frames) >= 1 and np.all((frames >= 44) & (frames <= 57))
    assert conf.max() > 0.8


def test_phase_boundaries_merge_and_order():
    T = 120
    state = np.zeros((T, 7))
    state[:, 0] = np.concatenate([np.linspace(0, 1, 60), np.full(10, 1.0), np.linspace(1, 2, 50)])
    action = np.zeros((T, 7))
    action[:, 6] = np.where(np.arange(T) < 62, -1.0, 1.0)  # close near the pause
    pb = phase_boundaries(state, action, gripper_dims=[6], window=3, min_separation=10)
    assert np.all(np.diff(pb.frames) > 0)
    assert GRIPPER_EVENT in pb.kinds.tolist()
    # the pause's speed minimum coincides with the gripper event and is merged into it
    assert sum(abs(f - 62) <= 3 for f in pb.frames) == 1
    ids = phase_ids(T, pb.frames)
    assert ids[0] == 0 and ids[-1] == len(pb.frames)
    _ = SPEED_MINIMUM


# ----------------------------------------------------------------------------- sidecar
def test_mask_bits_roundtrip():
    m = np.zeros((480, 640), bool)
    m[100:300, 200:400] = True
    small = bits_to_mask(mask_to_bits(m))
    assert small.shape == (32, 32) and 0.1 < small.mean() < 0.2


def test_store_is_resumable_atomic_and_bounded(tmp_path):
    store = SidecarStore(tmp_path, "org/ds", config={"stride": 2}, max_gb=1e-9 * 4000)  # ~4 kB cap
    rng = np.random.default_rng(0)
    size = store.write(0, {"frame_index": np.arange(10), "x": rng.random(200)})
    assert store.done(0) and size > 0 and not list(store.dir.glob(".tmp_*"))
    np.testing.assert_array_equal(store.read(0)["frame_index"], np.arange(10))
    reopened = SidecarStore(tmp_path, "org/ds", config={"stride": 2}, max_gb=1e-9 * 4000)
    assert reopened.done(0) and not reopened.done(1)
    with pytest.raises(ValueError, match="different labeler config"):
        SidecarStore(tmp_path, "org/ds", config={"stride": 4})
    with pytest.raises(OSError, match="cap"):  # nothing uploaded, nothing evictable
        reopened.write(1, {"x": rng.random(2000)})
    reopened.mark_uploaded(0)
    reopened.write(1, {"x": rng.random(300)})  # evicts the uploaded episode 0
    assert not reopened.path(0).exists() and reopened.path(1).exists() and reopened.done(0)


# ----------------------------------------------------------------------------- noun phrases
@pytest.mark.parametrize(
    ("instruction", "phrase"),
    [
        ("pick up the black bowl between the plate and the ramekin and place it on the plate", "black bowl"),
        ("put both the alphabet soup and the tomato sauce in the basket", "alphabet soup"),
        ("Lift the bottle with narrow top head-up from the table", "bottle"),
        ("open the top drawer of the cabinet", "top drawer"),
        ("turn on the stove", "stove"),
        ("put the yellow and white mug in the microwave and close it", "yellow and white mug"),
        ("pick up the black bowl and the plate", "black bowl"),
    ],
)
def test_rule_phrase(instruction, phrase):
    assert rule_phrase(instruction) == phrase


def test_vlm_answer_must_be_grounded():
    assert grounded("black bowl", "pick up the black bowl")
    assert not grounded("red cup", "pick up the black bowl")
    vlm = lambda text: "black bowl"  # noqa: E731
    assert extract_phrase("pick up the black bowl and place it on the plate", vlm) == ("black bowl", 1.0)
    assert extract_phrase("pick up the black bowl", None) == ("black bowl", 0.5)
    assert extract_phrase("hello there", None) == (None, 0.0)
    # "put X on Y": the VLM answering Y must not override the rule's X
    assert extract_phrase("put the white mug on the plate", lambda t: "plate") == ("white mug", 0.4)
    assert extract_phrase("the scalpel please", lambda t: "scalpel") == ("scalpel", 0.7)


def test_grid_points_lie_inside_the_mask():
    m = np.zeros((100, 100), bool)
    m[20:40, 50:90] = True
    pts = grid_points_in_mask(m, 16)
    assert 0 < len(pts) <= 16
    assert m[pts[:, 1].round().astype(int), pts[:, 0].round().astype(int)].all()


def test_carried_score_prefers_the_object_that_moves_with_the_gripper():
    from surgicalvla.pseudolabels.vision import _carried_score

    grip = [np.array([0.1 + 0.02 * t, 0.5]) for t in range(20)]
    carried = [np.array([0.3, 0.6])] * 8 + [np.array([0.1 + 0.02 * t, 0.55]) for t in range(8, 20)]
    static = [np.array([0.7, 0.7])] * 20
    assert _carried_score(carried, grip, grasp_frame=8) > 0.9
    assert _carried_score(static, grip, grasp_frame=8) == 0.0
    assert _carried_score(carried, grip, grasp_frame=None) == 0.0
