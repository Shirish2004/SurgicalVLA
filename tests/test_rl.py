"""Unit tests for the online-RL math and wire format.

Deliberately free of LeRobot, torch checkpoints and the simulator: the advantage
estimator and the protocol are the parts that can be wrong in a way the loss
curve would never reveal.
"""

import numpy as np
import pytest
import torch

from surgicalvla.rl.grpo import episode_reward, group_advantages, group_weights, kl_penalty
from surgicalvla.rl.protocol import decode, encode, decode_array, encode_array, dumps, loads


# --------------------------------------------------------------------- advantages
def test_rloo_rewards_binary():
    # one success out of four: +1 vs the mean of the others (0.0)
    adv = group_advantages([1.0, 0.0, 0.0, 0.0])
    assert adv.shape == (4,)
    assert adv[0] == pytest.approx(1.0)
    assert adv[1:].tolist() == pytest.approx([-1 / 3] * 3)


def test_rloo_sums_to_zero():
    for rewards in ([1, 0, 1, 0, 0], [1, 1, 1], [0, 0, 0]):
        assert float(group_advantages(rewards).sum()) == pytest.approx(0.0, abs=1e-6)


def test_group_of_one_has_no_advantage():
    assert group_advantages([1.0]).tolist() == [0.0]


def test_std_normalization_only_with_three_or_more():
    # with n == 2 the std of "the others" is degenerate, so we keep the RLOO scale
    a = group_advantages([1.0, 0.0], normalize="std")
    b = group_advantages([1.0, 0.0], normalize="rloo")
    assert a.tolist() == pytest.approx(b.tolist())
    # with n == 3 it actually renormalizes
    c = group_advantages([1.0, 0.0, 0.0], normalize="std")
    assert float(c[0]) > 1.0


def test_advantages_are_clipped():
    adv = group_advantages([100.0, 0.0, 0.0], clip=5.0)
    assert float(adv.abs().max()) <= 5.0


def test_bad_normalize_raises():
    with pytest.raises(ValueError):
        group_advantages([1.0, 0.0], normalize="zscore")


# ------------------------------------------------------------------------ weights
def test_variation_uses_advantages():
    weights, mode = group_weights([1.0, 0.0])
    assert mode == "advantage"
    # leave-one-out on a group of two: each member is compared to the other one
    assert weights.tolist() == pytest.approx([1.0, -1.0])


def test_all_success_falls_back_to_rejection():
    # RLOO would be all zeros here; filtered BC still has something to learn from
    weights, mode = group_weights([1.0, 1.0, 1.0])
    assert mode == "rejection"
    assert weights.tolist() == [1.0, 1.0, 1.0]


def test_no_success_means_no_update():
    weights, mode = group_weights([0.0, 0.0])
    assert mode == "none"
    assert float(weights.abs().sum()) == 0.0


def test_fallback_can_be_disabled():
    _, mode = group_weights([1.0, 1.0], fallback=False)
    assert mode == "none"


# ------------------------------------------------------------------------- reward
def test_reward_is_one_on_success():
    assert episode_reward(True, 100, 500) == 1.0
    assert episode_reward(False, 100, 500) == 0.0


def test_efficiency_bonus_only_scales_success():
    assert episode_reward(True, 250, 500, efficiency_bonus=0.5) == pytest.approx(1.25)
    assert episode_reward(False, 250, 500, efficiency_bonus=0.5) == 0.0


def test_efficiency_bonus_never_negative():
    assert episode_reward(True, 900, 500, efficiency_bonus=0.5) == 1.0


def test_kl_penalty_is_one_sided():
    assert kl_penalty(1.0, 2.0, 0.1) == 0.0
    assert kl_penalty(3.0, 2.0, 0.1) == pytest.approx(0.1)
    assert kl_penalty(3.0, 2.0, 0.0) == 0.0


# ----------------------------------------------------------------------- protocol
def test_array_roundtrip_preserves_dtype_and_shape():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = decode_array(encode_array(arr))
    assert out.dtype == np.float32
    assert out.shape == (3, 4)
    assert np.array_equal(out, arr)


def test_nested_message_roundtrip():
    message = {
        "cmd": "reset",
        "ok": True,
        "obs": {
            "images": {"agentview": np.zeros((8, 8, 3), dtype=np.uint8)},
            "state": np.linspace(-1, 1, 7, dtype=np.float32),
            "language": "pick up the black bowl",
        },
        "n": 3,
        "none": None,
    }
    out = loads(dumps(message))
    assert out["obs"]["language"] == message["obs"]["language"]
    assert out["n"] == 3
    assert out["none"] is None
    assert out["obs"]["images"]["agentview"].dtype == np.uint8
    assert np.allclose(out["obs"]["state"], message["obs"]["state"])


def test_message_is_a_single_line():
    payload = dumps({"cmd": "step", "action": np.zeros(7, dtype=np.float32)})
    # exactly one terminating newline, so `for line in socket_file` frames cleanly
    assert payload.endswith(b"\n")
    assert payload.count(b"\n") == 1


def test_encode_decode_pass_through_plain_values():
    assert decode(encode({"a": 1, "b": "text", "c": [1, 2]})) == {"a": 1, "b": "text", "c": [1, 2]}


def test_tensor_like_float64_is_preserved():
    arr = np.array([1.0, 2.5], dtype=np.float64)
    assert decode_array(encode_array(arr)).dtype == np.float64
