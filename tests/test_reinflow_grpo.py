"""Tests for the ReinFlow-GRPO math (reward, advantages, clipped surrogate)."""

from __future__ import annotations

import math

import pytest
import torch

from surgicalvla.rl.reinflow_grpo import (
    compute_handover_reward,
    gaussian_entropy,
    group_normalized_advantages,
    grpo_surrogate,
    sequence_log_prob,
    so3_geodesic_angle,
)


def _rot6d(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([c, s, 0.0, -s, c, 0.0])


def test_identity_poses_have_zero_error():
    pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert so3_geodesic_angle(pose, pose) == pytest.approx(0.0, abs=1e-5)


def test_geodesic_angle_matches_planar_rotation():
    a = torch.cat([torch.zeros(3), _rot6d(0.0)])
    b = torch.cat([torch.zeros(3), _rot6d(math.pi / 2)])
    assert so3_geodesic_angle(a, b) == pytest.approx(math.pi / 2, abs=1e-4)


def test_handover_reward_success_and_shaping():
    target = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    perfect = compute_handover_reward(True, target, target)
    assert perfect == pytest.approx(1.6, abs=1e-5)

    far = target.clone()
    far[:3] = 1.0
    assert compute_handover_reward(True, far, target) < perfect
    near_fail = compute_handover_reward(False, target, target)
    far_fail = compute_handover_reward(False, far, target)
    # Audit A5: among failures, closer to the target must score higher,
    # and no failure may out-rank any success.
    assert far_fail < near_fail < compute_handover_reward(True, far, target)


def test_handover_reward_rejects_shaping_that_beats_success():
    with pytest.raises(ValueError, match="near-miss"):
        compute_handover_reward(True, success_reward=0.5, position_weight=0.3, orientation_weight=0.3)


def test_group_advantages_are_zscored():
    adv = group_normalized_advantages([1.0, 0.0, 0.0, 0.0])
    assert adv.mean().abs() < 1e-6
    assert adv.std(unbiased=False) == pytest.approx(1.0, abs=1e-5)
    assert adv[0] > 0 > adv[1]


def test_group_advantages_constant_rewards_are_zero():
    adv = group_normalized_advantages([2.0, 2.0, 2.0])
    assert torch.count_nonzero(adv) == 0


def test_sequence_log_prob_sums_steps():
    steps = torch.arange(12, dtype=torch.float32).reshape(3, 4) * -0.1
    torch.testing.assert_close(sequence_log_prob(steps), steps.sum(0))


def test_grpo_surrogate_clips_and_moves_toward_positive_advantage():
    old = torch.full((3, 2), -5.0)
    better = old + 0.01  # ratio > 1
    good = grpo_surrogate(better, old, torch.tensor([1.0, 1.0]))
    worse = grpo_surrogate(old - 0.01, old, torch.tensor([1.0, 1.0]))
    assert good["policy_loss"] < worse["policy_loss"]
    assert good["clip_fraction"].item() == 0.0

    far = grpo_surrogate(old + 1.0, old, torch.tensor([1.0, 1.0]))
    assert far["clip_fraction"].item() == 1.0


def test_grpo_surrogate_kl_and_entropy_terms():
    old = torch.full((3, 2), -5.0)
    new = old.clone().requires_grad_(True)
    out = grpo_surrogate(
        new,
        old,
        torch.zeros(2),
        ref_log_probs=old + 1.0,
        kl_coef=0.1,
        entropy=torch.zeros(2),
        entropy_coef=0.01,
    )
    assert torch.isfinite(out["loss"])
    assert out["approx_kl"].item() > 0
    out["loss"].backward()
    assert new.grad is not None


def test_gaussian_entropy_masks_padding():
    log_std = torch.zeros(1, 4, 3)
    full = gaussian_entropy(log_std)
    masked = gaussian_entropy(log_std, action_dim=2, action_is_pad=torch.tensor([[False, False, True, True]]))
    assert full.item() > masked.item() > 0


def test_step_ratio_has_lower_variance_than_sequence_ratio():
    """Audit A6: per-step ratios clip far less often than the K-step product."""
    torch.manual_seed(0)
    k, b = 5, 4096
    old = torch.zeros(k, b)
    new = old + 0.08 * torch.randn(k, b)  # small per-step policy drift
    adv = torch.ones(b)
    step = grpo_surrogate(new, old, adv, clip=0.2, ratio_level="step")
    seq = grpo_surrogate(new, old, adv, clip=0.2, ratio_level="sequence")
    assert step["clip_fraction"] < seq["clip_fraction"]


def test_step_and_sequence_ratio_share_gradient_at_ratio_one():
    old = torch.full((3, 2), -5.0)
    adv = torch.tensor([1.0, -0.5])
    grads = []
    for level in ("step", "sequence"):
        new = old.clone().requires_grad_(True)
        grpo_surrogate(new, old, adv, ratio_level=level)["policy_loss"].backward()
        grads.append(new.grad.clone())
    # step averages over K*B terms, sequence over B: same direction, scale 1/K
    torch.testing.assert_close(grads[0] * 3, grads[1])
