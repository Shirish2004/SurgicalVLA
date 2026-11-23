"""ReinFlow-GRPO math: geodesic handover reward, group advantages, clipped surrogate."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from surgicalvla.modules.moe_layers import gaussian_entropy
from surgicalvla.modules.object_encoder import rotation_6d_to_matrix


def _pose_errors(object_pose_6d: Tensor, target_pose_6d: Tensor) -> tuple[Tensor, Tensor]:
    a = torch.as_tensor(object_pose_6d, dtype=torch.float32).reshape(9)
    b = torch.as_tensor(target_pose_6d, dtype=torch.float32).reshape(9)
    position = (a[:3] - b[:3]).norm()
    ra = rotation_6d_to_matrix(a[3:].unsqueeze(0))
    rb = rotation_6d_to_matrix(b[3:].unsqueeze(0))
    cos = ((ra.transpose(-1, -2) @ rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    angle = torch.acos(cos.clamp(-1.0, 1.0)).squeeze(0)
    return position, angle


def so3_geodesic_angle(object_pose_6d: Tensor, target_pose_6d: Tensor) -> float:
    _, angle = _pose_errors(object_pose_6d, target_pose_6d)
    return float(angle)


def compute_handover_reward(
    success: bool,
    object_pose_6d: Tensor | None = None,
    target_pose_6d: Tensor | None = None,
    *,
    success_reward: float = 1.0,
    position_weight: float = 0.3,
    orientation_weight: float = 0.3,
    position_scale: float = 0.05,
    orientation_scale: float = math.pi / 4.0,
    failure_penalty: float = 0.0,
) -> float:
    """``success_reward * 1[success] - failure_penalty * 1[fail] + shaped``.

    ``shaped = w_p exp(-|p - p*| / s_p) + w_r exp(-theta / s_r)`` with ``theta`` the
    SO(3) geodesic angle, so ``0 <= shaped <= w_p + w_r`` and it *increases* as the
    object approaches the target, for successes and failures alike. (Audit A5: the
    previous version subtracted ``shaped`` on failure, so a failed rollout that got
    closer to the target was rewarded *less*.) ``success_reward`` must exceed the
    maximum shaping so that no failure can out-rank a success inside a GRPO group.
    """
    if success_reward <= position_weight + orientation_weight:
        raise ValueError(
            f"success_reward={success_reward} must exceed position_weight + orientation_weight="
            f"{position_weight + orientation_weight}, otherwise a near-miss can outrank a success"
        )
    reward = float(success_reward) if success else -abs(float(failure_penalty))
    if object_pose_6d is None or target_pose_6d is None:
        return reward
    position, angle = _pose_errors(object_pose_6d, target_pose_6d)
    shaped = (
        position_weight * math.exp(-float(position) / max(position_scale, 1e-6))
        + orientation_weight * math.exp(-float(angle) / max(orientation_scale, 1e-6))
    )
    return reward + shaped


def group_normalized_advantages(rewards: Sequence[float], eps: float = 1e-8) -> Tensor:
    r = torch.as_tensor(list(rewards), dtype=torch.float32).reshape(-1)
    if r.numel() <= 1:
        return torch.zeros_like(r)
    return (r - r.mean()) / (r.std(unbiased=False) + eps)


def sequence_log_prob(step_log_probs: Tensor) -> Tensor:
    return step_log_probs.sum(dim=0) if step_log_probs.ndim == 2 else step_log_probs.reshape(-1)


def grpo_surrogate(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    *,
    ref_log_probs: Tensor | None = None,
    clip: float = 0.2,
    kl_coef: float = 0.0,
    entropy: Tensor | None = None,
    entropy_coef: float = 0.0,
) -> dict[str, Tensor]:
    new = sequence_log_prob(new_log_probs)
    old = sequence_log_prob(old_log_probs)
    adv = advantages.to(new.device, new.dtype).reshape(-1)
    log_ratio = (new - old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * adv
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    out = {
        "policy_loss": policy_loss,
        "ratio_mean": ratio.mean().detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip).float().mean().detach(),
    }
    loss = policy_loss
    if ref_log_probs is not None and kl_coef > 0:
        ref = sequence_log_prob(ref_log_probs).detach()
        diff = (ref - new).clamp(-20.0, 20.0)
        kl = (diff.exp() - diff - 1.0).mean()
        loss = loss + kl_coef * kl
        out["approx_kl"] = kl.detach()
    if entropy is not None and entropy_coef > 0:
        ent = entropy.to(new.device, new.dtype).reshape(-1).mean()
        loss = loss - entropy_coef * ent
        out["entropy"] = ent.detach()
    out["loss"] = loss
    return out


__all__ = [
    "compute_handover_reward",
    "gaussian_entropy",
    "grpo_surrogate",
    "group_normalized_advantages",
    "sequence_log_prob",
    "so3_geodesic_angle",
]
