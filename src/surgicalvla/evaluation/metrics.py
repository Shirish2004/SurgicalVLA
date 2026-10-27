from __future__ import annotations

import torch
from torch import Tensor


def action_error(pred: Tensor, target: Tensor, action_is_pad: Tensor | None = None) -> dict[str, float]:
    diff = pred - target
    if action_is_pad is not None:
        keep = (~action_is_pad).unsqueeze(-1).float()
        diff = diff * keep
        denominator = keep.sum().clamp_min(1.0)
    else:
        denominator = torch.tensor(float(diff.numel()), device=diff.device)
    return {
        "action_l1": float(diff.abs().sum() / denominator),
        "action_mse": float(diff.square().sum() / denominator),
    }


def rollout_metrics(pred: Tensor, target: Tensor, success: Tensor | None = None) -> dict[str, float]:
    if pred.shape[-1] < 7:
        raise ValueError("rollout actions require XYZ, RPY, and gripper dimensions")
    error = (pred - target).abs().mean(dim=tuple(range(pred.ndim - 1)))
    output = {
        "position_xyz_mae": float(error[:3].mean()),
        "rotation_rpy_mae": float(error[3:6].mean()),
        "gripper_mae": float(error[6:].mean()),
    }
    if success is not None:
        output["task_success_rate"] = float(success.float().mean())
    return output


def expert_utilization(selected_experts: Tensor, num_experts: int = 3) -> dict[str, float]:
    counts = torch.bincount(selected_experts.flatten(), minlength=num_experts).float()
    fractions = counts / counts.sum().clamp_min(1.0)
    return {f"expert_{index}_utilization": float(value) for index, value in enumerate(fractions)}
