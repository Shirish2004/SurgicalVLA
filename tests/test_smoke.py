import torch

from surgicalvla.evaluation.metrics import action_error, expert_utilization, rollout_metrics
from surgicalvla.training.losses import flow_matching_loss, surgical_total_loss


def test_flow_matching_and_total_loss():
    prediction = torch.randn(2, 4, 7)
    target = torch.randn(2, 4, 7)
    loss = flow_matching_loss(prediction, target)
    assert loss.shape == prediction.shape
    assert torch.isfinite(surgical_total_loss(loss))


def test_q1_metrics():
    prediction = torch.zeros(2, 4, 7)
    target = torch.ones(2, 4, 7)
    assert action_error(prediction, target)["action_mse"] > 0
    rollout = rollout_metrics(prediction, target, torch.tensor([True, False]))
    assert "position_xyz_mae" in rollout and "task_success_rate" in rollout
    utilization = expert_utilization(torch.tensor([[0, 1], [1, 2]]))
    assert sum(utilization.values()) == 1.0
