import torch

from surgicalvla.modules.moe import (
    ChebyshevKANLinear,
    FNOExpert2D,
    LiquidTimeConstantExpert,
    ScientificMoELayer,
    TaskConditionedTokenGate,
)
from surgicalvla.training.losses import load_balancing_loss, router_z_loss, surgical_total_loss


def test_task_gate_retention_and_gradient():
    visual = torch.randn(2, 16, 8, requires_grad=True)
    text = torch.randn(2, 4, 8)
    retained, scores = TaskConditionedTokenGate(8, 0.35)(visual, text)
    assert retained.shape == (2, 5, 8)
    assert scores.shape == (2, 16)
    retained.sum().backward()
    assert visual.grad is not None


def test_scientific_experts_preserve_shape():
    tokens = torch.randn(2, 9, 8, requires_grad=True)
    routed, selected, logits, dense = ScientificMoELayer(8, top_k=2)(tokens)
    assert routed.shape == tokens.shape
    assert selected.shape == (2, 9, 2)
    assert logits.shape == dense.shape == (2, 9, 3)
    routed.mean().backward()
    assert tokens.grad is not None
    assert FNOExpert2D(8)(tokens.detach()).shape == tokens.shape
    assert LiquidTimeConstantExpert(8)(tokens.detach()).shape == tokens.shape
    assert ChebyshevKANLinear(8, 8)(tokens.detach()).shape == tokens.shape


def test_moe_losses_and_total_loss():
    probabilities = torch.full((4, 3), 1 / 3)
    logits = torch.zeros(4, 3)
    assert torch.isfinite(load_balancing_loss(probabilities))
    assert torch.isfinite(router_z_loss(logits))
    total = surgical_total_loss(torch.ones(2, 4, 6), load_balancing_loss(probabilities), router_z_loss(logits))
    assert total.ndim == 0 and torch.isfinite(total)
