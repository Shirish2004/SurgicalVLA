"""Tests for the MoE action expert and the skill clusterer.

These stay free of LeRobot: ``PhaseMoEFFN`` only needs a plain gated-FFN-shaped
module, so the routing logic is testable in milliseconds instead of loading a
500M-parameter VLM.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from surgicalvla.modules.moe import (
    PhaseMoEFFN,
    RouteState,
    SlicedExpert,
    moe_aux_losses,
    select_experts,
    upcycle_to_moe,
)
from surgicalvla.modules.skills import NUM_FEATURES, SkillClusterer


class DenseFFN(nn.Module):
    """Minimal stand-in for a Gemma MLP (``down(act(gate(x)) * up(x))``)."""

    def __init__(self, hidden: int = 32, inter: int = 64):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_upcycle_is_identity_at_step_zero():
    dense = DenseFFN().eval()
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        reference = dense(x)
        moe = PhaseMoEFFN(dense, RouteState(4), layer_idx=0, num_experts=4, ratio=0.25).eval()
        out = moe(x)
    torch.testing.assert_close(out, reference)


def test_router_starts_uniform_and_uses_prior():
    route = RouteState(4)
    moe = PhaseMoEFFN(DenseFFN(), route, layer_idx=3, num_experts=4, ratio=0.25)
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        moe(x)
    # zero-initialised router -> uniform logits
    torch.testing.assert_close(route.logits[3], torch.zeros(2, 6, 4), rtol=0.0, atol=1e-6)

    route.reset()
    route.prior = torch.tensor([[0.0, 0.0, 50.0, 0.0], [50.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        moe(x)
    weights = route.weights[3]
    assert weights.shape == (2, 6, 4)
    assert weights[0, 0, 2] > 0.99 and weights[1, 0, 0] > 0.99


def test_routing_is_per_token():
    """Two tokens in one sample can take different experts."""
    route = RouteState(2)
    moe = PhaseMoEFFN(DenseFFN(), route, layer_idx=0, num_experts=2, ratio=0.25, prior_bias=1.0)
    with torch.no_grad():
        # make the hidden state a routing signal by decoupling the router
        moe.router.weight.copy_(torch.tensor([[10.0] * 32, [-10.0] * 32]))
        x = torch.zeros(1, 2, 32)
        x[0, 0] = 1.0
        x[0, 1] = -1.0
        moe(x)
    w = route.weights[0]
    assert w[0, 0, 0] > 0.9
    assert w[0, 1, 1] > 0.9


def test_off_mode_is_exactly_the_shared_ffn():
    route = RouteState(4)
    route.mode = "off"
    dense = DenseFFN().eval()
    moe = PhaseMoEFFN(dense, route, layer_idx=0, num_experts=4, ratio=0.25).eval()
    x = torch.randn(2, 3, 32)
    with torch.no_grad():
        torch.testing.assert_close(moe(x), dense(x))
    assert route.logits == {}


def test_oracle_override_is_one_hot():
    route = RouteState(3)
    route.mode = "oracle"
    route.override = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    moe = PhaseMoEFFN(DenseFFN(), route, layer_idx=1, num_experts=3, ratio=0.25)
    with torch.no_grad():
        moe(torch.randn(2, 4, 32))
    expected = torch.zeros(2, 4, 3)
    expected[0, :, 1] = 1.0
    expected[1, :, 0] = 1.0
    torch.testing.assert_close(route.weights[1], expected)


def test_select_experts_is_sparse_and_normalized():
    probs = torch.tensor([[0.85, 0.1, 0.05], [0.4, 0.35, 0.25]])
    w = select_experts(probs, max_k=2, top_p=0.7)
    assert torch.allclose(w.sum(-1), torch.ones(2), atol=1e-6)
    # confident row uses a single expert, uncertain row is allowed two
    assert (w[0] > 0).sum() == 1
    assert (w[1] > 0).sum() == 2


def test_upcycle_to_moe_only_touches_every_nth_layer():
    vlm = nn.Module()
    vlm.lm_expert = nn.Module()
    vlm.lm_expert.layers = nn.ModuleList([nn.Module() for _ in range(6)])
    for layer in vlm.lm_expert.layers:
        layer.mlp = DenseFFN()
    route = RouteState(2)
    upcycle_to_moe(vlm, route, num_experts=2, ratio=0.25, every_n=2)
    kinds = [type(layer.mlp).__name__ for layer in vlm.lm_expert.layers]
    assert kinds == ["DenseFFN", "PhaseMoEFFN"] * 3


def test_moe_aux_losses_without_routing_is_zero():
    loss, metrics = moe_aux_losses(RouteState(4))
    assert loss.item() == 0.0 and metrics == {}


def test_moe_aux_losses_rewards_balance():
    """A balanced dispatch must yield a lower load-balance term than a collapsed one."""

    def lb(weights, logits):
        route = RouteState(2)
        route.prior = None
        route.logits[0] = logits
        route.weights[0] = weights
        _, m = moe_aux_losses(route)
        return m["moe_load_balance"]

    balanced = torch.full((4, 1, 2), 0.5)
    collapsed = torch.zeros(4, 1, 2)
    collapsed[..., 0] = 1.0
    uniform_logits = torch.zeros(4, 1, 2)
    skewed_logits = torch.tensor([[[30.0, 0.0]]]).expand(4, 1, 2)
    assert lb(balanced, uniform_logits) < lb(collapsed, skewed_logits)


def test_moe_aux_losses_supervises_labels():
    route = RouteState(3)
    route.logits[0] = torch.zeros(5, 2, 3)
    route.logits[1] = torch.zeros(5, 2, 3)
    loss, metrics = moe_aux_losses(route, torch.zeros(5, dtype=torch.long))
    assert loss.item() > 0
    assert metrics["skill_ce"] > 0
    assert "route_entropy" in metrics and "moe_z" in metrics


def test_skill_clusterer_features_and_labels():
    sk = SkillClusterer(num_clusters=3, action_dim=7, gripper_dims=(6,))
    torch.manual_seed(0)
    still = torch.zeros(8, 10, 7)
    ramp = torch.linspace(0.0, 1.0, 10).view(1, 10, 1).expand(8, 10, 7)
    actions = torch.cat([still, ramp])
    feats = sk.features(actions)
    assert feats.shape == (16, NUM_FEATURES)
    assert torch.isfinite(feats).all()
    assert sk.labels(actions) is None  # not fitted yet

    sk.fit(feats, iters=10)
    labels = sk.labels(actions)
    assert labels.shape == (16,)
    assert set(labels.tolist()) <= {0, 1, 2}
    # identical chunks land in one cluster, and static is not the same cluster as moving
    assert labels[:8].unique().numel() == 1
    assert labels[8:].unique().numel() == 1
    assert labels[0] != labels[8]


def test_skill_clusterer_fit_with_zero_iters_is_an_assignment():
    sk = SkillClusterer(num_clusters=2, action_dim=4)
    torch.manual_seed(0)
    feats = torch.randn(16, NUM_FEATURES)
    sk.fit(feats, iters=0)
    labels = sk.labels(feats[:, None, :4].expand(-1, 2, -1))
    assert labels is not None and labels.shape == (16,)


def test_skill_clusterer_reset():
    sk = SkillClusterer(num_clusters=2, action_dim=4)
    sk.fit(torch.randn(8, NUM_FEATURES), iters=2)
    assert bool(sk.initialized)
    sk.reset()
    assert not bool(sk.initialized)
    assert int(sk.steps) == 0
    assert sk.labels(torch.zeros(2, 4, 4)) is None


@pytest.mark.parametrize("is_pad", [None, torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.bool)])
def test_skill_features_handle_padding(is_pad):
    sk = SkillClusterer(num_clusters=2, action_dim=3)
    f = sk.features(torch.randn(2, 3, 3), is_pad)
    assert f.shape == (2, NUM_FEATURES) and torch.isfinite(f).all()


def test_sliced_expert_is_a_zero_function_at_init():
    dense = DenseFFN()
    expert = SlicedExpert(dense, ratio=0.5)
    assert torch.count_nonzero(expert.down_proj.weight) == 0
    out = expert(torch.randn(4, 32))
    assert torch.count_nonzero(out) == 0
