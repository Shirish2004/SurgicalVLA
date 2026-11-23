"""Token-level, skill-routed Mixture-of-Experts for SmolVLA's action expert.

Layout
------
Every ``moe_every_n_layers``-th FFN inside the action expert (``lm_expert``) is
up-cycled into a :class:`PhaseMoEFFN`. The original dense FFN survives as an
always-on *shared* expert, and ``num_experts`` bandwidth-limited experts are
added next to it. Each expert is a random ``ratio`` slice of the dense FFN's
rows with a zero-initialised ``down_proj``, so at step 0 every expert outputs
exactly zero: up-cycling is an exact identity and the pretrained FFN behaviour
is preserved.

Routing granularity
-------------------
Routing is **per token**, not per sample. The action expert only ever sees the
``chunk_size`` action tokens, so a token is a chunk step: a 50-step chunk
(~1.7 s at 30 fps) can move through several skills instead of being forced into
one mixture for its whole duration.

Two levels cooperate:

* a chunk-level **phase prior** (:class:`~surgicalvla.modules.skills.PhaseHead`)
  read from the vision-language prefix, which biases every token's router and
  is what makes the routing conditioned on the scene and the instruction;
* a **per-layer, per-token router** (a float32 linear map on the expert hidden
  state) that lets individual steps deviate from that prior.

The pooled per-layer logits are what the skill cross-entropy supervises, so the
router stays anchored to the unsupervised skill clusters while still being free
to switch mid-chunk.
"""

from __future__ import annotations

import torch
from torch import nn

from surgicalvla.modules.nn import FP32Linear


class RouteState:
    """Mutable routing state shared by every MoE layer for one forward pass.

    Owned by the model and cleared after each training step / action sample, so
    it never carries stale routing across batches.
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.mode = "on"  # "on" | "off" | "oracle"
        self.prior: torch.Tensor | None = None  # (B, E) additive log-prior
        self.override: torch.Tensor | None = None  # (B, E) hard mixture (oracle / teacher forcing)
        self.labels: torch.Tensor | None = None  # (B,) skill-cluster ids
        self.logits: dict[int, torch.Tensor] = {}
        self.weights: dict[int, torch.Tensor] = {}
        self.usage: dict[int, torch.Tensor] = {}

    def record(self, layer: int, logits: torch.Tensor, weights: torch.Tensor) -> None:
        self.logits[layer] = logits
        self.weights[layer] = weights.detach()
        mass = weights.detach().sum(dim=tuple(range(weights.dim() - 1)))
        self.usage[layer] = self.usage.get(layer, 0) + mass

    def usage_fractions(self) -> dict[int, torch.Tensor]:
        """Per-layer expert utilization, normalized to sum to 1."""
        return {k: v / v.sum().clamp_min(1e-6) for k, v in self.usage.items()}

    def reset(self) -> None:
        self.prior = None
        self.override = None
        self.labels = None
        self.logits.clear()
        self.weights.clear()
        self.usage.clear()


class SlicedExpert(nn.Module):
    """A ``ratio``-wide slice of a dense gated FFN, with a zeroed output head.

    ``down_proj`` starts at zero so the expert is a no-op at up-cycling time;
    training then grows the slice into a specialised residual on top of the
    shared FFN.
    """

    def __init__(self, dense: nn.Module, ratio: float):
        super().__init__()
        ref = dense.gate_proj.weight
        inter, hidden = ref.shape
        width = max(16, int(inter * ratio))
        pick = torch.randperm(inter, device=ref.device)[:width]
        kw = dict(bias=False, dtype=ref.dtype, device=ref.device)
        self.gate_proj = nn.Linear(hidden, width, **kw)
        self.up_proj = nn.Linear(hidden, width, **kw)
        self.down_proj = nn.Linear(width, hidden, **kw)
        with torch.no_grad():
            self.gate_proj.weight.copy_(dense.gate_proj.weight[pick])
            self.up_proj.weight.copy_(dense.up_proj.weight[pick])
            self.down_proj.weight.zero_()
        self.act = dense.act_fn

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class PhaseMoEFFN(nn.Module):
    """Shared dense FFN + sparse, per-token routed experts + a per-layer router."""

    def __init__(
        self,
        dense: nn.Module,
        route: RouteState,
        layer_idx: int,
        num_experts: int,
        ratio: float,
        max_k: int = 2,
        top_p: float = 0.7,
        prior_bias: float = 1.0,
    ):
        super().__init__()
        self.shared = dense
        self.route = route
        self.layer_idx = layer_idx
        self.max_k = max_k
        self.top_p = top_p
        self.prior_bias = prior_bias
        self.experts = nn.ModuleList(SlicedExpert(dense, ratio) for _ in range(num_experts))
        hidden = dense.gate_proj.in_features
        # zero-init => uniform logits at step 0, so the phase prior decides routing
        self.router = FP32Linear(hidden, num_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, x):
        out = self.shared(x)
        route = self.route
        if route.mode == "off":
            return out

        logits = self.router(x)  # (B, T, E) in float32 by construction
        if route.prior is not None and self.prior_bias:
            logits = logits + self.prior_bias * route.prior[:, None, :]

        if route.override is not None:
            weights = route.override[:, None, :].expand_as(logits)
        else:
            weights = select_experts(logits.softmax(-1), self.max_k, self.top_p)
        route.record(self.layer_idx, logits, weights)

        b, t, _ = weights.shape
        flat_x = x.reshape(b * t, -1)
        flat_w = weights.reshape(b * t, -1).to(out.dtype)
        flat_out = out.reshape(b * t, -1)
        for e, expert in enumerate(self.experts):
            coef = flat_w[:, e]
            rows = (coef > 0).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            flat_out = flat_out.index_add(0, rows, expert(flat_x[rows]) * coef[rows].view(-1, 1))
        return flat_out.view(b, t, -1)


def select_experts(probs: torch.Tensor, max_k: int, top_p: float) -> torch.Tensor:
    """Nucleus selection over the top-``max_k`` experts, renormalized.

    Keeps the smallest prefix of the top-k whose cumulative probability is still
    below ``top_p``, so confident routers use one expert and uncertain ones use
    up to ``max_k``.
    """
    k = min(max_k, probs.shape[-1])
    vals, idx = probs.topk(k, dim=-1)
    keep = (vals.cumsum(-1) - vals) < top_p
    vals = vals * keep
    vals = vals / vals.sum(-1, keepdim=True).clamp_min(1e-6)
    return torch.zeros_like(probs).scatter(-1, idx, vals)


def upcycle_to_moe(
    vlm_with_expert,
    route: RouteState,
    num_experts: int,
    ratio: float,
    every_n: int,
    max_k: int = 2,
    top_p: float = 0.7,
    prior_bias: float = 1.0,
):
    """Replace every ``every_n``-th action-expert FFN with a :class:`PhaseMoEFFN`."""
    for i, layer in enumerate(vlm_with_expert.lm_expert.layers):
        if i % every_n == every_n - 1:
            layer.mlp = PhaseMoEFFN(
                layer.mlp, route, i, num_experts, ratio, max_k, top_p, prior_bias
            )
    return vlm_with_expert


def moe_layers(vlm_with_expert) -> list[PhaseMoEFFN]:
    return [layer.mlp for layer in vlm_with_expert.lm_expert.layers if isinstance(layer.mlp, PhaseMoEFFN)]


def moe_aux_losses(
    route: RouteState,
    labels: torch.Tensor | None = None,
    *,
    skill_coef: float = 0.1,
    z_coef: float = 1e-3,
    load_balance_coef: float = 0.01,
    label_smoothing: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Auxiliary router losses: skill CE + router z-loss + load balancing.

    CE is computed on the per-layer logits pooled over chunk steps, i.e. the
    mixture a *chunk* uses on average, which is the granularity the skill
    clusters are defined at.

    The load-balance term is the standard Switch-Transformer objective
    ``E * sum_e f_e * P_e`` with ``f`` the detached dispatch fraction and ``P``
    the mean softmax probability, so starving experts push probability mass back
    onto themselves.

    Returns ``(loss_to_add, metrics)``; metrics are plain floats for logging.
    """
    metrics: dict[str, float] = {}
    layers = sorted(route.logits)
    if not layers:  # routing disabled, or no MoE layer ran
        return torch.zeros(()), metrics

    ref = route.logits[layers[0]]
    extra = torch.zeros((), device=ref.device, dtype=torch.float32)
    z = torch.zeros((), device=ref.device, dtype=torch.float32)
    lb = torch.zeros((), device=ref.device, dtype=torch.float32)
    entropy = 0.0

    for layer in layers:
        logits = route.logits[layer]
        probs = logits.softmax(-1)
        z = z + torch.logsumexp(logits, dim=-1).pow(2).mean()
        if layer in route.weights:
            w = route.weights[layer]
            frac = w.sum(dim=tuple(range(w.dim() - 1)))
            frac = frac / frac.sum().clamp_min(1e-6)
            mean_p = probs.mean(dim=tuple(range(probs.dim() - 1)))
            lb = lb + (frac * mean_p).sum() * probs.shape[-1]
        entropy += -(probs * probs.clamp_min(1e-6).log()).sum(-1).mean().item()

    n = len(layers)
    extra = extra + z_coef * (z / n) + load_balance_coef * (lb / n)
    metrics["moe_z"] = (z / n).item()
    metrics["moe_load_balance"] = (lb / n).item()
    metrics["route_entropy"] = entropy / n

    if labels is not None and len(layers):
        pooled = torch.stack([route.logits[k].mean(dim=1) for k in layers]).mean(0)
        ce = nn.functional.cross_entropy(pooled, labels, label_smoothing=label_smoothing)
        extra = extra + skill_coef * ce
        metrics["skill_ce"] = ce.item()
        metrics["skill_acc"] = (pooled.argmax(-1) == labels).float().mean().item()

    return extra, metrics
