"""Phase-conditioned MoE FFN, ReinFlow noise head, Gaussian log-prob helpers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from surgicalvla.modules.nn import FP32Linear


class SwiGLUExpert(nn.Module):
    """SwiGLU FFN (SmolLM2/Gemma layout)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act_fn: nn.Module | None = None,
        *,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = act_fn if act_fn is not None else nn.SiLU()

    @classmethod
    def from_dense(cls, dense: nn.Module, *, zero_output: bool = False) -> SwiGLUExpert:
        """Copy a dense MLP into an expert (weight up-cycling)."""
        gate = getattr(dense, "gate_proj", None)
        up = getattr(dense, "up_proj", None)
        down = getattr(dense, "down_proj", None)
        if gate is None or up is None or down is None:
            raise ValueError(
                "dense MLP must expose gate_proj/up_proj/down_proj (SmolLM2/Gemma layout); "
                f"got {type(dense).__name__}"
            )
        expert = cls(
            hidden_size=gate.in_features,
            intermediate_size=gate.out_features,
            act_fn=getattr(dense, "act_fn", None),
            bias=gate.bias is not None,
        ).to(dtype=gate.weight.dtype, device=gate.weight.device)
        with torch.no_grad():
            expert.gate_proj.weight.copy_(gate.weight)
            expert.up_proj.weight.copy_(up.weight)
            expert.down_proj.weight.copy_(down.weight)
            if gate.bias is not None:
                expert.gate_proj.bias.copy_(gate.bias)
                expert.up_proj.bias.copy_(up.bias)
                expert.down_proj.bias.copy_(down.bias)
            if zero_output:
                expert.down_proj.weight.zero_()
                if expert.down_proj.bias is not None:
                    expert.down_proj.bias.zero_()
        return expert

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PhaseConditionedMoEFFN(nn.Module):
    """Top-k MoE FFN routed on ``[h_action, e_phase, stopgrad(obj_context)]``."""

    def __init__(
        self,
        dense_mlp: nn.Module,
        hidden_size: int,
        e_phase_dim: int,
        obj_ctx_dim: int,
        num_experts: int = 4,
        num_shared_experts: int = 1,
        top_k: int = 2,
        num_phases: int = 4,
        *,
        zero_routed_output: bool = True,
    ):
        super().__init__()
        if num_experts < 1 or num_shared_experts < 1:
            raise ValueError("num_experts and num_shared_experts must be >= 1")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.top_k = int(top_k)
        self.e_phase_dim = int(e_phase_dim)
        self.obj_ctx_dim = int(obj_ctx_dim)

        self.shared_experts = nn.ModuleList(
            [SwiGLUExpert.from_dense(dense_mlp) for _ in range(num_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [SwiGLUExpert.from_dense(dense_mlp, zero_output=zero_routed_output) for _ in range(num_experts)]
        )
        self.router = FP32Linear(hidden_size + e_phase_dim + obj_ctx_dim, num_experts, bias=True)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        self.phase_classifier = nn.Linear(e_phase_dim, num_phases)
        self.aux_loss: Tensor | None = None
        self.last_usage: Tensor | None = None
        self._e_phase: Tensor | None = None
        self._obj_ctx: Tensor | None = None

    # ------------------------------------------------------------------ context
    def set_context(self, e_phase: Tensor | None, obj_context: Tensor | None) -> None:
        """Set the per-chunk router context (object context is stop-gradiented)."""
        self._e_phase = None if e_phase is None else e_phase
        self._obj_ctx = None if obj_context is None else obj_context.detach()

    def clear_context(self) -> None:
        self._e_phase = None
        self._obj_ctx = None
        self.aux_loss = None

    def _broadcast(self, ctx: Tensor | None, dim: int, x: Tensor) -> Tensor:
        if ctx is None:
            return x.new_zeros(x.shape[0], x.shape[1], dim)
        if ctx.ndim == 1:
            ctx = ctx[None, :]
        if ctx.shape[0] != x.shape[0]:
            raise ValueError(f"context batch size {ctx.shape[0]} != hidden batch size {x.shape[0]}")
        return ctx[:, None, :].expand(-1, x.shape[1], -1).to(dtype=x.dtype)

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor) -> Tensor:
        e_phase = self._broadcast(self._e_phase, self.e_phase_dim, x)
        obj_ctx = self._broadcast(self._obj_ctx, self.obj_ctx_dim, x)

        router_in = torch.cat([x.float(), e_phase.float(), obj_ctx.float()], dim=-1)
        logits = self.router(router_in).to(dtype=x.dtype)  # (B, T, E)

        probs = torch.softmax(logits.float(), dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)
        top_weights = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        shared = sum(expert(x) for expert in self.shared_experts) / self.num_shared_experts
        routed = self._dispatch(x, top_weights, top_idx)
        out = shared + routed

        # Switch-Transformer load balancing: E * sum_i f_i * p_i.
        onehot = torch.zeros_like(probs).scatter_(-1, top_idx, 1.0)
        frac = onehot.mean(dim=(0, 1))
        mean_prob = probs.mean(dim=(0, 1))
        self.aux_loss = self.num_experts * torch.sum(frac * mean_prob)
        self.last_usage = (frac.detach() / self.top_k)
        return out

    def _dispatch(self, x: Tensor, weights: Tensor, indices: Tensor) -> Tensor:
        """Evaluate each routed expert only on the tokens that selected it."""
        bsize, seq_len, hidden = x.shape
        flat_x = x.reshape(-1, hidden)
        flat_w = weights.reshape(-1, self.top_k)
        flat_idx = indices.reshape(-1, self.top_k)
        out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.routed_experts):
            selected = flat_idx == expert_id
            tokens = selected.any(dim=-1)
            if not bool(tokens.any()):
                continue
            weight = (flat_w * selected.to(flat_w.dtype)).sum(dim=-1)[tokens].to(flat_x.dtype)
            out[tokens] = out[tokens] + weight[:, None] * expert(flat_x[tokens])
        return out.view(bsize, seq_len, hidden)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, experts={self.num_experts}, "
            f"shared={self.num_shared_experts}, top_k={self.top_k}"
        )


class ReinFlowNoiseHead(nn.Module):
    """Per-dimension ReinFlow log-std, clamped to ``[log(min_std), log(max_std)]``."""

    def __init__(
        self,
        hidden_size: int,
        time_emb_dim: int | None = None,
        min_std: float = 0.01,
        max_std: float = 0.15,
        hidden_mult: int = 2,
        out_dim: int | None = None,
    ):
        super().__init__()
        if min_std <= 0 or max_std < min_std:
            raise ValueError(f"expected 0 < min_std <= max_std, got ({min_std}, {max_std})")
        self.hidden_size = int(hidden_size)
        self.time_emb_dim = int(time_emb_dim if time_emb_dim is not None else hidden_size)
        self.out_dim = int(out_dim if out_dim is not None else hidden_size)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.log_std_min = math.log(self.min_std)
        self.log_std_max = math.log(self.max_std)

        inner = max(hidden_size, self.time_emb_dim) * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(hidden_size + self.time_emb_dim, inner),
            nn.SiLU(),
            nn.Linear(inner, self.out_dim),
        )
        # Start at the geometric mean of the allowed range, with a small variance.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.5 * (self.log_std_min + self.log_std_max))

    def forward(self, hidden: Tensor, time_emb: Tensor) -> Tensor:
        if time_emb.ndim == 2:
            time_emb = time_emb[:, None, :]
        if time_emb.shape[1] == 1 and hidden.shape[1] != 1:
            time_emb = time_emb.expand(-1, hidden.shape[1], -1)
        if time_emb.shape[:2] != hidden.shape[:2]:
            raise ValueError(
                f"time_emb shape {tuple(time_emb.shape)} is not broadcastable to hidden {tuple(hidden.shape)}"
            )
        log_std = self.net(torch.cat([hidden, time_emb], dim=-1))
        return torch.clamp(log_std, self.log_std_min, self.log_std_max)

    def std(self, hidden: Tensor, time_emb: Tensor) -> Tensor:
        return torch.exp(self.forward(hidden, time_emb))


def masked_gaussian_log_prob(
    sample: Tensor,
    mean: Tensor,
    log_std: Tensor,
    action_dim: int | None = None,
    action_is_pad: Tensor | None = None,
) -> Tensor:
    """Diagonal-Gaussian log-probability, summed over valid entries -> ``(B,)``."""
    if action_dim is not None:
        sample = sample[..., :action_dim]
        mean = mean[..., :action_dim]
        log_std = log_std[..., :action_dim]
    inv_var = torch.exp(-2.0 * log_std)
    per_element = 0.5 * ((sample - mean) ** 2 * inv_var + 2.0 * log_std + math.log(2.0 * math.pi))
    if action_is_pad is not None:
        per_element = per_element.masked_fill(action_is_pad[..., None], 0.0)
    return -per_element.sum(dim=(1, 2))


def gaussian_entropy(
    log_std: Tensor,
    action_dim: int | None = None,
    action_is_pad: Tensor | None = None,
) -> Tensor:
    """Differential entropy of a diagonal Gaussian, summed over valid entries."""
    if action_dim is not None:
        log_std = log_std[..., :action_dim]
    per_element = log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))
    if action_is_pad is not None:
        per_element = per_element.masked_fill(action_is_pad[..., None], 0.0)
    return per_element.sum(dim=(1, 2))


__all__ = [
    "PhaseConditionedMoEFFN",
    "ReinFlowNoiseHead",
    "SwiGLUExpert",
    "gaussian_entropy",
    "masked_gaussian_log_prob",
]
