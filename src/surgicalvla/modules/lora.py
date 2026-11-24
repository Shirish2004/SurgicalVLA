"""Low-rank adapters (LoRA) on frozen VLM linear layers.

``add_lora`` turns an ``nn.Linear`` into a :class:`LoRALinear` in place: the frozen weight keeps
its name (``q_proj.weight``), so pretrained SmolVLA checkpoints still load, and two small
trainable matrices are added (``lora_A`` (r, in), ``lora_B`` (out, r), ``B = 0`` at
initialisation). The output is ``W x + b + (alpha / r) · B A x``, identical to the original
layer until training moves ``B``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

TEXT_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Linear):
    lora_A: nn.Parameter
    lora_B: nn.Parameter
    lora_scale: float

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return out + (self.lora_scale * delta).to(out.dtype)


def add_lora(linear: nn.Linear, rank: int, alpha: float) -> LoRALinear:
    if isinstance(linear, LoRALinear):
        return linear
    linear.__class__ = LoRALinear
    dev = linear.weight.device
    linear.lora_A = nn.Parameter(torch.empty(rank, linear.in_features, device=dev, dtype=torch.float32))
    linear.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank, device=dev, dtype=torch.float32))
    nn.init.kaiming_uniform_(linear.lora_A, a=math.sqrt(5))
    linear.lora_scale = float(alpha) / float(rank)
    linear.weight.requires_grad_(False)
    if linear.bias is not None:
        linear.bias.requires_grad_(False)
    return linear


def apply_vlm_lora(vlm_with_expert, rank: int, alpha: float, parts=("text",), modules=TEXT_MODULES) -> int:
    """Add LoRA to the VLM's decoder layers (``"text"``) and/or vision encoder (``"vision"``).

    Only the layers the policy runs are adapted (SmolVLA keeps the first ``num_vlm_layers``).
    Returns the number of LoRA parameters added.
    """
    added = 0
    model = vlm_with_expert.get_vlm_model()
    targets = []
    if "text" in parts:
        for layer in model.text_model.layers:
            targets += [getattr(layer.self_attn, m, None) for m in modules] + [getattr(layer.mlp, m, None) for m in modules]
    if "vision" in parts:
        for layer in model.vision_model.encoder.layers:
            targets += [getattr(layer.self_attn, m, None) for m in ("q_proj", "k_proj", "v_proj", "out_proj")]
            targets += [getattr(layer.mlp, m, None) for m in ("fc1", "fc2")]
    for lin in targets:
        if isinstance(lin, nn.Linear) and not isinstance(lin, LoRALinear):
            add_lora(lin, rank, alpha)
            added += lin.lora_A.numel() + lin.lora_B.numel()
    return added


class lora_disabled:
    """Context manager: every LoRA layer under ``module`` acts as its frozen original."""

    def __init__(self, module: nn.Module):
        self.layers = [m for m in module.modules() if isinstance(m, LoRALinear)]
        self.scales: list[float] = []

    def __enter__(self):
        self.scales = [m.lora_scale for m in self.layers]
        for m in self.layers:
            m.lora_scale = 0.0
        return self

    def __exit__(self, *exc):
        for m, s in zip(self.layers, self.scales, strict=True):
            m.lora_scale = s
        return False


__all__ = ["TEXT_MODULES", "LoRALinear", "add_lora", "apply_vlm_lora", "lora_disabled"]
