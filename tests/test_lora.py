"""LoRA on frozen VLM layers: identity at initialisation, gradients only in the adapters, names kept."""

from __future__ import annotations

import torch
from torch import nn

from surgicalvla.modules.lora import LoRALinear, add_lora


def test_lora_starts_as_the_original_layer_and_trains_only_adapters():
    torch.manual_seed(0)
    lin = nn.Linear(8, 6)
    x = torch.randn(4, 8)
    before = lin(x).detach()
    names_before = set(dict(lin.named_parameters()))
    add_lora(lin, rank=2, alpha=4)
    assert isinstance(lin, LoRALinear) and {"weight", "bias"} <= set(dict(lin.named_parameters())) and names_before < set(dict(lin.named_parameters()))
    torch.testing.assert_close(lin(x), before)
    lin(x).sum().backward()
    assert lin.weight.grad is None and lin.bias.grad is None
    assert lin.lora_B.grad is not None and lin.lora_B.grad.abs().sum() > 0


def test_lora_state_dict_loads_pretrained_weights_by_their_original_names():
    src = nn.Linear(5, 3)
    dst = nn.Linear(5, 3)
    add_lora(dst, rank=2, alpha=2)
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert set(missing) == {"lora_A", "lora_B"} and not unexpected
    torch.testing.assert_close(dst.weight, src.weight)
