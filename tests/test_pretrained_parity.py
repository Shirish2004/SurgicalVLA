"""Regression tests against the real ``lerobot/smolvla_base`` checkpoint (audit A1, A2).

The stub backbone in ``test_smolvla_moe.py`` cannot exercise SmolVLA's VLM/expert
layer pairing (an expert layer per VLM layer). These tests load the real weights
and check that the up-cycled policy is *function-preserving* at initialisation:
every routed expert starts with a zero down-projection, so with no object/CoT
inputs it must reproduce ``SmolVLAPolicy`` actions for the same noise.

Skipped when CUDA or the cached checkpoint is unavailable.
"""

from __future__ import annotations

import os

import pytest
import torch

from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig

pytestmark = pytest.mark.gpu


def _has_checkpoint() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("lerobot/smolvla_base", "model.safetensors"), str)
    except Exception:
        return False


requires_base = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_checkpoint(),
    reason="needs CUDA and a cached lerobot/smolvla_base",
)


def test_partial_expert_depth_is_rejected():
    with pytest.raises(ValueError, match="skip action-expert layers"):
        SmolVLAMoEConfig(num_vlm_layers=16, expert_cross_attn_vlm_layers=8)
    assert SmolVLAMoEConfig(num_vlm_layers=16).expert_cross_attn_vlm_layers == 16
    assert SmolVLAMoEConfig(num_vlm_layers=16, expert_cross_attn_vlm_layers=-1).expert_cross_attn_vlm_layers == 16


@requires_base
def test_upcycled_policy_matches_smolvla_base_at_init():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from surgicalvla.modules.modeling_smolvla_moe import SmolVLAMoEPolicy

    torch.manual_seed(0)
    dev = "cuda"
    base = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(dev).eval()
    bc = base.config
    fields = SmolVLAMoEConfig.__dataclass_fields__
    kw = {k: getattr(bc, k) for k in bc.__dataclass_fields__ if k in fields and k != "type"}
    kw.update(use_cot=False, obj_tracker_mode="passthrough")
    pol = SmolVLAMoEPolicy.from_smolvla_pretrained("lerobot/smolvla_base", SmolVLAMoEConfig(**kw)).to(dev).eval()

    bsz = 2
    imgs = [torch.rand(bsz, 3, 512, 512, device=dev) * 2 - 1 for _ in bc.image_features]
    masks = [torch.ones(bsz, dtype=torch.bool, device=dev) for _ in imgs]
    tok = base.model.vlm_with_expert.processor.tokenizer(
        ["pick up the scalpel\n"] * bsz, return_tensors="pt", padding="max_length", max_length=48
    )
    lt, lm = tok.input_ids.to(dev), tok.attention_mask.bool().to(dev)
    state = torch.randn(bsz, bc.max_state_dim, device=dev)
    noise = torch.randn(bsz, bc.chunk_size, bc.max_action_dim, device=dev)

    ran = set()
    hooks = [
        layer.mlp.register_forward_hook(lambda *_, j=j: ran.add(j))
        for j, layer in enumerate(pol.model.vlm_with_expert.lm_expert.layers)
    ]
    with torch.no_grad():
        ref = base.model.sample_actions(imgs, masks, lt, lm, state, noise=noise.clone())
        out = pol.model.sample_actions(imgs, masks, lt, lm, state, noise=noise.clone())
    for h in hooks:
        h.remove()

    assert ran == set(range(len(pol.model.vlm_with_expert.lm_expert.layers))), "expert layers skipped"
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 2e-3, f"up-cycled policy diverges from smolvla_base at init: rel={rel:.3e}"
