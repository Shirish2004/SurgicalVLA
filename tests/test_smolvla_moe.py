"""Unit tests for the SurgicalVLA MoE policy.

Everything runs against a small in-process VLM stub, so the suite exercises the
real tensor plumbing (prefix masks, KV cache, MoE routing, ReinFlow log-probs)
without loading the 500M-parameter backbone.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from lerobot.policies.common.vla_utils import make_att_2d_masks
from torch import nn

from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig
from surgicalvla.modules.modeling_smolvla_moe import SmolVLAMoEModel, SmolVLAMoEPolicy
from surgicalvla.modules.moe_layers import (
    PhaseConditionedMoEFFN,
    ReinFlowNoiseHead,
    SwiGLUExpert,
    masked_gaussian_log_prob,
)
from surgicalvla.modules.object_encoder import (
    ObjectCentric6DProjector,
    geodesic_rotation_angle,
    rotation_6d_to_matrix,
)

H, E, VOCAB, C_SAM = 32, 24, 96, 8


# ---------------------------------------------------------------------------
# Stub backbone
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class CacheLayer:
    def __init__(self, keys, values, cross_keys=None, cross_values=None):
        self.keys, self.values = keys, values
        self.cross_keys, self.cross_values = cross_keys, cross_values

    def get_seq_length(self):
        return self.keys.shape[1]

    def crop(self, tokens_to_remove: int):
        if tokens_to_remove >= 0 or not hasattr(self, "keys"):
            return
        keep = self.keys.shape[1] + tokens_to_remove
        self.keys = self.keys[:, :keep]
        self.values = self.values[:, :keep]
        if self.cross_keys is not None:
            self.cross_keys = self.cross_keys[:, :keep]
            self.cross_values = self.cross_values[:, :keep]


class StubCache:
    def __init__(self):
        self.layers = []

    def _put(self, idx, keys, values, cross_keys=None, cross_values=None):
        while len(self.layers) <= idx:
            self.layers.append(None)
        self.layers[idx] = CacheLayer(keys, values, cross_keys, cross_values)


class ExpertLayer(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.mlp = MLP(hidden, inter)
        self.norm = nn.LayerNorm(hidden)


class ExpertStack(nn.Module):
    def __init__(self, hidden: int, inter: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([ExpertLayer(hidden, inter) for _ in range(layers)])


class StubTextModel(nn.Module):
    def __init__(self, hidden: int, layers: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.ModuleList([nn.Linear(hidden, 3 * hidden) for _ in range(layers)])
        self.out = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.mlp = nn.ModuleList([MLP(hidden, 2 * hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def _attn(self, q, k, v, mask):
        h = self.heads
        b, lq, d = q.shape
        lk = k.shape[1]
        q = q.view(b, lq, h, d // h).transpose(1, 2)
        k = k.view(b, lk, h, d // h).transpose(1, 2)
        v = v.view(b, lk, h, d // h).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // h)
        scores = scores.masked_fill(~mask[:, None], -1e4)
        return (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, lq, d)

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True):
        x = inputs_embeds
        b, length = x.shape[:2]
        cache = past_key_values if isinstance(past_key_values, StubCache) else (StubCache() if use_cache else None)
        for i, (qkv, out, mlp, norm) in enumerate(zip(self.qkv, self.out, self.mlp, self.norms, strict=True)):
            q, k, v = qkv(x).chunk(3, -1)
            if cache is not None and i < len(cache.layers) and cache.layers[i] is not None:
                k = torch.cat([cache.layers[i].keys, k], dim=1)
                v = torch.cat([cache.layers[i].values, v], dim=1)
            if cache is not None:
                cache._put(i, k.detach(), v.detach())
            total = k.shape[1]
            qpos = torch.arange(total - length, total, device=x.device)
            kpos = torch.arange(total, device=x.device)
            mask = (kpos[None, :] <= qpos[:, None])[None].expand(b, -1, -1)
            if attention_mask is not None:
                mask = mask & attention_mask[:, None, :].bool()
            x = x + out(self._attn(q, k, v, mask))
            x = x + mlp(norm(x))
        return SimpleNamespace(last_hidden_state=self.norm(x), past_key_values=cache if use_cache else None)


class StubVLMText(nn.Module):
    def __init__(self, hidden: int, vocab: int, layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.text_model = StubTextModel(hidden, layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


class StubVLM(nn.Module):
    def __init__(self, hidden=H, expert=E, vocab=VOCAB, layers=4, expert_layers=4, heads=4):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden, vocab_size=vocab)
        )
        self.expert_hidden_size = expert
        self.num_vlm_layers = layers
        self.num_expert_layers = expert_layers
        self.heads = heads
        self.token_embed = nn.Embedding(vocab, hidden)
        self.image_proj = nn.Linear(3, hidden)
        self.qkv = nn.ModuleList([nn.Linear(hidden, 3 * hidden) for _ in range(layers)])
        self.out = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.vlm_mlp = nn.ModuleList([MLP(hidden, 2 * hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.vlm_norm = nn.LayerNorm(hidden)
        self.cross_k = nn.ModuleList([nn.Linear(hidden, expert) for _ in range(layers)])
        self.cross_v = nn.ModuleList([nn.Linear(hidden, expert) for _ in range(layers)])
        self.expert_qkv = nn.ModuleList([nn.Linear(expert, 3 * expert) for _ in range(expert_layers)])
        self.expert_out = nn.ModuleList([nn.Linear(expert, expert) for _ in range(expert_layers)])
        self.lm_expert = ExpertStack(expert, 2 * expert, expert_layers)
        self.expert_norm = nn.LayerNorm(expert)
        self.vlm = StubVLMText(hidden, vocab, layers)
        self.processor = None

    def embed_image(self, image):
        return self.image_proj(image.mean(dim=(2, 3)))[:, None, :]

    def embed_language_tokens(self, tokens):
        return self.token_embed(tokens)

    def _attn(self, q, k, v, mask):
        h, d = self.heads, q.shape[-1]
        b, lq = q.shape[:2]
        lk = k.shape[1]
        if mask.shape[1] != lq:
            mask = mask[:, -lq:, :]
        q = q.view(b, lq, h, d // h).transpose(1, 2)
        k = k.view(b, lk, h, d // h).transpose(1, 2)
        v = v.view(b, lk, h, d // h).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // h)
        scores = scores.masked_fill(~mask[:, None], -1e4)
        return (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, lq, d)

    def _vlm_prefill(self, prefix, mask, cache):
        x = prefix
        cross = []
        for i in range(self.num_vlm_layers):
            q, k, v = self.qkv[i](x).chunk(3, -1)
            x = x + self.out[i](self._attn(q, k, v, mask))
            x = x + self.vlm_mlp[i](self.norms[i](x))
            ck, cv = self.cross_k[i](x), self.cross_v[i](x)
            if cache is not None:
                cache._put(i, k.detach(), v.detach(), ck, cv)
            cross.append((ck, cv))
        return self.vlm_norm(x), cross

    def _expert_stack(self, suffix, cross, mask):
        x = suffix
        for i in range(min(self.num_vlm_layers, len(self.lm_expert.layers))):
            q, k, v = self.expert_qkv[i](x).chunk(3, -1)
            ck, cv = cross[i]
            k = torch.cat([ck, k], dim=1)
            v = torch.cat([cv, v], dim=1)
            x = x + self.expert_out[i](self._attn(q, k, v, mask))
            layer = self.lm_expert.layers[i]
            x = x + layer.mlp(layer.norm(x))
        return self.expert_norm(x)

    def forward(self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache=None):
        prefix, suffix = inputs_embeds
        cache = past_key_values if isinstance(past_key_values, StubCache) else None

        if suffix is None:
            if cache is None:
                cache = StubCache()
            out, _ = self._vlm_prefill(prefix, attention_mask, cache)
            return [out, None], cache

        if prefix is not None:
            n_prefix = prefix.shape[1]
            cache = cache if cache is not None else StubCache()
            _, cross = self._vlm_prefill(prefix, attention_mask[:, :n_prefix, :n_prefix], cache)
        else:
            cross = [(layer.cross_keys, layer.cross_values) for layer in cache.layers]
        return [None, self._expert_stack(suffix, cross, attention_mask)], cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_config(**overrides) -> SmolVLAMoEConfig:
    kwargs = dict(
        chunk_size=6,
        n_action_steps=6,
        max_state_dim=4,
        max_action_dim=8,
        num_steps=3,
        num_vlm_layers=4,
        num_experts=4,
        num_shared_experts=1,
        moe_top_k=2,
        num_object_tokens=3,
        obj_sam_feat_dim=C_SAM,
        obj_fourier_bands=4,
        num_objects=2,
        use_cot=True,
        cot_max_tokens=4,
        cot_refresh_interval=2,
        cot_loss_weight=0.1,
        prefix_length=-1,
        use_cache=True,
    )
    kwargs.update(overrides)
    return SmolVLAMoEConfig(**kwargs)


def make_model(config=None) -> SmolVLAMoEModel:
    torch.manual_seed(0)
    config = config or make_config()
    model = SmolVLAMoEModel(config, vlm_with_expert=StubVLM(layers=config.num_vlm_layers))
    model.eval()
    return model


def make_batch(config, bsize=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    total = 1 + 5 + config.num_object_tokens + 1 + 4
    return dict(
        images=[torch.randn(bsize, 3, 8, 8, generator=g)],
        img_masks=[torch.ones(bsize, dtype=torch.bool)],
        lang_tokens=torch.randint(0, VOCAB, (bsize, 5), generator=g),
        lang_masks=torch.ones(bsize, 5, dtype=torch.bool),
        state=torch.randn(bsize, config.max_state_dim, generator=g),
        actions=torch.randn(bsize, config.chunk_size, config.max_action_dim, generator=g),
        obj_sam_feat=torch.randn(bsize, 2, C_SAM, generator=g),
        obj_pose_vec=torch.randn(bsize, 2, 13, generator=g),
        # Row 0 has one inactive slot; every other row has both slots active.
        obj_valid_mask=torch.tensor([[True, False]] + [[True, True]] * (bsize - 1)),
        cot_ids=torch.randint(0, VOCAB, (bsize, 4), generator=g),
        cot_mask=torch.ones(bsize, 4, dtype=torch.bool),
        cot_phase_positions=torch.tensor([2, 1]),
        subtask_phase_id=torch.tensor([1, 3]),
        target_relative_pose_6d=torch.randn(bsize, 9, generator=g),
        action_is_pad=F.pad(torch.zeros(bsize, config.chunk_size - 1, dtype=torch.bool), (0, 1), value=True),
        _total=total,
    )


def to_model_inputs(model, batch, **extra):
    objects = model.object_features(
        batch["state"], batch["obj_sam_feat"], batch["obj_pose_vec"], batch["obj_valid_mask"]
    )
    kwargs = dict(
        images=batch["images"],
        img_masks=batch["img_masks"],
        lang_tokens=batch["lang_tokens"],
        lang_masks=batch["lang_masks"],
        state=batch["state"],
        actions=batch["actions"],
        objects=objects,
        cot_ids=batch["cot_ids"],
        cot_mask=batch["cot_mask"],
        cot_phase_positions=batch["cot_phase_positions"],
        subtask_phase_id=batch["subtask_phase_id"],
        target_relative_pose_6d=batch["target_relative_pose_6d"],
        action_is_pad=batch["action_is_pad"],
        action_dim=5,
        noise=torch.full_like(batch["actions"], 0.5),
        time=torch.full((batch["actions"].shape[0],), 0.3),
    )
    kwargs.update(extra)
    return kwargs


# ---------------------------------------------------------------------------
# Config / registration
# ---------------------------------------------------------------------------


def test_config_defaults_and_registration():
    from lerobot.configs import PreTrainedConfig

    cfg = SmolVLAMoEConfig()
    assert cfg.num_object_tokens == 4 and cfg.obj_pose_dim == 13
    assert cfg.num_experts == 4 and cfg.moe_top_k == 2
    assert cfg.reinflow_min_std == 0.01 and cfg.reinflow_max_std == 0.15
    assert cfg.expert_cross_attn_vlm_layers == cfg.num_vlm_layers
    assert "smolvla_moe" in PreTrainedConfig.get_known_choices()
    assert SmolVLAMoEPolicy.config_class is SmolVLAMoEConfig
    assert SmolVLAMoEPolicy.name == "smolvla_moe"


def test_config_rejects_bad_topk():
    with pytest.raises(ValueError):
        make_config(moe_top_k=9)


# ---------------------------------------------------------------------------
# Object projector
# ---------------------------------------------------------------------------


def test_object_projector_shapes_and_masking():
    proj = ObjectCentric6DProjector(H, E, sam_feat_dim=C_SAM, num_tokens=3, fourier_bands=4)
    proj.eval()
    sam = torch.randn(2, 2, C_SAM)
    pose = torch.randn(2, 2, 13)
    valid = torch.tensor([[True, False], [False, False]])
    out = proj(sam, pose, valid)
    assert out.tokens.shape == (2, 3, H)
    assert out.context.shape == (2, E)
    assert out.token_mask.tolist() == [[True] * 3, [False] * 3]
    assert torch.count_nonzero(out.tokens[1]) == 0
    assert torch.count_nonzero(out.context[1]) == 0
    assert torch.isfinite(out.tokens).all()


def test_object_projector_empty_is_zero_and_train_jitter():
    proj = ObjectCentric6DProjector(H, E, sam_feat_dim=C_SAM, num_tokens=3, fourier_bands=4)
    empty = proj(torch.zeros(2, 0, C_SAM), torch.zeros(2, 0, 13), torch.zeros(2, 0, dtype=torch.bool))
    assert empty.tokens.shape == (2, 3, H) and torch.count_nonzero(empty.tokens) == 0

    sam, pose = torch.randn(2, 2, C_SAM), torch.randn(2, 2, 13)
    valid = torch.ones(2, 2, dtype=torch.bool)
    proj.eval()
    with torch.no_grad():
        a = proj(sam, pose, valid)
    proj.train()
    torch.manual_seed(0)
    with torch.no_grad():
        b = proj(sam, pose, valid)
    assert not torch.allclose(a.tokens, b.tokens)


def test_object_projector_gradients():
    proj = ObjectCentric6DProjector(H, E, sam_feat_dim=C_SAM, num_tokens=3, fourier_bands=4)
    out = proj(torch.randn(2, 2, C_SAM), torch.randn(2, 2, 13), torch.ones(2, 2, dtype=torch.bool))
    out.tokens.sum().backward()
    assert proj.token_head.weight.grad is not None
    assert proj.token_head.weight.grad.abs().sum() > 0


def test_rotation_helpers_roundtrip():
    r = rotation_6d_to_matrix(torch.tensor([[1.0, 0, 0, 0, 1, 0]]))
    torch.testing.assert_close(r[0], torch.eye(3))
    angle = geodesic_rotation_angle(r, r)
    assert torch.allclose(angle, torch.zeros(1), atol=1e-3)


# ---------------------------------------------------------------------------
# MoE FFN / noise head
# ---------------------------------------------------------------------------


def test_moe_is_identity_at_init():
    dense = MLP(H, 2 * H).eval()
    x = torch.randn(2, 5, H)
    moe = PhaseConditionedMoEFFN(dense, H, H, H, num_experts=4, num_shared_experts=1, top_k=2).eval()
    with torch.no_grad():
        torch.testing.assert_close(moe(x), dense(x))


def test_moe_expert_matches_dense_dtype():
    dense = MLP(H, 2 * H).to(torch.bfloat16)
    x = torch.randn(2, 5, H, dtype=torch.bfloat16)
    expert = SwiGLUExpert.from_dense(dense)
    assert expert.gate_proj.weight.dtype == torch.bfloat16
    assert expert(x).dtype == torch.bfloat16


def test_moe_topk_and_aux_loss():
    dense = MLP(H, 2 * H)
    moe = PhaseConditionedMoEFFN(dense, H, H, H, num_experts=4, num_shared_experts=1, top_k=2)
    with torch.no_grad():
        moe.router.weight.normal_(0, 0.5)
    x = torch.randn(2, 6, H)
    moe.set_context(torch.randn(2, H), torch.randn(2, H))
    out = moe(x)
    assert out.shape == x.shape
    assert moe.aux_loss is not None and moe.aux_loss.ndim == 0
    assert moe.last_usage is not None and moe.last_usage.shape == (4,)
    assert torch.isfinite(moe.aux_loss)


def test_moe_usage_respects_topk():
    dense = MLP(H, 2 * H)
    moe = PhaseConditionedMoEFFN(dense, H, H, H, num_experts=4, top_k=2)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, :] = 10.0
        moe.router.weight[1, :] = 9.0
    moe(torch.rand(2, 6, H) + 0.5)
    assert int((moe.last_usage > 0).sum()) == 2


def test_reinflow_head_clamps_std():
    head = ReinFlowNoiseHead(8, 8, min_std=0.01, max_std=0.15)
    hidden = torch.randn(2, 5, 8) * 100
    emb = torch.randn(2, 8) * 100
    log_std = head(hidden, emb)
    assert torch.all(log_std >= math.log(0.01) - 1e-6)
    assert torch.all(log_std <= math.log(0.15) + 1e-6)


def test_masked_gaussian_log_prob_matches_formula():
    sample = torch.randn(2, 4, 3)
    mean = torch.randn(2, 4, 3)
    log_std = torch.randn(2, 4, 3) * 0.1 - 2
    got = masked_gaussian_log_prob(sample, mean, log_std)
    var = torch.exp(2 * log_std)
    expected = -0.5 * (((sample - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi)).sum((1, 2))
    torch.testing.assert_close(got, expected)


# ---------------------------------------------------------------------------
# Model: prefix, masks, losses, gradients
# ---------------------------------------------------------------------------


def test_prefix_order_and_attention_masks():
    cfg = make_config()
    model = make_model(cfg)
    batch = make_batch(cfg)
    objects = model.object_features(
        batch["state"], batch["obj_sam_feat"], batch["obj_pose_vec"], batch["obj_valid_mask"]
    )
    embs, pad, att, cot_slice = model.embed_prefix(
        batch["images"],
        batch["img_masks"],
        batch["lang_tokens"],
        batch["lang_masks"],
        batch["state"],
        object_tokens=objects.tokens,
        object_token_mask=objects.token_mask,
        cot_ids=batch["cot_ids"],
        cot_mask=batch["cot_mask"],
    )
    img, text, obj, state, cot = 1, 5, cfg.num_object_tokens, 1, 4
    assert embs.shape[1] == img + text + obj + state + cot
    assert att[0].tolist() == [0] * (img + text + obj) + [1] * (state + cot)
    assert cot_slice == (img + text + obj + state, img + text + obj + state + cot)

    mask = make_att_2d_masks(pad, att)[0]
    perception = slice(0, img + text + obj)
    state_i = img + text + obj
    cot_slice_i = slice(state_i + 1, state_i + 1 + cot)
    assert mask[perception, perception].all()
    assert not mask[perception, state_i].any()
    assert mask[state_i, perception].all()
    assert not mask[state_i, cot_slice_i].any()
    assert mask[cot_slice_i, state_i].all()
    assert not mask[state_i + 1, state_i + 2]
    assert mask[state_i + 2, state_i + 1]


def test_model_losses_and_gradients():
    cfg = make_config()
    model = make_model(cfg)
    model.train()
    batch = make_batch(cfg)
    out = model(**to_model_inputs(model, batch))
    for key in ("fm_loss", "cot_loss", "moe_aux_loss", "phase_loss", "pose_aux_loss", "loss"):
        assert key in out and torch.isfinite(out[key]), key
    out["loss"].backward()
    grads = {
        "object": model.obj_projector.token_head.weight.grad,
        "action": model.action_out_proj.weight.grad,
        "pose": model.pose_head.net[0].weight.grad,
        "router": model._moe_layers[0].router.weight.grad,
        "cot": model.cot_lm_head.weight.grad,
    }
    for name, grad in grads.items():
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_model_skips_optional_losses():
    cfg = make_config()
    model = make_model(cfg)
    batch = make_batch(cfg)
    inputs = to_model_inputs(model, batch)
    inputs["cot_ids"] = None
    inputs["cot_mask"] = None
    inputs["cot_phase_positions"] = None
    inputs["subtask_phase_id"] = None
    inputs["target_relative_pose_6d"] = None
    out = model(**inputs)
    assert "cot_loss" not in out and "phase_loss" not in out and "pose_aux_loss" not in out
    assert torch.isfinite(out["loss"])


def test_kv_cache_parity():
    cfg = make_config(use_cache=True)
    model = make_model(cfg)
    batch = make_batch(cfg)
    inputs = to_model_inputs(model, batch)
    with torch.no_grad():
        first = model(**inputs)["fm_loss"]
        second = model(**inputs)["fm_loss"]
    torch.testing.assert_close(first, second)

    prefix = model.prepare_prefix(
        batch["images"],
        batch["img_masks"],
        batch["lang_tokens"],
        batch["lang_masks"],
        batch["state"],
        model.object_features(
            batch["state"], batch["obj_sam_feat"], batch["obj_pose_vec"], batch["obj_valid_mask"]
        ),
        cot_ids=batch["cot_ids"],
        cot_mask=batch["cot_mask"],
        cot_phase_positions=batch["cot_phase_positions"],
        use_cache=True,
    )
    prefix_len = prefix.pad_masks.shape[1]
    model._set_moe_context(prefix.e_phase, prefix.obj_context)
    x = torch.randn(2, cfg.chunk_size, cfg.max_action_dim)
    t = torch.full((2,), 0.5)
    with torch.no_grad():
        cached = model.denoise_step(prefix.pad_masks, prefix.cache, x, t)
    assert all(layer.get_seq_length() == prefix_len for layer in prefix.cache.layers)
    with torch.no_grad():
        full = model.denoise_step(
            prefix.pad_masks,
            None,
            x,
            t,
            prefix_embs=prefix.embs,
            prefix_att_masks=prefix.att_masks,
        )
    torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-4)


def _objects_and_args(model, batch):
    objects = model.object_features(
        batch["state"], batch["obj_sam_feat"], batch["obj_pose_vec"], batch["obj_valid_mask"]
    )
    return objects, (batch["images"], batch["img_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"])


def test_generate_cot_batch_is_consistent():
    cfg = make_config(cot_max_tokens=4)
    model = make_model(cfg)
    batch = make_batch(cfg, bsize=3)
    objects, args = _objects_and_args(model, batch)
    out = model.generate_cot(*args, objects)
    assert out is not None
    ids, mask, positions = out
    assert ids.shape[0] == 3 and ids.shape[1] <= len(model._cot_prompt_ids) + cfg.cot_max_tokens
    assert mask.shape == ids.shape
    assert positions is None or positions.shape == (3,)


def test_generated_cot_matches_teacher_forced_training_logits():
    """Decoding uses the training attention pattern, so the training-path
    logits at every CoT position re-predict exactly the tokens that were generated."""
    cfg = make_config(cot_max_tokens=5)
    model = make_model(cfg)
    batch = make_batch(cfg)
    objects, args = _objects_and_args(model, batch)
    with torch.no_grad():
        ids, mask, _ = model.generate_cot(*args, objects)
        prefix = model.prepare_prefix(*args, objects, cot_ids=ids, cot_mask=mask, use_cache=False)
        start, end = prefix.cot_slice
        logits = model.cot_lm_head(prefix.out[:, start:end])
    n_prompt = len(model._cot_prompt_ids)
    # logits at position t predict token t+1 (training's shifted CE)
    pred = logits[:, max(n_prompt - 1, 0) : -1].argmax(-1)
    torch.testing.assert_close(pred, ids[:, max(n_prompt, 1) :])


def test_sde_log_probs_mask_unexecuted_steps():
    """Masked + complement-masked log-probs add up to the unmasked total."""
    cfg = make_config(use_reinflow=True)
    model = make_model(cfg)
    batch = make_batch(cfg)
    objects, args = _objects_and_args(model, batch)
    with torch.no_grad():
        _, latents, _ = model.sample_actions_sde(*args, objects=objects, num_steps=3, action_dim=5)
        prefix = model.prepare_prefix(*args, objects)
        model._set_moe_context(prefix.e_phase, prefix.obj_context)
        executed = torch.zeros(2, cfg.chunk_size, dtype=torch.bool)
        executed[:, : cfg.chunk_size // 2] = True
        full = model.compute_sde_log_probs(prefix, latents, 3, action_dim=5)
        part = model.compute_sde_log_probs(prefix, latents, 3, action_dim=5, action_is_pad=~executed)
        rest = model.compute_sde_log_probs(prefix, latents, 3, action_dim=5, action_is_pad=executed)
    torch.testing.assert_close(part + rest, full, rtol=1e-5, atol=1e-4)
    assert not torch.allclose(part, full)


def test_expert_usage_matches_reported_layers():
    cfg = make_config()
    model = make_model(cfg)
    batch = make_batch(cfg)
    model(**to_model_inputs(model, batch))
    usage = model.expert_usage()
    assert set(usage) == set(range(len(model._moe_layers)))
    for value in usage.values():
        torch.testing.assert_close(value.sum(), torch.tensor(1.0))


# ---------------------------------------------------------------------------
# ReinFlow SDE
# ---------------------------------------------------------------------------


def test_sde_shapes_and_logprob_replay():
    cfg = make_config(use_reinflow=True)
    model = make_model(cfg)
    batch = make_batch(cfg)
    objects = model.object_features(
        batch["state"], batch["obj_sam_feat"], batch["obj_pose_vec"], batch["obj_valid_mask"]
    )
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        final, latents, log_probs = model.sample_actions_sde(
            batch["images"],
            batch["img_masks"],
            batch["lang_tokens"],
            batch["lang_masks"],
            batch["state"],
            objects=objects,
            num_steps=3,
            action_dim=5,
            action_is_pad=batch["action_is_pad"],
            generator=generator,
            cot_ids=batch["cot_ids"],
            cot_mask=batch["cot_mask"],
            cot_phase_positions=batch["cot_phase_positions"],
        )
    assert final.shape == (2, cfg.chunk_size, cfg.max_action_dim)
    assert latents.shape == (4, 2, cfg.chunk_size, cfg.max_action_dim)
    assert log_probs.shape == (3, 2) and torch.isfinite(log_probs).all()
    assert torch.isfinite(final).all()

    prefix = model.prepare_prefix(
        batch["images"],
        batch["img_masks"],
        batch["lang_tokens"],
        batch["lang_masks"],
        batch["state"],
        objects,
        cot_ids=batch["cot_ids"],
        cot_mask=batch["cot_mask"],
        cot_phase_positions=batch["cot_phase_positions"],
    )
    model._set_moe_context(prefix.e_phase, prefix.obj_context)
    with torch.no_grad():
        replay = model.compute_sde_log_probs(
            prefix, latents, 3, action_dim=5, action_is_pad=batch["action_is_pad"]
        )
    torch.testing.assert_close(replay, log_probs, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Warm-start loading
# ---------------------------------------------------------------------------


def test_load_smolvla_state_upcycles_dense_weights():
    cfg = make_config()
    model = make_model(cfg)
    own = model.state_dict()
    seed = {k: v.clone() for k, v in own.items() if k.startswith("state_proj")}
    dense = {}
    for i, layer in enumerate(model.vlm_with_expert.lm_expert.layers):
        shared = layer.mlp.shared_experts[0]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            dense[f"vlm_with_expert.lm_expert.layers.{i}.mlp.{proj}.weight"] = getattr(
                shared, proj
            ).weight.detach().clone() + 0.1
    report = model.load_smolvla_state({**seed, **dense})
    assert report["mlp_tensors"] == 3 * len(model.vlm_with_expert.lm_expert.layers)
    assert torch.allclose(model.state_proj.weight, seed["state_proj.weight"])
    for i, layer in enumerate(model.vlm_with_expert.lm_expert.layers):
        torch.testing.assert_close(
            layer.mlp.shared_experts[0].gate_proj.weight,
            dense[f"vlm_with_expert.lm_expert.layers.{i}.mlp.gate_proj.weight"],
        )
        for expert in layer.mlp.routed_experts:
            assert torch.count_nonzero(expert.down_proj.weight) == 0


def test_flow_time_matches_smolvla_beta_schedule():
    """Training time follows SmolVLA's 0.999*Beta(1.5,1)+0.001, not U(0,1)."""
    model = make_model(make_config())
    torch.manual_seed(0)
    t = model.sample_time(200_000, "cpu")
    assert float(t.min()) >= 0.001 and float(t.max()) <= 1.0
    # E[Beta(1.5,1)] = 0.6 -> 0.999*0.6+0.001 = 0.6004 (uniform would give 0.5)
    assert abs(float(t.mean()) - 0.6004) < 3e-3


def test_cot_not_generated_until_cot_supervised():
    """An untrained CoT head must not inject tokens into the inference prefix."""
    cfg = make_config(cot_max_tokens=3)
    model = make_model(cfg)
    batch = make_batch(cfg)
    objects = model.object_features(
        batch["state"], batch["obj_sam_feat"], batch["obj_pose_vec"], batch["obj_valid_mask"]
    )
    args = (batch["images"], batch["img_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"])
    with torch.no_grad():
        model.sample_actions(*args, objects=objects, refresh_cot=True)
    assert model._cot_cache is None and not model.cot_inference_enabled

    model.train()
    model(**to_model_inputs(model, batch))  # carries cot_ids -> CoT-supervised step
    model.eval()
    assert int(model.cot_trained_steps) == 1 and model.cot_inference_enabled
    assert "cot_trained_steps" in model.state_dict()
    with torch.no_grad():
        model.sample_actions(*args, objects=objects, refresh_cot=True)
    assert model._cot_cache is not None


def test_cot_cache_is_not_reused_for_another_batch():
    """A cached trace from a batch of 2 must not be spliced into a batch of 1."""
    cfg = make_config(cot_max_tokens=3)
    model = make_model(cfg)
    batch = make_batch(cfg)
    model.train()
    model(**to_model_inputs(model, batch))  # CoT-supervised step enables inference
    model.eval()
    objects = model.object_features(batch["state"], batch["obj_sam_feat"], batch["obj_pose_vec"], batch["obj_valid_mask"])
    args = (batch["images"], batch["img_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"])
    with torch.no_grad():
        model.sample_actions(*args, objects=objects, refresh_cot=True)
        assert model._cot_cache[0].shape[0] == 2
        one = [a[:1] if torch.is_tensor(a) else [x[:1] for x in a] for a in args]
        obj1 = model.object_features(batch["state"][:1], batch["obj_sam_feat"][:1], batch["obj_pose_vec"][:1], batch["obj_valid_mask"][:1])
        model.sample_actions(*one, objects=obj1, refresh_cot=False)  # no refresh requested
    assert model._cot_cache[0].shape[0] == 1
