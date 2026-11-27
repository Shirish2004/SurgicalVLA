"""SurgicalVLA model/policy: object-6D tokens + async CoT + phase MoE + ReinFlow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.policies.common.flow_matching import (
    FlowConvention,
    make_flow_matching_inputs,
    sample_time_beta,
)
from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding, make_att_2d_masks
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_tensor
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package
from torch import Tensor, nn

from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig
from surgicalvla.modules.aux_heads import AuxVisionHeads
from surgicalvla.modules.latent_cot import EMATargetEncoder, LatentReasoner
from surgicalvla.modules.moe_layers import (
    PhaseConditionedMoEFFN,
    ReinFlowNoiseHead,
    SwiGLUExpert,
    gaussian_entropy,
    masked_gaussian_log_prob,
)
from surgicalvla.modules.object_encoder import (
    ObjectCentric6DProjector,
    ObjectFeatures,
    build_object_tracker,
    geodesic_rotation_loss,
)


@dataclass
class PrefixState:
    embs: Tensor
    pad_masks: Tensor
    att_masks: Tensor
    cache: Any
    out: Tensor
    e_phase: Tensor
    obj_context: Tensor
    cot_slice: tuple[int, int] | None
    cot_ids: Tensor | None
    cot_mask: Tensor | None
    cot_phase_positions: Tensor | None
    latent_out: Tensor | None = None


class PoseAuxHead(nn.Module):
    def __init__(self, expert_dim: int, vlm_dim: int, out_dim: int = 9):
        super().__init__()
        self.vlm_pool = nn.Linear(vlm_dim, expert_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * expert_dim, 2 * expert_dim),
            nn.SiLU(),
            nn.Linear(2 * expert_dim, out_dim),
        )

    def forward(self, obj_context: Tensor, vlm_pooled: Tensor) -> Tensor:
        dtype = self.net[0].weight.dtype
        return self.net(torch.cat([obj_context.to(dtype), self.vlm_pool(vlm_pooled.to(dtype))], dim=-1))


def _crop_cache(cache: Any, length: int) -> None:
    """Restore a prefix-only KV cache after appending suffix tokens."""
    for layer in getattr(cache, "layers", []):
        delta = length - layer.get_seq_length()
        if hasattr(layer, "crop"):
            layer.crop(delta)
        else:
            layer.keys = layer.keys[:, :length]
            layer.values = layer.values[:, :length]


class SmolVLAMoEModel(nn.Module):
    def __init__(
        self,
        config: SmolVLAMoEConfig,
        rtc_processor=None,
        vlm_with_expert: SmolVLMWithExpertModel | None = None,
    ):
        super().__init__()
        self.config = config
        self.vlm_with_expert = vlm_with_expert or SmolVLMWithExpertModel(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            load_vlm_weights=config.load_vlm_weights,
            attention_mode=config.attention_mode,
            num_expert_layers=config.num_expert_layers,
            num_vlm_layers=config.num_vlm_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
            device=config.device if config.device is not None else "auto",
        )
        if config.vlm_lora_rank > 0:
            from surgicalvla.modules.lora import apply_vlm_lora

            self.vlm_lora_params = apply_vlm_lora(self.vlm_with_expert, config.vlm_lora_rank, config.vlm_lora_alpha,
                                                  parts=tuple(config.vlm_lora_parts))
        self.vlm_hidden_size = int(self.vlm_with_expert.config.text_config.hidden_size)
        self.expert_hidden_size = int(self.vlm_with_expert.expert_hidden_size)

        self.state_proj = nn.Linear(config.max_state_dim, self.vlm_hidden_size)
        self.action_in_proj = nn.Linear(config.max_action_dim, self.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.expert_hidden_size, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(2 * self.expert_hidden_size, self.expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(self.expert_hidden_size, self.expert_hidden_size)

        self.obj_projector = ObjectCentric6DProjector(
            hidden_size_vlm=self.vlm_hidden_size,
            hidden_size_expert=self.expert_hidden_size,
            sam_feat_dim=config.obj_sam_feat_dim,
            pose_dim=config.obj_pose_dim,
            num_tokens=config.num_object_tokens,
            fourier_bands=config.obj_fourier_bands,
            dropout=config.obj_dropout,
            jitter_translation=config.obj_jitter_translation,
            jitter_rotation=config.obj_jitter_rotation,
            layout=config.obj_token_layout,
        )
        self.phase_proj = nn.Linear(self.vlm_hidden_size, self.expert_hidden_size)
        self.null_phase = nn.Parameter(torch.zeros(self.expert_hidden_size))
        self.pose_head = PoseAuxHead(self.expert_hidden_size, self.vlm_hidden_size, config.pose_target_dim)
        self.phase_head = nn.Linear(self.expert_hidden_size, config.num_phases)
        aux_in = self.expert_hidden_size if config.aux_source == "expert" else self.vlm_hidden_size
        self.aux_heads = AuxVisionHeads(aux_in, config.aux_feat_dim) if config.aux_vision else None
        self.latent = LatentReasoner(self.vlm_hidden_size, config.latent_cot_tokens) if config.latent_cot else None
        self.latent_target = (
            EMATargetEncoder(self.vlm_with_expert, config.latent_ema_decay) if config.latent_cot else None
        )
        self._last_latent_slice: tuple[int, int] | None = None
        # optimisation steps taken; drives the explicit-to-latent CoT curriculum
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long), persistent=True)
        self.reinflow_head = ReinFlowNoiseHead(
            self.expert_hidden_size,
            self.expert_hidden_size,
            min_std=config.reinflow_min_std,
            max_std=config.reinflow_max_std,
            out_dim=config.max_action_dim,
        )
        self.cot_lm_head = self._build_cot_head()
        self.add_image_special_tokens = bool(getattr(config, "add_image_special_tokens", False))
        self.prefix_length = int(getattr(config, "prefix_length", 0))
        tokenizer = self._tokenizer()
        self.fake_image_token = getattr(tokenizer, "fake_image_token_id", None)
        self.global_image_token = getattr(tokenizer, "global_image_token_id", None)
        if self.add_image_special_tokens and None not in (self.fake_image_token, self.global_image_token):
            self.register_buffer(
                "global_image_start_token",
                torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "image_end_token", torch.tensor([self.fake_image_token], dtype=torch.long), persistent=False
            )
        else:
            self.add_image_special_tokens = False
            self.global_image_start_token = None
            self.image_end_token = None
        self._cot_prompt_ids = self._tokenize(config.cot_prompt)
        self._cot_stop_ids = self._tokenize(config.cot_stop)
        self._phase_token_ids = self._tokenize_phases()
        self._moe_layers = self._upcycle_expert_ffns()
        self._cot_cache: tuple[Tensor, Tensor, Tensor | None] | None = None
        # Number of optimisation steps that carried CoT supervision. Inference only
        # decodes CoT once this is > 0: a model fit without CoT tokens in its prefix must
        # not be fed untrained LM-head output at test time. Persistent, so it survives
        # save/load; checkpoints without this buffer load it as 0.
        self.register_buffer("cot_trained_steps", torch.zeros((), dtype=torch.long), persistent=True)
        self._cot_step = 0
        self.rtc_processor = rtc_processor
        if config.compile_model:
            raise ValueError("compile_model is not supported by SmolVLAMoEModel; run eager")

    # ---------------------------------------------------------------- construction
    def _build_cot_head(self) -> nn.Linear:
        vocab = int(self.vlm_with_expert.config.text_config.vocab_size)
        head = nn.Linear(self.vlm_hidden_size, vocab, bias=False)
        lm_head = getattr(getattr(self.vlm_with_expert, "vlm", None), "lm_head", None)
        if lm_head is not None and lm_head.weight.shape == head.weight.shape:
            with torch.no_grad():
                head.weight.copy_(lm_head.weight)
        return head

    def _tokenizer(self):
        return getattr(getattr(self.vlm_with_expert, "processor", None), "tokenizer", None)

    def _tokenize(self, text: str) -> list[int]:
        tok = self._tokenizer()
        if tok is None or not text:
            return []
        return list(tok.encode(text, add_special_tokens=False))

    def _tokenize_phases(self) -> list[int]:
        tok = self._tokenizer()
        if tok is None:
            return []
        ids = []
        for name in self.config.phase_names:
            enc = tok.encode(name, add_special_tokens=False)
            if enc:
                ids.append(int(enc[0]))
        return ids

    def _upcycle_expert_ffns(self) -> list[PhaseConditionedMoEFFN]:
        """Replace each action-expert FFN according to ``config.moe_variant``; returns the MoE layers."""
        layers = getattr(getattr(self.vlm_with_expert, "lm_expert", None), "layers", [])
        if not layers:
            raise RuntimeError("no action-expert layers found")
        variant = self.config.moe_variant
        out = []
        for i, layer in enumerate(layers):
            dense = getattr(layer, "mlp", None)
            if dense is None or isinstance(dense, (PhaseConditionedMoEFFN, SwiGLUExpert)) or variant == "dense":
                continue
            if variant == "dense_param_matched":
                width = dense.gate_proj.out_features
                extra = width * (self.config.num_shared_experts + self.config.num_experts - 1)
                rows = torch.cat([torch.arange(width), torch.full((extra,), -1)])
                layer.mlp = SwiGLUExpert.from_dense(dense, rows=rows)
                continue
            moe = PhaseConditionedMoEFFN(
                dense,
                hidden_size=self.expert_hidden_size,
                e_phase_dim=self.expert_hidden_size,
                obj_ctx_dim=self.expert_hidden_size,
                num_experts=self.config.num_experts,
                num_shared_experts=self.config.num_shared_experts,
                top_k=self.config.moe_top_k,
                num_phases=self.config.num_phases,
                variant=variant,
                seed=i,
            )
            layer.mlp = moe
            out.append(moe)
        return out

    # ---------------------------------------------------------------- prefix / CoT
    def _embed_text(self, tokens: Tensor) -> Tensor:
        emb = self.vlm_with_expert.embed_language_tokens(tokens)
        return emb * math.sqrt(emb.shape[-1])

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: Tensor,
        object_tokens: Tensor | None = None,
        object_token_mask: Tensor | None = None,
        cot_ids: Tensor | None = None,
        cot_mask: Tensor | None = None,
        latent_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[int, int] | None]:
        embs, pad, att = [], [], []
        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                # Upstream SmolVLA does not sqrt(d)-scale the image special tokens.
                start = self.global_image_start_token.to(device=img.device)[None].expand(img.shape[0], -1)
                start_emb = self.vlm_with_expert.embed_language_tokens(start)
                embs.append(start_emb)
                pad.append(torch.ones(start_emb.shape[:2], dtype=torch.bool, device=start_emb.device))
                att += [0] * start_emb.shape[1]
            img_emb = self.vlm_with_expert.embed_image(img)
            # Same op order/dtype as upstream: sqrt(d) is cast to the embedding dtype first
            # (bf16 rounds sqrt(960)=30.98 to 31.0), which the pretrained weights expect.
            img_emb = img_emb * torch.tensor(img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
            bsize, n_img = img_emb.shape[:2]
            embs.append(img_emb)
            pad.append(img_mask[:, None].expand(bsize, n_img))
            att += [0] * n_img
            if self.add_image_special_tokens:
                end = self.image_end_token.to(device=img.device)[None].expand(img.shape[0], -1)
                end_emb = self.vlm_with_expert.embed_language_tokens(end)
                embs.append(end_emb)
                pad.append(torch.ones(end_emb.shape[:2], dtype=torch.bool, device=end_emb.device))
                att += [0] * end_emb.shape[1]

        embs.append(self._embed_text(lang_tokens))
        pad.append(lang_masks)
        att += [0] * lang_tokens.shape[1]

        # Drop the object slots when no sample in the batch has a valid object: fully
        # masked tokens are mathematically inert but change the sequence length, which
        # changes bf16 kernel accumulation and with it exact parity with smolvla_base.
        if object_tokens is not None and object_token_mask is not None and not bool(object_token_mask.any()):
            object_tokens = None
        if object_tokens is not None:
            embs.append(object_tokens)
            if object_token_mask is None:
                object_token_mask = torch.ones(
                    object_tokens.shape[:2], dtype=torch.bool, device=object_tokens.device
                )
            pad.append(object_token_mask)
            att += [0] * object_tokens.shape[1]

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        pad.append(torch.ones(state_emb.shape[:2], dtype=torch.bool, device=state_emb.device))
        att += [1] * state_emb.shape[1]

        cot_slice = None
        if cot_ids is not None:
            start = sum(p.shape[1] for p in pad)
            cot = self._embed_text(cot_ids)
            if cot_mask is None:
                cot_mask = torch.ones(cot_ids.shape, dtype=torch.bool, device=cot_ids.device)
            embs.append(cot)
            pad.append(cot_mask)
            att += [1] * cot_ids.shape[1]
            cot_slice = (start, start + cot_ids.shape[1])

        self._last_latent_slice = None
        if latent_tokens is not None:
            # latent queries come last with causal attention: they read the whole prefix,
            # and nothing before them (including text CoT) depends on them
            start = sum(p.shape[1] for p in pad)
            embs.append(latent_tokens.to(embs[0].dtype))
            pad.append(torch.ones(latent_tokens.shape[:2], dtype=torch.bool, device=latent_tokens.device))
            att += [1] * latent_tokens.shape[1]
            self._last_latent_slice = (start, start + latent_tokens.shape[1])

        embs = torch.cat(embs, dim=1)
        pad = torch.cat(pad, dim=1)
        att = torch.tensor(att, dtype=torch.bool, device=pad.device)[None, :].expand(pad.shape[0], -1)
        if 0 < self.config.prefix_length and embs.shape[1] < self.config.prefix_length:
            embs = pad_tensor(embs, self.config.prefix_length, pad_value=0)
            pad = pad_tensor(pad, self.config.prefix_length, pad_value=0)
            att = pad_tensor(att, self.config.prefix_length, pad_value=0)
        return embs, pad, att, cot_slice

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=action_emb.device,
        ).to(dtype=action_emb.dtype)
        action_time = torch.cat([action_emb, time_emb[:, None, :].expand_as(action_emb)], dim=-1)
        action_time = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time)))
        bsize, seq_len = action_time.shape[:2]
        pad = torch.ones(bsize, seq_len, dtype=torch.bool, device=action_time.device)
        att = torch.ones(bsize, seq_len, dtype=torch.bool, device=action_time.device)
        return action_time, pad, att

    def _empty_objects(self, state: Tensor) -> ObjectFeatures:
        bsize = state.shape[0]
        device = state.device
        return self.obj_projector(
            torch.zeros(bsize, 0, self.config.obj_sam_feat_dim, device=device),
            torch.zeros(bsize, 0, self.config.obj_pose_dim, device=device),
            torch.zeros(bsize, 0, dtype=torch.bool, device=device),
        )

    def object_features(
        self,
        state: Tensor,
        sam_roi_feat: Tensor | None,
        obj_pose_vec: Tensor | None,
        obj_valid_mask: Tensor | None,
    ) -> ObjectFeatures:
        if sam_roi_feat is None or obj_pose_vec is None:
            return self._empty_objects(state)
        return self.obj_projector(sam_roi_feat, obj_pose_vec, obj_valid_mask)

    def _phase_from_hidden(
        self,
        prefix_out: Tensor,
        cot_slice: tuple[int, int] | None,
        positions: Tensor | None,
        pad_masks: Tensor,
        cot_mask: Tensor | None,
    ) -> Tensor:
        bsize = prefix_out.shape[0]
        if cot_slice is None:
            return self.null_phase[None, :].expand(bsize, -1)
        start, end = cot_slice
        if positions is None:
            if cot_mask is not None:
                positions = (cot_mask.sum(dim=1) - 1).clamp_min(0)
            else:
                positions = torch.full((bsize,), end - start - 1, device=prefix_out.device)
        positions = positions.to(prefix_out.device).long().clamp(0, end - start - 1)
        idx = start + positions
        hidden = prefix_out[torch.arange(bsize, device=prefix_out.device), idx]
        return self.phase_proj(hidden.to(self.phase_proj.weight.dtype))

    @torch.no_grad()
    def generate_cot(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: Tensor,
        objects: ObjectFeatures,
    ) -> tuple[Tensor, Tensor, Tensor | None] | None:
        """Greedy CoT decoding under exactly the training attention pattern.

        Each step re-embeds ``[images | text | objects | state | CoT so far]`` with
        :meth:`embed_prefix` and runs it through ``vlm_with_expert`` with the same
        ``make_att_2d_masks`` mask training uses (bidirectional perception block, causal
        CoT), so the logits at step ``k`` are the teacher-forced training logits. A plain
        causal decoder would instead hide the instruction from the image tokens. Re-encoding costs
        ``cot_max_tokens`` VLM passes per refresh; System 2 runs every
        ``cot_refresh_interval`` control steps, so this is off the action-chunk hot path.
        """
        if not self.config.use_cot:
            return None
        bsize = state.shape[0]
        device = state.device
        ids = torch.tensor(self._cot_prompt_ids, dtype=torch.long, device=device)[None].expand(bsize, -1)
        prompt_len = ids.shape[1]
        eos = getattr(self._tokenizer(), "eos_token_id", None)
        stop = self._cot_stop_ids
        phase_ids = torch.tensor(self._phase_token_ids, dtype=torch.long, device=device) if self._phase_token_ids else None
        phase_pos = None
        n_new = 0
        for step in range(self.config.cot_max_tokens):
            embs, pad, att, cot_slice = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state,
                object_tokens=objects.tokens, object_token_mask=objects.token_mask,
                cot_ids=ids if ids.shape[1] else None,
            )
            # use_cache=True: a prefix-only pass must run every layer as self-attention (the
            # cache-filling path, as in prepare_prefix); the cache itself is discarded
            outputs, _ = self.vlm_with_expert.forward(
                attention_mask=make_att_2d_masks(pad, att),
                position_ids=torch.cumsum(pad, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=True,
            )
            last = (cot_slice[1] if cot_slice is not None else pad.shape[1]) - 1
            hidden = outputs[0][:, last]
            token = self.cot_lm_head(hidden.to(self.cot_lm_head.weight.dtype)).argmax(dim=-1)
            ids = torch.cat([ids, token[:, None]], dim=1)
            n_new += 1
            if phase_pos is None and phase_ids is not None:
                match = (token[:, None] == phase_ids[None]).any(dim=1)
                if bool(match.any()):
                    phase_pos = torch.where(match, torch.full_like(token, prompt_len + step), torch.full_like(token, -1))
            if eos is not None and bool((token == eos).all()):
                break
            if stop and n_new >= len(stop):
                recent = ids[:, -len(stop):]
                if bool((recent == torch.tensor(stop, device=device)[None]).all()):
                    break
        if n_new == 0:
            return None
        mask = torch.ones(ids.shape, dtype=torch.bool, device=device)
        positions = None if phase_pos is None else phase_pos.clamp_min(prompt_len)
        return ids, mask, positions

    @property
    def cot_inference_enabled(self) -> bool:
        return bool(self.cot_trained_steps.item() > 0)

    def reset_cot(self) -> None:
        self._cot_cache = None
        self._cot_step = 0

    def prepare_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: Tensor,
        objects: ObjectFeatures,
        cot_ids: Tensor | None = None,
        cot_mask: Tensor | None = None,
        cot_phase_positions: Tensor | None = None,
        refresh_cot: bool = False,
        use_cache: bool | None = None,
    ) -> PrefixState:
        if self._cot_cache is not None and self._cot_cache[0].shape[0] != state.shape[0]:
            self._cot_cache = None  # cached for another batch: never reuse it
            refresh_cot = True
        if self.config.use_cot and cot_ids is None and refresh_cot and self.cot_inference_enabled:
            generated = self.generate_cot(images, img_masks, lang_tokens, lang_masks, state, objects)
            if generated is not None:
                self._cot_cache = generated
        if cot_ids is None and self.config.use_cot and self.cot_inference_enabled:
            cached = self._cot_cache
            if cached is not None:
                cot_ids, cot_mask, cot_phase_positions = cached

        embs, pad, att, cot_slice = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            object_tokens=objects.tokens,
            object_token_mask=objects.token_mask,
            cot_ids=cot_ids,
            cot_mask=cot_mask,
            latent_tokens=None if self.latent is None else self.latent.tokens(state.shape[0]),
        )
        latent_slice = self._last_latent_slice
        att_2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        outputs, cache = self.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=self.config.use_cache if use_cache is None else bool(use_cache),
        )
        prefix_out = outputs[0]
        e_phase = self._phase_from_hidden(prefix_out, cot_slice, cot_phase_positions, pad, cot_mask)
        return PrefixState(
            embs=embs,
            pad_masks=pad,
            att_masks=att,
            cache=cache,
            out=prefix_out,
            e_phase=e_phase,
            obj_context=objects.context,
            cot_slice=cot_slice,
            cot_ids=cot_ids,
            cot_mask=cot_mask,
            cot_phase_positions=cot_phase_positions,
            latent_out=None if latent_slice is None else prefix_out[:, latent_slice[0] : latent_slice[1]],
        )

    # ---------------------------------------------------------------- MoE context
    def _set_moe_context(self, e_phase: Tensor, obj_context: Tensor) -> None:
        for layer in self._moe_layers:
            layer.set_context(e_phase, obj_context)

    def _reset_moe_stats(self) -> None:
        for layer in self._moe_layers:
            layer.aux_loss = None
            layer.last_usage = None

    def sample_time(self, bsize: int, device) -> Tensor:
        """Flow time exactly as SmolVLA pretraining drew it: 0.999 * Beta(1.5, 1) + 0.001.

        A uniform draw would put less mass near the noise end (t -> 1) than the pretrained
        velocity field was fit on.
        """
        return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        *,
        objects: ObjectFeatures | None = None,
        cot_ids: Tensor | None = None,
        cot_mask: Tensor | None = None,
        cot_phase_positions: Tensor | None = None,
        subtask_phase_id: Tensor | None = None,
        target_relative_pose_6d: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        action_dim: int | None = None,
        loss_weights: Tensor | None = None,
        aux: dict[str, Tensor] | None = None,
        future_latent: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        if objects is None:
            objects = self._empty_objects(state)
        self._reset_moe_stats()
        x_t, u_t, model_time = make_flow_matching_inputs(
            actions, noise, time, convention=FlowConvention.NOISE_AT_ONE
        )
        prefix = self.prepare_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            objects,
            cot_ids=cot_ids,
            cot_mask=cot_mask,
            cot_phase_positions=cot_phase_positions,
            use_cache=True,
        )
        self._set_moe_context(prefix.e_phase, prefix.obj_context)

        suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, model_time)
        bsize, suffix_len = suffix_pad.shape
        prefix_len = prefix.pad_masks.shape[1]
        prefix_2d = prefix.pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        att_2d = torch.cat([prefix_2d, make_att_2d_masks(suffix_pad, suffix_att)], dim=2)
        pos = prefix.pad_masks.sum(1)[:, None] + torch.cumsum(suffix_pad, 1) - 1
        outputs, _ = self.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos,
            past_key_values=prefix.cache,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
        )
        if prefix.cache is not None:
            _crop_cache(prefix.cache, prefix_len)
        suffix_out = outputs[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        if action_is_pad is not None:
            losses = losses.masked_fill(action_is_pad[..., None], 0.0)
        if action_dim is not None:
            losses = losses[..., :action_dim]
        per_sample = losses.mean(dim=(1, 2))
        if loss_weights is None:
            fm_loss = per_sample.mean()
        else:
            w = loss_weights.to(per_sample.device, per_sample.dtype).reshape(-1)
            fm_loss = (per_sample * w).sum() / w.abs().sum().clamp_min(1e-6)
        out: dict[str, Tensor] = {"fm_loss": fm_loss, "fm_loss_per_sample": per_sample}

        if cot_ids is not None and prefix.cot_slice is not None and self.config.cot_loss_weight > 0:
            start, end = prefix.cot_slice
            logits = self.cot_lm_head(prefix.out[:, start:end].to(self.cot_lm_head.weight.dtype))
            target = cot_ids[:, 1:]
            mask = (cot_mask[:, 1:] if cot_mask is not None else torch.ones_like(target, dtype=torch.bool))
            ce = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                reduction="none",
            ).view(target.shape)
            out["cot_loss"] = (ce * mask).sum() / mask.sum().clamp_min(1)
            with torch.no_grad():
                hit = (logits[:, :-1].argmax(-1) == target) & mask
                out["cot_token_acc"] = hit.sum().float() / mask.sum().clamp_min(1)
                out["cot_trace_len"] = mask.sum(1).float().mean()
            if self.training:
                self.cot_trained_steps += 1

        balance = [m.aux_loss for m in self._moe_layers if m.aux_loss is not None]
        out["moe_aux_loss"] = torch.stack(balance).sum() if balance else fm_loss.new_zeros(())

        pooled = (prefix.out * prefix.pad_masks[..., None]).sum(1) / prefix.pad_masks.sum(1, keepdim=True).clamp_min(1)
        if subtask_phase_id is not None:
            # phase from the CoT's phase token when CoT is present, else from pooled perception
            feats = prefix.e_phase if prefix.cot_slice is not None else self.phase_proj(pooled.to(self.phase_proj.weight.dtype))
            out["phase_loss"] = F.cross_entropy(self.phase_head(feats.to(self.phase_head.weight.dtype)), subtask_phase_id.long())

        if target_relative_pose_6d is not None:
            pred = self.pose_head(prefix.obj_context, pooled)
            target_pose = target_relative_pose_6d.to(pred.dtype)
            pos_loss = F.smooth_l1_loss(pred[..., :3], target_pose[..., :3])
            rot_loss = geodesic_rotation_loss(pred[..., 3:], target_pose[..., 3:])
            out["pose_aux_loss"] = pos_loss + rot_loss

        if self.aux_heads is not None and aux:
            # "expert": mean of the action expert's outputs, so the targets shape trainable features
            # even with a frozen VLM; "prefix": the pooled VLM prefix
            src = suffix_out.mean(dim=1) if self.config.aux_source == "expert" else pooled
            out.update(self.aux_heads.losses(src.to(next(self.aux_heads.parameters()).dtype), aux))
        if self.latent is not None and future_latent is not None and prefix.latent_out is not None:
            out["latent_cot_loss"] = LatentReasoner.loss(self.latent.predict(prefix.latent_out), future_latent)
        # train_steps counts optimiser steps (SmolVLAMoEPolicy.on_optimizer_step), not forward
        # passes: gradient accumulation and diagnostic forwards must not advance the curriculum.

        total = out["fm_loss"].new_zeros(())
        for name, weight in self.loss_weights().items():
            if name in out:
                total = total + weight * out[name]
        out["loss"] = total
        return out

    def loss_weights(self) -> dict[str, float]:
        """Weight of every loss term in the total."""
        c = self.config
        return {"fm_loss": 1.0, "cot_loss": c.cot_loss_weight, "moe_aux_loss": c.moe_aux_loss_weight,
                "phase_loss": c.phase_loss_weight, "pose_aux_loss": c.pose_aux_loss_weight,
                "aux_feat_loss": c.aux_feat_weight, "aux_motion_loss": c.aux_motion_weight,
                "aux_track_loss": c.aux_track_weight, "aux_rel_loss": c.aux_rel_weight,
                "latent_cot_loss": c.latent_cot_weight}

    # ---------------------------------------------------------------- denoise
    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: Any,
        x_t: Tensor,
        timestep: Tensor,
        *,
        prefix_embs: Tensor | None = None,
        prefix_att_masks: Tensor | None = None,
        return_hidden: bool = False,
    ):
        suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad.shape[1]
        bsize, prefix_len = prefix_pad_masks.shape
        if past_key_values is not None:
            prefix_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_2d = torch.cat([prefix_2d, suffix_2d], dim=2)
            pos = prefix_pad_masks.sum(1)[:, None] + torch.cumsum(suffix_pad, 1) - 1
            outputs, _ = self.vlm_with_expert.forward(
                attention_mask=full_2d,
                position_ids=pos,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=True,
            )
        else:
            if prefix_embs is None:
                raise ValueError("prefix_embs is required when no KV cache is used")
            pad = torch.cat([prefix_pad_masks, suffix_pad], dim=1)
            att = torch.cat(
                [
                    prefix_att_masks
                    if prefix_att_masks is not None
                    else torch.ones_like(prefix_pad_masks),
                    suffix_att,
                ],
                dim=1,
            )
            outputs, _ = self.vlm_with_expert.forward(
                attention_mask=make_att_2d_masks(pad, att),
                position_ids=torch.cumsum(pad, 1) - 1,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
            )
        if past_key_values is not None:
            _crop_cache(past_key_values, prefix_len)
        hidden = outputs[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
        velocity = self.action_out_proj(hidden)
        return (velocity, hidden) if return_hidden else velocity

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        objects: ObjectFeatures | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        refresh_cot: bool = False,
        cot_ids: Tensor | None = None,
        cot_mask: Tensor | None = None,
        cot_phase_positions: Tensor | None = None,
    ) -> Tensor:
        if objects is None:
            objects = self._empty_objects(state)
        self._reset_moe_stats()
        prefix = self.prepare_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            objects,
            cot_ids=cot_ids,
            cot_mask=cot_mask,
            cot_phase_positions=cot_phase_positions,
            refresh_cot=refresh_cot,
            use_cache=True,
        )
        self._set_moe_context(prefix.e_phase, prefix.obj_context)
        if noise is None:
            noise = torch.randn(
                state.shape[0], self.config.chunk_size, self.config.max_action_dim, device=state.device
            )
        steps = self.config.num_steps if num_steps is None else int(num_steps)
        x = noise
        for step in range(steps):
            t = torch.full((state.shape[0],), 1.0 - step / steps, dtype=torch.float32, device=state.device)
            v = self.denoise_step(
                prefix.pad_masks,
                prefix.cache,
                x,
                t,
                prefix_embs=prefix.embs,
                prefix_att_masks=prefix.att_masks,
            )
            x = x + v * (-1.0 / steps)
        return x

    # ---------------------------------------------------------------- ReinFlow
    def _time_embedding(self, t: float, hidden: Tensor) -> Tensor:
        time = torch.full((hidden.shape[0],), t, dtype=torch.float32, device=hidden.device)
        return create_sinusoidal_pos_embedding(
            time, self.expert_hidden_size, self.config.min_period, self.config.max_period, device=hidden.device
        ).to(hidden.dtype)

    def _transition(self, prefix: PrefixState, x: Tensor, t: float) -> tuple[Tensor, Tensor, Tensor]:
        time = torch.full((x.shape[0],), t, dtype=torch.float32, device=x.device)
        velocity, hidden = self.denoise_step(
            prefix.pad_masks,
            prefix.cache,
            x,
            time,
            prefix_embs=prefix.embs,
            prefix_att_masks=prefix.att_masks,
            return_hidden=True,
        )
        log_std = self.reinflow_head(hidden, self._time_embedding(t, hidden))
        return velocity, hidden, log_std

    def sample_actions_sde(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        objects: ObjectFeatures | None = None,
        noise: Tensor | None = None,
        num_steps: int = 5,
        action_dim: int | None = None,
        action_is_pad: Tensor | None = None,
        refresh_cot: bool = False,
        generator: torch.Generator | None = None,
        cot_ids: Tensor | None = None,
        cot_mask: Tensor | None = None,
        cot_phase_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if objects is None:
            objects = self._empty_objects(state)
        self._reset_moe_stats()
        prefix = self.prepare_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            objects,
            cot_ids=cot_ids,
            cot_mask=cot_mask,
            cot_phase_positions=cot_phase_positions,
            refresh_cot=refresh_cot,
            use_cache=True,
        )
        self._set_moe_context(prefix.e_phase, prefix.obj_context)
        if noise is None:
            noise = torch.randn(
                state.shape[0],
                self.config.chunk_size,
                self.config.max_action_dim,
                device=state.device,
                generator=generator,
            )
        x = noise
        dt = -1.0 / num_steps
        latents = [x]
        log_probs = []
        for step in range(num_steps):
            t = 1.0 + step * dt
            velocity, _, log_std = self._transition(prefix, x, t)
            mu = x + velocity * dt
            eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            log_sigma = log_std + 0.5 * math.log(abs(dt))
            sigma = torch.exp(log_sigma)
            x = mu + sigma * eps
            log_probs.append(
                masked_gaussian_log_prob(x, mu, log_sigma, action_dim=action_dim, action_is_pad=action_is_pad)
            )
            latents.append(x)
        return x, torch.stack(latents), torch.stack(log_probs)

    def compute_sde_log_probs(
        self,
        prefix: PrefixState,
        trajectory_latents: Tensor,
        num_steps: int,
        action_dim: int | None = None,
        action_is_pad: Tensor | None = None,
        return_entropy: bool = False,
    ):
        dt = -1.0 / num_steps
        out = []
        ent = []
        for step in range(num_steps):
            t = 1.0 + step * dt
            x = trajectory_latents[step]
            velocity, _, log_std = self._transition(prefix, x, t)
            mu = x + velocity * dt
            log_sigma = log_std + 0.5 * math.log(abs(dt))
            out.append(
                masked_gaussian_log_prob(
                    trajectory_latents[step + 1],
                    mu,
                    log_sigma,
                    action_dim=action_dim,
                    action_is_pad=action_is_pad,
                )
            )
            if return_entropy:
                ent.append(gaussian_entropy(log_sigma, action_dim=action_dim, action_is_pad=action_is_pad))
        log_probs = torch.stack(out)
        return (log_probs, torch.stack(ent)) if return_entropy else log_probs

    # ---------------------------------------------------------------- loading
    def load_smolvla_state(self, state_dict: dict[str, Tensor]) -> dict[str, list[str]]:
        """Load SmolVLA weights; every replaced action-expert FFN is re-cut from its dense weights."""
        own = self.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = self.load_state_dict(matched, strict=False)
        copied = 0
        for i, layer in enumerate(getattr(self.vlm_with_expert.lm_expert, "layers", [])):
            mlp = getattr(layer, "mlp", None)
            keys = [f"vlm_with_expert.lm_expert.layers.{i}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")]
            if hasattr(mlp, "load_dense") and all(k in state_dict for k in keys):
                mlp.load_dense(*(state_dict[k] for k in keys))
                copied += 1
        lm_head = getattr(getattr(self.vlm_with_expert, "vlm", None), "lm_head", None)
        if lm_head is not None and lm_head.weight.shape == self.cot_lm_head.weight.shape:
            with torch.no_grad():
                self.cot_lm_head.weight.copy_(lm_head.weight)
        return {"missing": list(missing), "unexpected": list(unexpected), "mlp_tensors": 3 * copied}

    def expert_usage(self) -> dict[int, Tensor]:
        return {
            i: layer.last_usage.detach().float().cpu()
            for i, layer in enumerate(self._moe_layers)
            if layer.last_usage is not None
        }


def ablate_object_inputs(batch: dict, mode: str) -> dict:
    """Object-token usage probe: ``zero`` hides every object; ``shuffle`` permutes the object
    slots of each sample (the slot embedding then marks the wrong object)."""
    valid = batch["obj_valid_mask"]
    if mode == "zero":
        return {"obj_valid_mask": torch.zeros_like(valid)}
    perm = torch.stack([torch.randperm(valid.shape[1], device=valid.device) for _ in range(valid.shape[0])])
    take = lambda x: torch.gather(x, 1, perm.view(*perm.shape, *([1] * (x.dim() - 2))).expand_as(x))  # noqa: E731
    return {"obj_sam_feat": take(batch["obj_sam_feat"]), "obj_pose_vec": take(batch["obj_pose_vec"]), "obj_valid_mask": take(valid)}


class SmolVLAMoEPolicy(SmolVLAPolicy):
    config_class = SmolVLAMoEConfig
    name = "smolvla_moe"

    def __init__(self, config: SmolVLAMoEConfig, **kwargs):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = SmolVLAMoEModel(config, rtc_processor=self.rtc_processor)
        self.object_tracker = build_object_tracker(config)
        self.loss_terms: dict[str, Tensor] = {}
        self.reset()

    def reset(self):
        super().reset()
        self.model.reset_cot()

    @torch.no_grad()
    def vlm_feature_drift(self, batch: dict[str, Tensor]) -> float | None:
        """Forgetting check for VLM adapters: mean cosine between the pooled VLM prefix with and
        without LoRA on ``batch`` (1 = the VLM's representation is unchanged)."""
        if self.config.vlm_lora_rank <= 0:
            return None
        from surgicalvla.modules.lora import lora_disabled

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        args = (images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state, self._objects(batch))

        def pooled():
            p = self.model.prepare_prefix(*args, use_cache=True)
            m = p.pad_masks[..., None].to(p.out.dtype)
            return (p.out * m).sum(1) / m.sum(1).clamp_min(1)

        adapted = pooled()
        with lora_disabled(self.model.vlm_with_expert):
            frozen = pooled()
        return float(F.cosine_similarity(adapted.float(), frozen.float(), dim=-1).mean())

    def attach_live_perception(self, extractor, camera_key: str | None = None) -> None:
        """Compute object inputs from the current frame at inference (evaluation, deployment).

        ``extractor``: ``surgicalvla.perception.object_inputs.ObjectInputExtractor``. The frame is
        ``batch[camera_key]`` (default: the first policy camera) and the phrase comes from
        ``batch["task"]`` the way the cache made it (rule, or the grounded VLM when the rule
        finds none). Not a submodule: it is frozen and not saved with the policy.
        """
        object.__setattr__(self, "_live_perception", extractor)
        self._live_camera = camera_key or next(iter(self.config.image_features))
        self._phrases: dict[str, str] = {}

    def _phrase(self, task: str) -> str:
        if task not in self._phrases:
            from surgicalvla.pseudolabels.nounphrase import rule_phrase

            phrase = rule_phrase(task)
            if phrase is None:
                from surgicalvla.pseudolabels.nounphrase import VLMPhraseExtractor

                if getattr(self, "_phrase_vlm", None) is None:
                    object.__setattr__(self, "_phrase_vlm", VLMPhraseExtractor(device=str(self.config.device or "cuda")))
                phrase = self._phrase_vlm(task)
            self._phrases[task] = phrase or ""
        return self._phrases[task]

    def _live_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        import numpy as np

        images = batch[self._live_camera]
        tasks = batch.get("task") or [""] * images.shape[0]
        tasks = [tasks] * images.shape[0] if isinstance(tasks, str) else list(tasks)
        key = "grid" if self.config.object_input == "grid" else "obj"
        feats, poses, valid = [], [], []
        for img, task in zip(images, tasks, strict=True):
            frame = (img.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            out = self._live_perception(frame[None], self._phrase(str(task)))
            feats.append(torch.from_numpy(out[f"{key}_feat"][0].astype(np.float32)))
            poses.append(torch.from_numpy(out[f"{key}_pose"][0].astype(np.float32)))
            valid.append(torch.from_numpy(out["obj_valid"][0]) if key == "obj" else torch.ones(4, dtype=torch.bool))
        dev = images.device
        return {"obj_sam_feat": torch.stack(feats).to(dev), "obj_pose_vec": torch.stack(poses).to(dev),
                "obj_valid_mask": torch.stack(valid).to(dev)}

    def _objects(self, batch: dict[str, Tensor]) -> ObjectFeatures:
        if self.config.object_input != "none" and "obj_sam_feat" not in batch:
            if getattr(self, "_live_perception", None) is None:
                raise RuntimeError("object_input is on but the batch has no object features and no live perception "
                                   "is attached (attach_live_perception)")
            batch = {**batch, **self._live_inputs(batch)}
        if self.config.object_input != "none" and self.config.object_token_ablation != "none":
            batch = {**batch, **ablate_object_inputs(batch, self.config.object_token_ablation)}
        tracked = self.object_tracker(batch)
        state = self.prepare_state(batch)
        return self.model.object_features(state, tracked.sam_roi_feat, tracked.obj_pose_vec, tracked.obj_valid_mask)

    def _should_refresh_cot(self) -> bool:
        if not self.config.use_cot:
            return False
        steps = self.config.cot_curriculum_steps
        if steps > 0 and int(self.model.train_steps) >= steps:
            return False  # the curriculum ended latent-only: no explicit text at inference
        interval = max(1, self.config.cot_refresh_interval)
        refresh = self.model._cot_cache is None or self.model._cot_step % interval == 0
        self.model._cot_step += 1
        return refresh

    def _get_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None, **kwargs) -> Tensor:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        objects = self._objects(batch)
        actions = self.model.sample_actions(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            objects=objects,
            noise=noise,
            refresh_cot=self._should_refresh_cot(),
        )
        action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :action_dim]
        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        return actions

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
        weights: Tensor | None = None,
        keep_loss_terms: bool = False,
    ) -> tuple[Tensor, dict[str, float]]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        objects = self._objects(batch)
        future_latent = None
        if self.model.latent_target is not None and all(f"future.{k}" in batch for k in self.config.image_features):
            future = {**batch, **{k: batch[f"future.{k}"] for k in self.config.image_features}}
            future_latent = self.model.latent_target(self.prepare_images(future)[0])
        cot_ids, cot_mask = batch.get("cot_input_ids"), batch.get("cot_attention_mask")
        steps = self.config.cot_curriculum_steps
        if steps > 0 and cot_ids is not None and self.training:
            p_explicit = max(0.0, 1.0 - float(self.model.train_steps) / steps)
            if float(torch.rand(())) >= p_explicit:
                cot_ids = cot_mask = None  # latent-only step
        out = self.model.forward(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            actions,
            noise=noise,
            time=time,
            objects=objects,
            cot_ids=cot_ids,
            cot_mask=cot_mask,
            cot_phase_positions=batch.get("cot_phase_positions") if cot_ids is not None else None,
            subtask_phase_id=batch.get("subtask_phase_id"),
            target_relative_pose_6d=batch.get("target_relative_pose_6d"),
            action_is_pad=batch.get("action_is_pad"),
            action_dim=self.config.action_feature.shape[0],
            loss_weights=weights,
            aux={k: v for k, v in batch.items() if k.startswith("aux.")} or None,
            future_latent=future_latent,
        )
        loss = out["loss"]
        if keep_loss_terms:  # weighted terms as they enter the total (gradient diagnostics)
            w = self.model.loss_weights()
            self.loss_terms = {k: w[k] * v for k, v in out.items() if k in w and v.dim() == 0 and v.requires_grad}
        if reduction == "none":
            loss = out["fm_loss_per_sample"]
        metrics = {k: float(v.detach()) for k, v in out.items() if v.numel() == 1}
        metrics["losses_after_rm_padding"] = metrics["fm_loss"]
        return loss, metrics

    def expert_usage(self) -> dict[int, Tensor]:
        return self.model.expert_usage()

    def update_ema(self) -> None:
        """Call after each optimiser step: advances the latent-CoT EMA target encoder."""
        if self.model.latent_target is not None:
            self.model.latent_target.update()

    def on_optimizer_step(self) -> None:
        """Call once after each optimiser step: advances the step counter that drives the
        explicit -> latent CoT curriculum, and the latent-CoT EMA target encoder."""
        self.model.train_steps += 1
        if self.model.latent_target is not None:
            self.model.latent_target.update()

    def sample_actions_sde(self, batch: dict[str, Tensor], num_steps: int = 5, generator=None, return_cot: bool = False):
        """SDE rollout. With ``return_cot`` also returns the CoT that conditioned it (replayed in the update)."""
        self.eval()
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        objects = self._objects(batch)
        action_dim = self.config.action_feature.shape[0]
        final, latents, log_probs = self.model.sample_actions_sde(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            objects=objects,
            num_steps=num_steps,
            action_dim=action_dim,
            generator=generator,
            refresh_cot=self._should_refresh_cot(),
        )
        out = (final[:, :, :action_dim], latents, log_probs)
        return (*out, self.model._cot_cache if self.model.cot_inference_enabled else None) if return_cot else out

    def sde_log_probs(
        self,
        batch: dict[str, Tensor],
        latents: Tensor,
        num_steps: int = 5,
        return_entropy: bool = False,
        action_is_pad: Tensor | None = None,
        cot: tuple | None = None,
    ):
        """Replay log-probs of recorded SDE latents.

        ``action_is_pad`` masks chunk steps that were not executed: they never reach the
        environment, so their noise carries no reward signal. ``cot`` is the CoT recorded
        at rollout time; without it the prefix is built CoT-free, never from whatever
        happens to be cached now.
        """
        was_training = self.model.obj_projector.training
        self.model.obj_projector.eval()
        try:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            objects = self._objects(batch)
            cot_ids, cot_mask, cot_pos = (None, None, None) if cot is None else (t if t is None else t.to(state.device) for t in cot)
            cached, self.model._cot_cache = self.model._cot_cache, None  # never replay a stale cache
            try:
                prefix = self.model.prepare_prefix(
                    images,
                    img_masks,
                    batch[OBS_LANGUAGE_TOKENS],
                    batch[OBS_LANGUAGE_ATTENTION_MASK],
                    state,
                    objects,
                    cot_ids=cot_ids,
                    cot_mask=cot_mask,
                    cot_phase_positions=cot_pos,
                    use_cache=True,
                )
            finally:
                self.model._cot_cache = cached
            self.model._set_moe_context(prefix.e_phase, prefix.obj_context)
            return self.model.compute_sde_log_probs(
                prefix,
                latents.to(state.device),
                num_steps,
                action_dim=self.config.action_feature.shape[0],
                action_is_pad=None if action_is_pad is None else action_is_pad.to(state.device),
                return_entropy=return_entropy,
            )
        finally:
            self.model.obj_projector.train(was_training)

    def enable_live_perception(self, device: str | None = None, **kwargs):
        from surgicalvla.vision import build_live_stack

        stack = build_live_stack(
            device=device,
            num_objects=self.config.num_objects,
            sam_feat_dim=self.config.obj_sam_feat_dim,
            **kwargs,
        )
        stack.attach_to(self.object_tracker)
        object.__setattr__(self, "perception", stack)
        return stack

    @classmethod
    def from_smolvla_pretrained(cls, pretrained_name_or_path: str, config: SmolVLAMoEConfig, **kwargs):
        from pathlib import Path

        policy = cls(config, **kwargs)
        path = Path(pretrained_name_or_path)
        safetensors = path / "model.safetensors"
        if safetensors.is_file():
            from safetensors.torch import load_file

            state = {
                k.removeprefix("model."): v
                for k, v in load_file(str(safetensors)).items()
                if k.startswith("model.")
            }
        else:
            base = SmolVLAPolicy.from_pretrained(pretrained_name_or_path)
            state = base.model.state_dict()
        policy.moe_load_report = policy.model.load_smolvla_state(state)
        return policy


__all__ = ["PrefixState", "SmolVLAMoEModel", "SmolVLAMoEPolicy"]
