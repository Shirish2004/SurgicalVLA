"""SurgicalVLA model/policy: object-6D tokens + async CoT + phase MoE + ReinFlow."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs
from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding, make_att_2d_masks
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_tensor
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package
from torch import Tensor, nn

from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig
from surgicalvla.modules.moe_layers import (
    PhaseConditionedMoEFFN,
    ReinFlowNoiseHead,
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
        )
        self.phase_proj = nn.Linear(self.vlm_hidden_size, self.expert_hidden_size)
        self.null_phase = nn.Parameter(torch.zeros(self.expert_hidden_size))
        self.pose_head = PoseAuxHead(self.expert_hidden_size, self.vlm_hidden_size, config.pose_target_dim)
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
        layers = getattr(getattr(self.vlm_with_expert, "lm_expert", None), "layers", [])
        out = []
        for layer in layers:
            dense = getattr(layer, "mlp", None)
            if dense is None or isinstance(dense, PhaseConditionedMoEFFN):
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
            )
            layer.mlp = moe
            out.append(moe)
        if not out:
            raise RuntimeError("no action-expert FFN was found to up-cycle")
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
    ) -> tuple[Tensor, Tensor, Tensor, tuple[int, int] | None]:
        embs, pad, att = [], [], []
        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                start = self.global_image_start_token.to(device=img.device)[None].expand(img.shape[0], -1)
                start_emb = self._embed_text(start)
                embs.append(start_emb)
                pad.append(torch.ones(start_emb.shape[:2], dtype=torch.bool, device=start_emb.device))
                att += [0] * start_emb.shape[1]
            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb * math.sqrt(img_emb.shape[-1])
            bsize, n_img = img_emb.shape[:2]
            embs.append(img_emb)
            pad.append(img_mask[:, None].expand(bsize, n_img))
            att += [0] * n_img
            if self.add_image_special_tokens:
                end = self.image_end_token.to(device=img.device)[None].expand(img.shape[0], -1)
                end_emb = self._embed_text(end)
                embs.append(end_emb)
                pad.append(torch.ones(end_emb.shape[:2], dtype=torch.bool, device=end_emb.device))
                att += [0] * end_emb.shape[1]

        embs.append(self._embed_text(lang_tokens))
        pad.append(lang_masks)
        att += [0] * lang_tokens.shape[1]

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
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor | None] | None:
        if not self.config.use_cot:
            return None
        vlm = getattr(self.vlm_with_expert, "vlm", None)
        text_model = getattr(getattr(vlm, "model", None), "text_model", None)
        if text_model is None:
            return None
        device = prefix_embs.device
        prompt = torch.tensor(self._cot_prompt_ids, dtype=torch.long, device=device)
        prompt = prompt[None, :].expand(batch_size, -1) if prompt.numel() else None
        embs = prefix_embs if prompt is None else torch.cat([prefix_embs, self._embed_text(prompt)], dim=1)
        try:
            embs = embs.to(dtype=text_model.get_input_embeddings().weight.dtype)
        except AttributeError:
            embs = embs.to(dtype=next(text_model.parameters()).dtype)
        pad = prefix_pad_masks if prompt is None else torch.cat(
            [prefix_pad_masks, torch.ones(batch_size, prompt.shape[1], dtype=torch.bool, device=device)], dim=1
        )
        pos = torch.cumsum(pad, dim=1) - 1
        out = text_model(
            inputs_embeds=embs,
            attention_mask=pad.long(),
            position_ids=pos,
            past_key_values=_new_cache(),
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1]
        cache = out.past_key_values
        generated = []
        phase_pos = None
        eos = getattr(self._tokenizer(), "eos_token_id", None)
        stop_ids = torch.tensor(self._cot_stop_ids, dtype=torch.long, device=device) if self._cot_stop_ids else None
        phase_ids = torch.tensor(self._phase_token_ids, dtype=torch.long, device=device) if self._phase_token_ids else None
        prompt_len = 0 if prompt is None else prompt.shape[1]
        for step in range(self.config.cot_max_tokens):
            token = self.cot_lm_head(hidden.to(self.cot_lm_head.weight.dtype)).argmax(dim=-1)
            generated.append(token)
            if phase_pos is None and phase_ids is not None:
                match = (token[:, None] == phase_ids[None]).any(dim=1)
                if bool(match.any()):
                    phase_pos = torch.where(
                        match, torch.full_like(token, prompt_len + step), torch.full_like(token, -1)
                    )
            if eos is not None and bool((token == eos).all()):
                break
            if stop_ids is not None and len(generated) >= len(self._cot_stop_ids):
                recent = torch.stack(generated[-len(self._cot_stop_ids) :], dim=1)
                if bool((recent == stop_ids[None]).all(dim=1).all()):
                    break
            pad = torch.cat([pad, torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1)
            pos = pos[:, -1:] + 1
            out = text_model(
                inputs_embeds=self._embed_text(token[:, None]),
                attention_mask=pad.long(),
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
            )
            hidden = out.last_hidden_state[:, -1]
            cache = out.past_key_values
        if not generated:
            return None
        gen = torch.stack(generated, dim=1)
        cot_ids = gen if prompt is None else torch.cat([prompt, gen], dim=1)
        cot_mask = torch.ones(cot_ids.shape, dtype=torch.bool, device=device)
        positions = None if phase_pos is None else phase_pos.clamp_min(prompt_len)
        return cot_ids, cot_mask, positions

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
        if (
            self.config.use_cot
            and cot_ids is None
            and refresh_cot
        ):
            base, base_pad, _, _ = self.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                object_tokens=objects.tokens,
                object_token_mask=objects.token_mask,
            )
            generated = self.generate_cot(base, base_pad, state.shape[0])
            if generated is not None:
                self._cot_cache = generated
        if cot_ids is None and self.config.use_cot:
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
        )
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
        )

    # ---------------------------------------------------------------- expert depth
    @contextmanager
    def _expert_depth(self):
        vlm = self.vlm_with_expert
        old = getattr(vlm, "num_vlm_layers", None)
        depth = self.config.expert_cross_attn_vlm_layers
        if old is not None and 0 < depth < old:
            vlm.num_vlm_layers = depth
        try:
            yield
        finally:
            if old is not None:
                vlm.num_vlm_layers = old

    def _set_moe_context(self, e_phase: Tensor, obj_context: Tensor) -> None:
        for layer in self._moe_layers:
            layer.set_context(e_phase, obj_context)

    def _reset_moe_stats(self) -> None:
        for layer in self._moe_layers:
            layer.aux_loss = None
            layer.last_usage = None

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
    ) -> dict[str, Tensor]:
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = torch.rand(actions.shape[0], device=actions.device).clamp(0.001, 0.999)
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
        with self._expert_depth():
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

        aux = [m.aux_loss for m in self._moe_layers if m.aux_loss is not None]
        out["moe_aux_loss"] = torch.stack(aux).sum() if aux else fm_loss.new_zeros(())

        if subtask_phase_id is not None:
            phase_losses = [
                F.cross_entropy(m.phase_classifier(prefix.e_phase), subtask_phase_id)
                for m in self._moe_layers
            ]
            out["phase_loss"] = torch.stack(phase_losses).mean()

        if target_relative_pose_6d is not None:
            pooled = (prefix.out * prefix.pad_masks[..., None]).sum(1) / prefix.pad_masks.sum(
                1, keepdim=True
            ).clamp_min(1)
            pred = self.pose_head(prefix.obj_context, pooled)
            target_pose = target_relative_pose_6d.to(pred.dtype)
            pos_loss = F.smooth_l1_loss(pred[..., :3], target_pose[..., :3])
            rot_loss = geodesic_rotation_loss(pred[..., 3:], target_pose[..., 3:])
            out["pose_aux_loss"] = pos_loss + rot_loss

        total = out["fm_loss"]
        total = total + self.config.cot_loss_weight * out.get("cot_loss", total.new_zeros(()))
        total = total + self.config.moe_aux_loss_weight * out["moe_aux_loss"]
        total = total + self.config.phase_loss_weight * out.get("phase_loss", total.new_zeros(()))
        total = total + self.config.pose_aux_loss_weight * out.get("pose_aux_loss", total.new_zeros(()))
        out["loss"] = total
        return out

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
        with self._expert_depth():
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
        own = self.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = self.load_state_dict(matched, strict=False)
        layers = getattr(self.vlm_with_expert.lm_expert, "layers", [])
        copied = 0
        prefix = "vlm_with_expert.lm_expert.layers."
        for key, value in state_dict.items():
            if not key.startswith(prefix) or ".mlp." not in key:
                continue
            head, _, tail = key.partition(".mlp.")
            try:
                idx = int(head.rsplit(".layers.", 1)[1])
            except (IndexError, ValueError):
                continue
            if idx >= len(layers):
                continue
            moe = getattr(layers[idx], "mlp", None)
            if not isinstance(moe, PhaseConditionedMoEFFN):
                continue
            proj, _, attr = tail.partition(".")
            for expert in list(moe.shared_experts) + list(moe.routed_experts):
                target = getattr(expert, proj, None)
                param = getattr(target, attr, None)
                if param is not None and param.shape == value.shape:
                    with torch.no_grad():
                        param.copy_(value)
            copied += 1
        for moe in self._moe_layers:
            for expert in moe.routed_experts:
                with torch.no_grad():
                    expert.down_proj.weight.zero_()
        lm_head = getattr(getattr(self.vlm_with_expert, "vlm", None), "lm_head", None)
        if lm_head is not None and lm_head.weight.shape == self.cot_lm_head.weight.shape:
            with torch.no_grad():
                self.cot_lm_head.weight.copy_(lm_head.weight)
        return {"missing": list(missing), "unexpected": list(unexpected), "mlp_tensors": copied}

    def expert_usage(self) -> dict[int, Tensor]:
        return {
            i: layer.last_usage.detach().float().cpu()
            for i, layer in enumerate(self._moe_layers)
            if layer.last_usage is not None
        }


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
        self.reset()

    def reset(self):
        super().reset()
        self.model.reset_cot()

    def _objects(self, batch: dict[str, Tensor]) -> ObjectFeatures:
        tracked = self.object_tracker(batch)
        state = self.prepare_state(batch)
        return self.model.object_features(state, tracked.sam_roi_feat, tracked.obj_pose_vec, tracked.obj_valid_mask)

    def _should_refresh_cot(self) -> bool:
        if not self.config.use_cot:
            return False
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
    ) -> tuple[Tensor, dict[str, float]]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        objects = self._objects(batch)
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
            cot_ids=batch.get("cot_input_ids"),
            cot_mask=batch.get("cot_attention_mask"),
            cot_phase_positions=batch.get("cot_phase_positions"),
            subtask_phase_id=batch.get("subtask_phase_id"),
            target_relative_pose_6d=batch.get("target_relative_pose_6d"),
            action_is_pad=batch.get("action_is_pad"),
            action_dim=self.config.action_feature.shape[0],
            loss_weights=weights,
        )
        loss = out["loss"]
        if reduction == "none":
            loss = out["fm_loss_per_sample"]
        metrics = {k: float(v.detach()) for k, v in out.items() if v.numel() == 1}
        metrics["losses_after_rm_padding"] = metrics["fm_loss"]
        return loss, metrics

    def expert_usage(self) -> dict[int, Tensor]:
        return self.model.expert_usage()

    def sample_actions_sde(self, batch: dict[str, Tensor], num_steps: int = 5, generator=None):
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
        return final[:, :, :action_dim], latents, log_probs

    def sde_log_probs(self, batch: dict[str, Tensor], latents: Tensor, num_steps: int = 5, return_entropy: bool = False):
        was_training = self.model.obj_projector.training
        self.model.obj_projector.eval()
        try:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            objects = self._objects(batch)
            prefix = self.model.prepare_prefix(
                images,
                img_masks,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                state,
                objects,
                use_cache=True,
            )
            self.model._set_moe_context(prefix.e_phase, prefix.obj_context)
            return self.model.compute_sde_log_probs(
                prefix,
                latents.to(state.device),
                num_steps,
                action_dim=self.config.action_feature.shape[0],
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


def _new_cache():
    try:
        from transformers import DynamicCache

        return DynamicCache()
    except Exception:  # transformers not installed (unit-test stubs provide their own)
        return None


__all__ = ["PrefixState", "SmolVLAMoEModel", "SmolVLAMoEPolicy"]
