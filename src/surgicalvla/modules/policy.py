"""Skill-routed Mixture-of-Experts policy on top of SmolVLA.

The architecture in one paragraph: the pretrained SmolVLA vision-language stack
proposes a **chunk-level phase prior** over ``num_experts`` skills from the
images + instruction (:class:`~surgicalvla.modules.skills.PhaseHead`), and every
``moe_every_n_layers``-th FFN of the action expert becomes a
:class:`~surgicalvla.modules.moe.PhaseMoEFFN`: the pretrained dense FFN stays as
an always-on shared expert, and small bandwidth-limited experts are added next
to it, routed **per action token** (i.e. per chunk step) by a per-layer router
that is biased by that prior. Skill clusters are discovered unsupervised from
chunk kinematics and supervise both the prior and the pooled router logits.

See ``README.md`` ("Architecture") for the layer diagram and the honest list of
what this does and does not do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.import_utils import require_package

from surgicalvla.modules.moe import RouteState, moe_aux_losses, upcycle_to_moe
from surgicalvla.modules.nn import FP32Linear
from surgicalvla.modules.skills import PhaseHead, SkillClusterer

MOE_MODES = ("on", "off", "oracle")


@PreTrainedConfig.register_subclass("surgicalvla")
@dataclass
class SurgicalVLAConfig(SmolVLAConfig):
    # --- action chunking / training -------------------------------------
    n_action_steps: int = 25
    train_expert_only: bool = True
    delta_actions: bool = True  # regress action - current state (masked at grippers)
    fp32_action_head: bool = True
    early_step_weight: float = 1.0
    early_step_tau: float = 10.0
    uniform_time_mix: float = 0.3
    # --- MoE action expert ----------------------------------------------
    num_experts: int = 8
    moe_every_n_layers: int = 2
    moe_expert_ratio: float = 0.25
    moe_max_k: int = 2
    moe_top_p: float = 0.7
    moe_prior_bias: float = 1.0  # weight of the vision-language prior on each router
    moe_z_coef: float = 1e-3
    moe_load_balance_coef: float = 0.01
    # --- skill clustering / router supervision ---------------------------
    gripper_dims: list[int] = field(default_factory=lambda: [6])
    skill_loss_coef: float = 0.1
    skill_label_smoothing: float = 0.1
    skill_ema: float = 0.99
    skill_freeze_steps: int = 2000
    phase_dim: int = 256
    phase_queries: int = 4
    phase_heads: int = 4
    # teacher forcing of the phase prior early in training (1.0 -> tf_end)
    tf_start: float = 1.0
    tf_end: float = 0.0
    tf_decay_steps: int = 5000
    tf_bias: float = 12.0


class SurgicalFlowMatching(VLAFlowMatching):
    """``VLAFlowMatching`` + phase prior + MoE action expert + skill clusterer."""

    def __init__(self, config: SurgicalVLAConfig, rtc_processor=None):
        compile_requested, config.compile_model = config.compile_model, False
        super().__init__(config, rtc_processor=rtc_processor)

        if config.fp32_action_head:
            self.action_out_proj.__class__ = FP32Linear

        mask = torch.zeros(config.max_action_dim)
        for i in range(config.action_feature.shape[0]):
            if i not in config.gripper_dims:
                mask[i] = 1.0
        self.register_buffer("delta_mask", mask.view(1, 1, -1), persistent=False)

        self.route = RouteState(config.num_experts)
        self.phase_head = PhaseHead(
            self.vlm_with_expert.config.text_config.hidden_size,
            config.num_experts,
            config.phase_dim,
            config.phase_queries,
            config.phase_heads,
        )
        self.skills = SkillClusterer(
            config.num_experts,
            config.max_action_dim,
            config.gripper_dims,
            config.skill_ema,
            config.skill_freeze_steps,
        )
        upcycle_to_moe(
            self.vlm_with_expert,
            self.route,
            config.num_experts,
            config.moe_expert_ratio,
            config.moe_every_n_layers,
            config.moe_max_k,
            config.moe_top_p,
            config.moe_prior_bias,
        )
        self.vlm_with_expert.set_requires_grad()

        t = torch.arange(config.chunk_size, dtype=torch.float32)
        w = 1.0 + config.early_step_weight * torch.exp(-t / config.early_step_tau)
        self.register_buffer("step_weights", (w / w.mean()).view(1, -1, 1), persistent=False)

        config.compile_model = compile_requested
        if compile_requested:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    # ------------------------------------------------------------------ routing
    @property
    def moe_mode(self) -> str:
        """``"on"`` (learned routing) | ``"off"`` (shared FFN only) | ``"oracle"`` (skill labels)."""
        return self.route.mode

    @moe_mode.setter
    def moe_mode(self, value: str):
        if value not in MOE_MODES:
            raise ValueError(f"moe_mode must be one of {MOE_MODES}, got {value!r}")
        self.route.mode = value

    def sample_time(self, bsize, device):
        t = super().sample_time(bsize, device)
        u = torch.rand(bsize, device=device) * 0.998 + 0.001
        return torch.where(torch.rand(bsize, device=device) < self.config.uniform_time_mix, u, t)

    def _model_target(self, actions, state=None):
        """Map raw actions into the space the network predicts (and the clusterer sees).

        With ``delta_actions`` the target is ``action - state`` on the arm
        dimensions and the raw action on the gripper dimensions. Using this same
        transform for skill features and for the regression target is what keeps
        the two consistent -- cluster kinematics are computed on exactly the
        quantity the model is asked to produce.
        """
        if not self.config.delta_actions or state is None:
            return actions
        return actions - state[:, None, :].to(actions.dtype) * self.delta_mask.to(actions.dtype)

    def _raw_actions(self, predicted, state):
        if not self.config.delta_actions or state is None:
            return predicted
        return predicted + state[:, None, :].to(predicted.dtype) * self.delta_mask.to(predicted.dtype)

    @torch.no_grad()
    def set_skill_labels(self, actions, state=None, is_pad=None, train: bool | None = None):
        """Assign skill-cluster labels to this batch and store them on the route state."""
        target = self._model_target(actions, state)
        labels = self.skills.labels(target, is_pad, train=self.training if train is None else train)
        self.route.labels = labels
        return labels

    def _set_prior(self, embs, pad):
        """Chunk-level prior from the vision-language prefix, plus optional teacher forcing."""
        cfg = self.config
        logits = self.phase_head(embs, pad).float()
        prior = logits
        labels = self.route.labels
        usable = labels is not None and labels.shape[0] == logits.shape[0]

        if usable and self.training and cfg.tf_start > 0:
            progress = min(1.0, int(self.skills.steps) / max(1, cfg.tf_decay_steps))
            p_tf = cfg.tf_start + (cfg.tf_end - cfg.tf_start) * progress
            if p_tf > 0:
                take = torch.rand(logits.shape[0], device=logits.device) < p_tf
                if bool(take.any()):
                    hard = F.one_hot(labels, logits.shape[-1]).to(logits.dtype) * cfg.tf_bias
                    prior = torch.where(take[:, None], hard, prior)

        if self.route.mode == "oracle":
            if usable:
                self.route.override = F.one_hot(labels, logits.shape[-1]).to(logits.dtype)
        self.route.prior = prior

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        embs, pad, att = super().embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        self._set_prior(embs, pad)
        return embs, pad, att

    # ------------------------------------------------------------------ forward
    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        actions = self._model_target(actions, state)
        losses = super().forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        return losses * self.step_weights

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        try:
            out = super().sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
            )
        finally:
            self.route.reset()
        return self._raw_actions(out, state)

    def expert_usage(self) -> dict[int, torch.Tensor]:
        """Per-MoE-layer expert dispatch fractions (sums to 1 per layer)."""
        return self.route.usage_fractions()


class SurgicalVLAPolicy(SmolVLAPolicy):
    config_class = SurgicalVLAConfig
    name = "surgicalvla"

    def __init__(self, config: SurgicalVLAConfig, **kwargs):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = SurgicalFlowMatching(config, rtc_processor=self.rtc_processor)
        self._last_usage: dict[int, torch.Tensor] = {}
        self.reset()

    def forward(self, batch, noise=None, time=None, reduction: str = "mean", weights=None):
        """Flow-matching loss on the action chunk + auxiliary router losses.

        ``weights`` (shape ``(B,)``) turns the flow loss into a *per-sample
        weighted* mean instead of a plain mean:

        .. math:: L = \\sum_i |w_i| L_i / \\sum_i |w_i|

        This is how online RL injects group advantages -- a sampled chunk that
        led to a high-reward episode is imitated more strongly than one from a
        failed episode, and the ``sum(|w|)`` denominator keeps the gradient
        scale independent of how the advantages happen to be normalized. The
        auxiliary router losses are added **unweighted**: routing quality is a
        property of the batch, not of the sampled action.
        """
        cfg = self.config
        model = self.model
        route = model.route
        route.reset()
        is_pad = batch.get("action_is_pad")
        model.set_skill_labels(self.prepare_action(batch), self.prepare_state(batch), is_pad)
        try:
            if weights is None:
                loss, loss_dict = super().forward(batch, noise=noise, time=time, reduction=reduction)
            else:
                per_sample, loss_dict = super().forward(batch, noise=noise, time=time, reduction="none")
                w = torch.as_tensor(weights, device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
                if w.shape[0] != per_sample.shape[0]:
                    raise ValueError(
                        f"weights has {w.shape[0]} entries but the batch has {per_sample.shape[0]} samples"
                    )
                denom = w.abs().sum().clamp_min(1e-6)
                loss = (per_sample * w).sum() / denom
                loss_dict["adv_weighted_loss"] = float(loss.detach())
                loss_dict["adv_abs_weight"] = float(w.abs().mean())
            aux, metrics = moe_aux_losses(
                route,
                route.labels,
                skill_coef=cfg.skill_loss_coef,
                z_coef=cfg.moe_z_coef,
                load_balance_coef=cfg.moe_load_balance_coef,
                label_smoothing=cfg.skill_label_smoothing,
            )
            loss = loss + aux
            loss_dict.update(metrics)
            loss_dict["loss"] = float(loss.detach().mean())
            return loss, loss_dict
        finally:
            # keep the routing statistics (the trainer logs them after this call)
            # before clearing the per-forward state.
            self._last_usage = model.expert_usage()
            route.reset()

    def reset_skills(self):
        self.model.skills.reset()

    @torch.no_grad()
    def fit_skills(self, loader, max_batches: int = 200, iters: int = 30):
        """Fit the k-means skill clusters on chunk kinematics from ``loader``."""
        device = next(self.parameters()).device
        sk = self.model.skills
        feats = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            state = self.prepare_state(batch).to(device)
            target = self.model._model_target(self.prepare_action(batch).to(device), state)
            pad = batch.get("action_is_pad")
            feats.append(sk.features(target, None if pad is None else pad.to(device)))
        sk.fit(torch.cat(feats), iters=iters, warm=bool(sk.initialized))

    def expert_usage(self) -> dict[int, torch.Tensor]:
        """Per-MoE-layer expert dispatch fractions from the most recent training step."""
        return {k: v.detach().float().cpu() for k, v in self._last_usage.items()}
