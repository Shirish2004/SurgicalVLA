from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from surgicalvla.modules.moe import RouteState, upcycle_to_moe
from surgicalvla.modules.skills import PhaseHead, SkillClusterer, select_experts


@PreTrainedConfig.register_subclass("surgicalvla")
@dataclass
class SurgicalVLAConfig(SmolVLAConfig):
    n_action_steps: int = 25
    train_expert_only: bool = True
    num_experts: int = 8
    gripper_dims: list[int] = field(default_factory=lambda: [6])
    moe_every_n_layers: int = 2
    moe_expert_ratio: float = 0.25
    moe_max_k: int = 2
    moe_top_p: float = 0.7
    moe_z_coef: float = 1e-3
    skill_loss_coef: float = 0.1
    skill_label_smoothing: float = 0.1
    skill_ema: float = 0.99
    skill_freeze_steps: int = 2000
    tf_start: float = 1.0
    tf_end: float = 0.3
    tf_decay_steps: int = 5000
    phase_dim: int = 256
    phase_queries: int = 4
    early_step_weight: float = 1.0
    early_step_tau: float = 10.0
    uniform_time_mix: float = 0.3
    delta_actions: bool = True
    fp32_action_head: bool = True


class FP32Linear(nn.Linear):
    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())


class SurgicalFlowMatching(VLAFlowMatching):
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
        self.route = RouteState()
        self.phase_head = PhaseHead(
            self.vlm_with_expert.config.text_config.hidden_size,
            config.num_experts,
            config.phase_dim,
            config.phase_queries,
        )
        self.skills = SkillClusterer(
            config.num_experts, config.max_action_dim, config.gripper_dims, config.skill_ema, config.skill_freeze_steps
        )
        upcycle_to_moe(
            self.vlm_with_expert, self.route, config.num_experts, config.moe_expert_ratio, config.moe_every_n_layers
        )
        self.vlm_with_expert.set_requires_grad()
        t = torch.arange(config.chunk_size, dtype=torch.float32)
        w = 1.0 + config.early_step_weight * torch.exp(-t / config.early_step_tau)
        self.register_buffer("step_weights", (w / w.mean()).view(1, -1, 1), persistent=False)
        self._labels = None
        self._aux = {}
        self._usage = None
        self.moe_mode = "on"

        config.compile_model = compile_requested
        if compile_requested:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def sample_time(self, bsize, device):
        t = super().sample_time(bsize, device)
        u = torch.rand(bsize, device=device) * 0.998 + 0.001
        return torch.where(torch.rand(bsize, device=device) < self.config.uniform_time_mix, u, t)

    def prime(self, actions, is_pad=None):
        self._labels = self.skills.labels(actions, is_pad, train=self.training)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        embs, pad, att = super().embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        self._route(embs, pad)
        return embs, pad, att

    def _route(self, embs, pad):
        cfg = self.config
        logits = self.phase_head(embs, pad).float()
        probs = F.softmax(logits, dim=-1)
        labels = self._labels
        aux = {"z": torch.logsumexp(logits, dim=-1).pow(2).mean()}
        if labels is not None and labels.shape[0] == probs.shape[0]:
            aux["skill_ce"] = F.cross_entropy(logits, labels, label_smoothing=cfg.skill_label_smoothing)
            aux["skill_acc"] = (logits.argmax(-1) == labels).float().mean().detach()
            if self.training:
                progress = min(1.0, int(self.skills.steps) / max(1, cfg.tf_decay_steps))
                p_tf = cfg.tf_start + (cfg.tf_end - cfg.tf_start) * progress
                tf = torch.rand(probs.shape[0], device=probs.device) < p_tf
                probs = torch.where(tf[:, None], F.one_hot(labels, probs.shape[-1]).float(), probs)
        if self.moe_mode == "oracle" and labels is not None and labels.shape[0] == probs.shape[0]:
            probs = F.one_hot(labels, probs.shape[-1]).float()
        self._aux = aux
        self.route.weights = None if self.moe_mode == "off" else select_experts(probs, cfg.moe_max_k, cfg.moe_top_p)
        self._usage = None if self.route.weights is None else self.route.weights.detach().sum(0)

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        if self.config.delta_actions:
            actions = actions - state[:, None, :].to(actions.dtype) * self.delta_mask.to(actions.dtype)
        losses = super().forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        return losses * self.step_weights

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if self.moe_mode != "oracle":
            self._labels = None
        try:
            out = super().sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)
        finally:
            self.route.weights = None
        if self.config.delta_actions:
            out = out + state[:, None, :].to(out.dtype) * self.delta_mask.to(out.dtype)
        return out

    def pop_aux(self):
        aux, self._aux, self._labels = self._aux, {}, None
        self.route.weights = None
        return aux

    def expert_usage(self):
        return {} if self._usage is None else {0: self._usage.cpu()}


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
        self.reset()

    def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        self.model.prime(self.prepare_action(batch), batch.get("action_is_pad"))
        loss, loss_dict = super().forward(batch, noise=noise, time=time, reduction=reduction)
        aux = self.model.pop_aux()
        extra = self.config.moe_z_coef * aux["z"]
        loss_dict["moe_z"] = aux["z"].item()
        if "skill_ce" in aux:
            extra = extra + self.config.skill_loss_coef * aux["skill_ce"]
            loss_dict["skill_ce"] = aux["skill_ce"].item()
            loss_dict["skill_acc"] = aux["skill_acc"].item()
        usage = self.model._usage
        if usage is not None:
            usage = usage.cpu()
            p = usage / usage.sum().clamp_min(1e-6)
            loss_dict["route_entropy"] = -(p * p.clamp_min(1e-6).log()).sum().item()
        loss = loss + extra
        loss_dict["loss"] = loss.mean().item()
        return loss, loss_dict

    def reset_skills(self):
        self.model.skills.reset()

    @torch.no_grad()
    def fit_skills(self, loader, max_batches: int = 200, iters: int = 30):
        device = next(self.parameters()).device
        sk = self.model.skills
        feats = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            a = self.prepare_action({ACTION: batch[ACTION].to(device)})
            pad = batch.get("action_is_pad")
            feats.append(sk.features(a, None if pad is None else pad.to(device)))
        sk.fit(torch.cat(feats), iters=iters, warm=bool(sk.initialized))

    def expert_usage(self):
        return self.model.expert_usage()