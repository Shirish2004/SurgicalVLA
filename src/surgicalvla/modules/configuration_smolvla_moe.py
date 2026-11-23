"""SmolVLA config + object-6D tokens, async CoT, phase MoE, ReinFlow."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

PHASE_NAMES = ("approach", "grasp", "reorient", "handover")


@PreTrainedConfig.register_subclass("smolvla_moe")
@dataclass
class SmolVLAMoEConfig(SmolVLAConfig):
    """SmolVLA + object-6D tokens + CoT + phase-conditioned MoE."""

    # --- object-centric 6-D stream ---------------------------------------
    num_object_tokens: int = 4
    obj_sam_feat_dim: int = 256
    obj_pose_dim: int = 13  # [x1, y1, x2, y2, tx, ty, tz, r1..r6]
    num_objects: int = 2  # padded object slots (1 for single-arm, 2 for handover)
    obj_fourier_bands: int = 16
    obj_dropout: float = 0.15
    obj_jitter_translation: float = 0.01
    obj_jitter_rotation: float = 0.02
    obj_tracker_mode: str = "passthrough"  # passthrough | cached | mock | live

    # --- asynchronous System-2 chain of thought --------------------------
    use_cot: bool = True
    cot_max_tokens: int = 48
    cot_refresh_interval: int = 10
    cot_loss_weight: float = 0.1
    cot_prompt: str = "<cot> phase:"
    cot_stop: str = "</cot>"
    # Number of leading VLM layers whose K/V the action expert cross-attends to.
    # SmolVLA pairs expert layer i with VLM layer i (``get_model_layers``), so any value
    # below ``num_vlm_layers`` silently skips expert layers and breaks the pretrained
    # function (audit A1: 8/16 expert layers ran, actions off by 152% vs smolvla_base).
    # Only the full depth is valid; ``0``/``-1`` resolve to it. Kept for config round-trip.
    expert_cross_attn_vlm_layers: int = 0

    # --- phase-conditioned MoE action expert -----------------------------
    num_experts: int = 4
    num_shared_experts: int = 1
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01
    phase_loss_weight: float = 0.1
    num_phases: int = 4
    phase_names: tuple[str, ...] = PHASE_NAMES

    # --- auxiliary 6-D pose head -----------------------------------------
    pose_aux_loss_weight: float = 0.2
    pose_target_dim: int = 9  # relative translation (3) + continuous rotation (6)

    # --- ReinFlow / online RL --------------------------------------------
    use_reinflow: bool = False
    reinflow_min_std: float = 0.01
    reinflow_max_std: float = 0.15

    # --- integration with the existing SurgicalVLA training loop ---------
    # ``Trainer``/``chunk_metrics`` use this to exclude the gripper from smoothness
    # statistics; it does not change the model graph.
    gripper_dims: list[int] = field(default_factory=lambda: [6])

    def __post_init__(self):
        super().__post_init__()
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")
        if self.num_shared_experts < 1:
            raise ValueError(f"num_shared_experts must be >= 1, got {self.num_shared_experts}")
        if not 1 <= self.moe_top_k <= self.num_experts:
            raise ValueError(
                f"moe_top_k must be in [1, num_experts={self.num_experts}], got {self.moe_top_k}"
            )
        if self.num_phases != len(self.phase_names):
            raise ValueError(
                f"num_phases ({self.num_phases}) must match len(phase_names) ({len(self.phase_names)})"
            )
        if self.reinflow_min_std <= 0 or self.reinflow_max_std < self.reinflow_min_std:
            raise ValueError(
                "expected 0 < reinflow_min_std <= reinflow_max_std, got "
                f"({self.reinflow_min_std}, {self.reinflow_max_std})"
            )
        if self.expert_cross_attn_vlm_layers <= 0:
            self.expert_cross_attn_vlm_layers = self.num_vlm_layers
        if self.expert_cross_attn_vlm_layers != self.num_vlm_layers:
            raise ValueError(
                f"expert_cross_attn_vlm_layers={self.expert_cross_attn_vlm_layers} != num_vlm_layers="
                f"{self.num_vlm_layers}: SmolVLA interleaves expert and VLM layers one-to-one, so a "
                "shallower cross-attention depth would skip action-expert layers. Reduce "
                "num_vlm_layers instead."
            )
