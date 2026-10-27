"""SurgicalVLA: Surgical Vision-Language-Action model built on LeRobot."""

__version__ = "0.1.0"

from surgicalvla.data import (
    build_dataset,
    infer_features,
    resolve_camera_map,
)
from surgicalvla.evaluation import action_error, evaluate, rollout_policy
from surgicalvla.modules import (
    SurgicalVLAConfig,
    SurgicalFlowMatching,
)
try:
    from surgicalvla.modules.policy import SurgicalVLAPolicy
except Exception:
    try:
        from surgicalvla.modules import SurgicalVLAPolicy
    except Exception:
        SurgicalVLAPolicy = None
try:
    from surgicalvla.training import train_base_stage, train_surgical_stage
except Exception:
    train_base_stage = train_surgical_stage = None
try:
    from surgicalvla.utils.config import build_surgical_config, load_config
except Exception:
    build_surgical_config = load_config = None

__all__ = [
    "SurgicalVLAConfig",
    "SurgicalVLAPolicy",
    "SurgicalFlowMatching",
    "__version__",
    "action_error",
    "build_surgical_config",
    "build_dataset",
    "evaluate",
    "infer_features",
    "load_config",
    "resolve_camera_map",
    "rollout_policy",
    "train_base_stage",
    "train_surgical_stage",
]
