"""SurgicalVLA: object-centric 6-D MoE VLA built on LeRobot's SmolVLA."""

__version__ = "0.2.0"

from surgicalvla.data import build_dataset, infer_features, resolve_camera_map
from surgicalvla.modules import (
    SmolVLAMoEConfig,
    SmolVLAMoEModel,
    SmolVLAMoEPolicy,
    SurgicalFlowMatching,
    SurgicalVLAConfig,
    SurgicalVLAPolicy,
)
from surgicalvla.training import Trainer

__all__ = [
    "SmolVLAMoEConfig",
    "SmolVLAMoEModel",
    "SmolVLAMoEPolicy",
    "SurgicalFlowMatching",
    "SurgicalVLAConfig",
    "SurgicalVLAPolicy",
    "Trainer",
    "__version__",
    "build_dataset",
    "infer_features",
    "resolve_camera_map",
]
