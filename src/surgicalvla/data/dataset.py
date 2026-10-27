"""Builds a LeRobotDataset (local root or Hub repo_id, any codebase_version LeRobotDataset supports)
with delta_timestamps matching a SmolVLA-family config, and resolves raw dataset camera/feature keys
to the names the policy config expects.

Frame-window logic only depends on `config.observation_delta_indices` / `config.action_delta_indices`
(SmolVLAConfig) and the dataset's own fps/feature metadata, not on a specific dataset's layout or
format version -- version compatibility is handled by LeRobotDataset itself.
"""

import warnings

from torch.utils.data import Dataset

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION


def infer_features(meta: LeRobotDatasetMetadata) -> dict[str, PolicyFeature]:
    """Raw-dataset-key -> PolicyFeature, inferred from dataset metadata (visual/state/action only)."""
    features = {}
    for key, spec in meta.features.items():
        shape = tuple(spec["shape"])
        if key.startswith("observation.image") or spec.get("dtype") in ("image", "video"):
            ftype = FeatureType.VISUAL
        elif key == ACTION:
            ftype = FeatureType.ACTION
        elif key.startswith("observation."):
            ftype = FeatureType.STATE
        else:
            continue
        features[key] = PolicyFeature(type=ftype, shape=shape)
    return features


def resolve_camera_map(meta: LeRobotDatasetMetadata, camera_map: dict[str, str] | None) -> dict[str, str]:
    """dataset-key -> config-key. Explicit map wins; unmapped keys pass through unchanged."""
    dataset_cams = [k for k in meta.features if k.startswith("observation.image")]
    camera_map = dict(camera_map or {})
    unmapped = [k for k in dataset_cams if k not in camera_map]
    if unmapped:
        warnings.warn(
            f"No dataset.camera_map entry for {unmapped}; using their dataset names as-is. "
            "If loading pretrained SmolVLA/SurgicalVLA weights, these must match the checkpoint's "
            "camera feature names exactly or the policy will not find them in the batch."
        )
    return camera_map


class _RenamedDataset(Dataset):
    """Renames feature keys in every sample: dataset-key -> config-key."""

    def __init__(self, base: Dataset, key_map: dict[str, str]):
        self.base, self.key_map = base, key_map

    def __len__(self):
        return len(self.base)

    def _rename(self, key: str) -> str:
        if key in self.key_map:
            return self.key_map[key]
        for src, dst in self.key_map.items():
            if key == f"{src}_is_pad":  # keep an image key and its pad-mask companion in sync
                return f"{dst}_is_pad"
        return key

    def __getitem__(self, idx):
        return {self._rename(k): v for k, v in self.base[idx].items()}

    def __getattr__(self, name):
        return getattr(self.base, name)


def _delta_timestamps(config, meta: LeRobotDatasetMetadata) -> dict[str, list[float]]:
    fps = meta.fps
    obs_dt = [i / fps for i in config.observation_delta_indices]
    act_dt = [i / fps for i in config.action_delta_indices]
    deltas = {}
    for key in meta.features:
        if key == ACTION:
            deltas[key] = act_dt
        elif key.startswith("observation."):
            deltas[key] = obs_dt
    return deltas


def build_dataset(
    repo_id: str,
    config,
    streaming: bool = False,
    streaming_buffer_size: int = 1000,
    streaming_seed: int = 42,
    root: str | None = None,
    episodes: list[int] | None = None,
    camera_map: dict[str, str] | None = None,
):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = _delta_timestamps(config, meta)
    key_map = resolve_camera_map(meta, camera_map)

    if streaming:
        try:
            from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
        except ImportError as e:
            raise ImportError(
                "run.streaming=true but StreamingLeRobotDataset is unavailable in this lerobot install."
            ) from e
        dataset = StreamingLeRobotDataset(
            repo_id,
            delta_timestamps=delta_timestamps,
            shuffle_buffer_size=streaming_buffer_size,
            seed=streaming_seed,
        )
    else:
        dataset = LeRobotDataset(repo_id, root=root, episodes=episodes, delta_timestamps=delta_timestamps)

    return _RenamedDataset(dataset, key_map) if key_map else dataset