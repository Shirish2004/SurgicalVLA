"""Turn a policy + the LIBERO simulator into advantage-weighted training samples.

One *episode* is a sequence of *decision points*. At each decision point the
policy observes the simulator, samples an action chunk, and executes the first
``n_exec_steps`` actions of it; that chunk becomes one training sample, and the
episode's final reward becomes its advantage.

Two representation details are worth stating outright because they are easy to
get wrong and impossible to see in the loss curve:

**The chunk we train on is the normalized chunk.** ``predict_action_chunk``
returns actions in the model's own (mean/std-normalized) space -- the delta
action transform is applied inside that space -- so the sampled chunk is
already what ``policy.forward`` wants as ``batch["action"]``. Feeding the
post-processed *real* actions back in would silently train on a different
distribution. Real simulator actions therefore only exist for ``env.step``.

**The 7-D LIBERO observation is tiled into the checkpoint's 14-D layout.**
The checkpoint was trained on a bimanual ALOHA-style dataset, so its state,
action and gripper-dimension indices (``gripper_dims=[6, 13]``) are all 14-D
while LIBERO's ``OSC_POSE`` interface is 7-D. Tiling the single arm into both
halves keeps every index consistent (dim 6 and dim 13 become the same gripper)
and is far less destructive than zero-padding, but it is still a projection of
a bimanual space onto a single arm -- the RL stage has to learn through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def lift(vector: np.ndarray | torch.Tensor, dim: int):
    """Tile a ``(d,)`` vector up to ``dim`` so 7-D LIBERO fits the 14-D checkpoint."""
    vector = np.asarray(vector)
    if vector.shape[-1] == dim:
        return vector
    if vector.shape[-1] > dim:
        raise ValueError(f"cannot lift {vector.shape[-1]} dims into {dim}: the source is wider than the target")
    reps = int(np.ceil(dim / vector.shape[-1]))
    return np.tile(vector, (*([1] * (vector.ndim - 1)), reps))[..., :dim]


def observation_batch(config, obs, camera_map: dict[str, str], device="cpu") -> dict:
    """One simulator observation -> the raw batch dict ``preprocessor`` expects.

    Batched on purpose: ``AddBatchDimensionProcessorStep`` only adds dims to 1-D
    state / 3-D images, so a ``(C, H, W)`` image and ``(D,)`` state remain
    unbatched and the pipeline returns ``(1, ...)``... which it does not, so we
    build the ``(1, ...)`` versions ourselves. That way the preprocessor is
    called once per decision point instead of once per sample, and the same
    function serves single-episode rollouts and batched validation.
    """
    missing = [k for k in config.image_features if k not in camera_map]
    if missing:
        raise KeyError(
            f"camera_map is missing {missing}; it must map every policy camera "
            f"({list(config.image_features)}) to a simulator camera ({sorted(obs.images)})."
        )
    batch: dict[str, object] = {}
    for policy_key in config.image_features:
        image = obs.images[camera_map[policy_key]]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
        batch[policy_key] = tensor.unsqueeze(0)
    state_dim = int(config.input_features["observation.state"].shape[0])
    state = np.asarray(obs.state, dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(lift(state, state_dim)).float().unsqueeze(0)
    batch["task"] = [obs.language]
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def env_action(
    action: torch.Tensor | np.ndarray,
    action_slice: tuple[int, int] = (0, 7),
    scale: float = 1.0,
    gripper_sign: float = -1.0,
    gripper_dims: tuple[int, ...] = (6,),
) -> np.ndarray:
    """One post-processed 14-D action -> the 7-D ``OSC_POSE`` action LIBERO takes.

    ``gripper_sign`` exists because the two conventions disagree: robosuite
    closes on ``+1`` while the checkpoint's gripper convention (and the
    simulator's own reported proprioceptive openness) is open on ``+1``. The
    default flips it; if a run learns nothing, check this before blaming the
    policy.
    """
    a = torch.as_tensor(action, dtype=torch.float32).detach().reshape(-1).cpu().numpy()
    lo, hi = action_slice
    out = np.array(a[lo:hi], dtype=np.float32) * float(scale)
    for dim in gripper_dims:
        if 0 <= dim < out.shape[0]:
            out[dim] *= float(gripper_sign)
    return np.clip(out, -1.0, 1.0)


@dataclass
class ChunkRecord:
    """One decision point: what the policy saw, what it chose, what happened."""

    batch: dict[str, torch.Tensor]
    chunk: torch.Tensor
    actions: np.ndarray
    advantage: float = 0.0
    latents: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None

    def to(self, device) -> ChunkRecord:
        self.batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in self.batch.items()}
        return self


@dataclass
class EpisodeResult:
    task: str
    success: bool
    reward: float
    steps: int
    decisions: int
    records: list[ChunkRecord] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)
    final_observation: object | None = None

    @property
    def length(self) -> int:
        return self.steps


def _snapshot(batch: dict) -> dict[str, torch.Tensor]:
    """Detach a preprocessed batch and park it on the CPU (it must outlive ``obs``)."""
    return {k: (v.detach().to("cpu") if torch.is_tensor(v) else v) for k, v in batch.items()}


def _flatten(values: list) -> list:
    out = []
    for value in values:
        out.extend(value if isinstance(value, (list, tuple)) else [value])
    return out


def rollout_episode(
    policy,
    client,
    *,
    suite: str,
    task: str,
    camera_map: dict[str, str],
    device="cuda",
    episode: int = 0,
    seed: int = 0,
    n_exec_steps: int = 25,
    max_steps: int | None = None,
    max_decisions: int | None = None,
    action_slice: tuple[int, int] = (0, 7),
    action_scale: float = 1.0,
    gripper_sign: float = -1.0,
    env_gripper_dims: tuple[int, ...] = (6,),
    frame_every: int = 0,
    frame_camera: str | None = None,
    generator: torch.Generator | None = None,
    collect: bool = True,
    sde_steps: int = 0,
) -> EpisodeResult:
    """Roll the policy out in the simulator and return the recorded decision points.

    ``episode`` selects the simulator's init state from the task's ``.pruned_init``
    set, and ``seed`` seeds the environment; together they make a rollout
    reproducible. Gradients are not tracked: the samples this produces are the
    *data* for the update that follows, and advantages are not a differentiable
    function of the policy anyway.
    """
    cfg = policy.config
    preprocessor = policy._rl_preprocessor  # attached by train_rl.build_policy()
    postprocessor = policy._rl_postprocessor

    policy.reset()
    obs = client.reset(suite, task, episode=episode, seed=seed)
    records: list[ChunkRecord] = []
    frames: list[np.ndarray] = []
    cam = frame_camera or camera_map[list(cfg.image_features)[0]]

    steps, success, decisions = 0, False, 0

    with torch.no_grad():
        while True:
            if max_decisions is not None and decisions >= max_decisions:
                break
            raw = observation_batch(cfg, obs, camera_map, device=device)
            batch = preprocessor(raw)
            noise = torch.randn(
                1, cfg.chunk_size, cfg.max_action_dim, device=device, generator=generator, dtype=torch.float32
            )
            latents = log_probs = None
            if sde_steps > 0 and hasattr(policy, "sample_actions_sde"):
                chunk, latents, log_probs = policy.sample_actions_sde(
                    batch, num_steps=sde_steps, generator=generator
                )
            else:
                chunk = policy.predict_action_chunk(batch, noise=noise)  # (1, T, action_dim) normalized
            real = postprocessor(chunk)  # (1, T, action_dim) in dataset action units
            chunk = chunk[0].detach()

            # Execute the first `n` steps of the chunk, stopping early if the
            # episode ends. Only the steps actually executed are recorded, so the
            # chunk we train on is the one the reward actually applies to.
            n = max(1, min(n_exec_steps, int(real.shape[1])))
            sent: list[np.ndarray] = []
            stop = False
            for i in range(n):
                act = env_action(
                    real[0, i],
                    action_slice=action_slice,
                    scale=action_scale,
                    gripper_sign=gripper_sign,
                    gripper_dims=env_gripper_dims,
                )
                result = client.step(act)
                sent.append(act)
                steps += 1
                if frame_every and steps % frame_every == 0:
                    frames.append(result.observation.images[cam])
                if result.success:
                    success = True
                if result.done or success or (max_steps is not None and steps >= max_steps):
                    stop = True
                    break

            if collect:
                actions = np.stack(sent) if sent else np.zeros((0, action_slice[1] - action_slice[0]))
                records.append(
                    ChunkRecord(
                        batch=_snapshot(batch),
                        chunk=chunk,
                        actions=actions,
                        latents=None if latents is None else latents.detach().to("cpu"),
                        log_probs=None if log_probs is None else log_probs.detach().to("cpu"),
                    )
                )
            decisions += 1
            if stop:
                break
            obs = result.observation

    # The simulator already clips at its own horizon; a policy that survives it
    # without succeeding has failed, which is what `success=False` says.
    return EpisodeResult(
        task=task,
        success=success,
        reward=float(success),
        steps=steps,
        decisions=decisions,
        records=records,
        frames=frames,
        final_observation=obs,
    )


def collate(records: list[ChunkRecord], policy) -> dict:
    """Stack per-decision-point samples into one training batch."""
    if not records:
        raise ValueError("collate() got no records -- the rollout produced no decision points")
    keys = [k for k in records[0].batch if all(k in r.batch for r in records)]
    batch = {}
    for k in keys:
        values = [r.batch[k] for r in records]
        # `task` is a list of instruction strings; everything else is a tensor
        batch[k] = torch.cat(values, dim=0) if torch.is_tensor(values[0]) else _flatten(values)
    dim = int(policy.config.action_feature.shape[0])
    batch["action"] = torch.stack([r.chunk[:, :dim] for r in records]).float()
    batch["action_is_pad"] = torch.zeros(
        batch["action"].shape[:2], dtype=torch.bool, device=batch["action"].device
    )
    return {k: (v.to(policy.config.device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def frame_panel(frames: list[np.ndarray], max_frames: int = 8) -> torch.Tensor:
    """Downsample a rollout's frames into a ``(N, 3, H, W)`` float tensor for logging."""
    if not frames:
        return torch.zeros(0, 3, 1, 1)
    stride = max(1, len(frames) // max_frames)
    picked = frames[::stride][:max_frames]
    return torch.stack([torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float() / 255.0 for f in picked])
