"""Online RL: roll out the policy in LIBERO and update it from group advantages.

The pieces, in the order the trainer uses them:

* :mod:`surgicalvla.rl.protocol` -- the wire format shared with the simulator.
* :class:`~surgicalvla.rl.env.LiberoClient` -- spawns/talks to
  ``scripts/libero_server.py``, which runs in the separate ``libero`` pixi
  environment (LIBERO pins Python 3.10 / numpy 1.22, this package needs 3.13 /
  numpy 2).
* :func:`~surgicalvla.rl.rollout.rollout_episode` -- observations in, sampled
  chunks + per-episode reward out.
* :func:`~surgicalvla.rl.grpo.group_advantages` -- rewards to per-episode
  advantages; :class:`~surgicalvla.rl.grpo.FrozenReference` -- the optional
  flow-loss-gap penalty.

Nothing here imports LeRobot at module import time, so the wire format and the
advantage math can be unit-tested without a checkpoint or a GPU.
"""

from surgicalvla.rl.env import LiberoClient, LiberoError, Observation, StepResult, TaskInfo
from surgicalvla.rl.grpo import FrozenReference, episode_reward, group_advantages, group_weights, kl_penalty
from surgicalvla.rl.reinflow_grpo import (
    compute_handover_reward,
    gaussian_entropy,
    group_normalized_advantages,
    grpo_surrogate,
    sequence_log_prob,
    so3_geodesic_angle,
)
from surgicalvla.rl.rollout import (
    ChunkRecord,
    EpisodeResult,
    collate,
    env_action,
    frame_panel,
    lift,
    policy_state,
    observation_batch,
    rollout_episode,
)

__all__ = [
    "ChunkRecord",
    "EpisodeResult",
    "FrozenReference",
    "LiberoClient",
    "LiberoError",
    "Observation",
    "StepResult",
    "TaskInfo",
    "collate",
    "compute_handover_reward",
    "env_action",
    "episode_reward",
    "frame_panel",
    "gaussian_entropy",
    "grpo_surrogate",
    "group_advantages",
    "group_normalized_advantages",
    "group_weights",
    "kl_penalty",
    "lift",
    "policy_state",
    "observation_batch",
    "rollout_episode",
    "sequence_log_prob",
    "so3_geodesic_angle",
]
