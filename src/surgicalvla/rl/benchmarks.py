"""Benchmark contracts: a policy may only be rolled out in the space it was trained in.

Audit A10: the RL configs ran a RoboTwin (aloha, 14-D joint-position) checkpoint
with RoboTwin normalisation stats in LIBERO (7-D end-effector deltas), slicing
actions ``[0:7]`` and feeding ``agentview`` into two camera slots. Nothing about
that run was interpretable. :func:`validate_rl_setup` makes the combination
impossible to configure: the state/action layout, normalisation statistics and
cameras must all belong to the benchmark the simulator implements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    state_dim: int
    action_dim: int
    state_layout: str
    action_layout: str
    cameras: tuple[str, ...]
    datasets: tuple[str, ...]  # datasets whose stats/semantics match this benchmark
    gripper_dims: tuple[int, ...]


LIBERO = BenchmarkSpec(
    name="libero",
    state_dim=8,
    action_dim=7,
    state_layout="eef_pos(3) axis_angle(3) gripper_qpos(2)  [LeRobot LIBERO observation.state]",
    action_layout="delta eef_pos(3) delta axis_angle(3) gripper(1, -1 open / +1 close)  [OSC_POSE]",
    # LeRobot env names (image, image2) and the legacy socket bridge's names
    cameras=("image", "image2", "agentview", "eye_in_hand"),
    datasets=("lerobot/libero", "HuggingFaceVLA/libero"),
    gripper_dims=(6,),
)

ROBOTWIN = BenchmarkSpec(
    name="robotwin",
    state_dim=14,
    action_dim=14,
    state_layout="aloha bimanual joint positions: left arm(6) left gripper(1) right arm(6) right gripper(1)",
    action_layout="absolute aloha joint positions, same layout as the state",
    cameras=("head_camera", "left_camera", "right_camera"),
    # pepijn223/robotwin_unified_v3: what the reference lerobot/smolvla_robotwin was trained on
    datasets=("lerobot/robotwin_unified", "pepijn223/robotwin_unified_v3"),
    gripper_dims=(6, 13),
)

VLABENCH = BenchmarkSpec(
    name="vlabench",
    state_dim=7,
    action_dim=7,
    state_layout="eef_pos(3) eef_euler(3) gripper_opening(1)  [LeRobot VLABench conventions()]",
    action_layout="absolute eef pose, same layout as the state",
    cameras=("image", "second_image", "wrist_image"),
    datasets=(
        "lerobot/vlabench_unified",
        "VLABench/vlabench_primitive_ft_lerobot_video",
        "VLABench/vlabench_composite_ft_lerobot_video",
    ),
    gripper_dims=(6,),
)

BENCHMARKS = {spec.name: spec for spec in (LIBERO, ROBOTWIN, VLABENCH)}


class BenchmarkMismatch(ValueError):
    """The policy, its statistics or the config do not belong to the simulator's benchmark."""


def get_benchmark(name: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise BenchmarkMismatch(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}") from None


def validate_rl_setup(
    spec: BenchmarkSpec,
    *,
    policy_state_dim: int,
    policy_action_dim: int,
    policy_cameras: Sequence[str],
    camera_map: Mapping[str, str],
    action_slice: Sequence[int],
    stats_state_dim: int | None = None,
    stats_action_dim: int | None = None,
    dataset_id: str | None = None,
    env_gripper_dims: Sequence[int] | None = None,
) -> None:
    """Raise :class:`BenchmarkMismatch` unless the setup lives entirely in ``spec``'s space."""
    problems: list[str] = []
    if policy_state_dim != spec.state_dim:
        problems.append(f"policy state is {policy_state_dim}-D, {spec.name} provides {spec.state_dim}-D ({spec.state_layout})")
    if policy_action_dim != spec.action_dim:
        problems.append(f"policy acts in {policy_action_dim}-D, {spec.name} takes {spec.action_dim}-D ({spec.action_layout})")
    if tuple(action_slice) != (0, policy_action_dim):
        problems.append(
            f"action_slice={list(action_slice)} does not cover the policy's whole {policy_action_dim}-D action: "
            "slicing a policy's action vector re-interprets it in another space"
        )
    if stats_state_dim is not None and stats_state_dim != spec.state_dim:
        problems.append(f"normalisation stats describe a {stats_state_dim}-D state, not {spec.name}'s {spec.state_dim}-D")
    if stats_action_dim is not None and stats_action_dim != spec.action_dim:
        problems.append(f"normalisation stats describe a {stats_action_dim}-D action, not {spec.name}'s {spec.action_dim}-D")
    if dataset_id is not None and dataset_id not in spec.datasets:
        problems.append(f"policy/stats come from dataset {dataset_id!r}, which is not a {spec.name} dataset {spec.datasets}")
    missing = [k for k in policy_cameras if k not in camera_map]
    if missing:
        problems.append(f"camera_map does not map policy cameras {missing}")
    unknown = sorted({v for v in camera_map.values() if v not in spec.cameras})
    if unknown:
        problems.append(f"camera_map uses cameras {unknown} that {spec.name} does not have {spec.cameras}")
    used = [camera_map[k] for k in policy_cameras if k in camera_map]
    dupes = sorted({c for c in used if used.count(c) > 1})
    if dupes:
        problems.append(f"simulator cameras {dupes} feed several policy camera slots; each slot must keep its identity")
    if env_gripper_dims is not None and tuple(env_gripper_dims) != spec.gripper_dims:
        problems.append(f"env_gripper_dims={list(env_gripper_dims)} but {spec.name}'s gripper dims are {list(spec.gripper_dims)}")
    if problems:
        raise BenchmarkMismatch(f"RL setup is not valid for {spec.name}:\n  - " + "\n  - ".join(problems))


__all__ = ["BENCHMARKS", "LIBERO", "ROBOTWIN", "VLABENCH", "BenchmarkMismatch", "BenchmarkSpec", "get_benchmark", "validate_rl_setup"]
