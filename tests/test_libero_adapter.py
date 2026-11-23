"""The LIBERO eval adapter must feed policies the state they were trained on.

``lerobot/smolvla_libero`` was trained on ``lerobot/libero`` whose
``observation.state`` is LeRobot's 8-D LIBERO state: end-effector position (3),
axis-angle (3), the two finger joints (2). A 7-D state (gripper *openness* in
[-1, 1]) tiled to 8-D, i.e. ``[pos, axis-angle, openness, pos_x]``, puts dim 6
about 70 sigma outside the training distribution for an open gripper.

* ``test_policy_state_*``: selection logic, no simulator needed.
* ``test_adapter_matches_lerobot_libero_env``: our server's observation equals
  LeRobot's own ``LiberoEnv`` observation for the same task and init state.
* ``test_adapter_state_in_smolvla_libero_training_distribution``: real adapter
  states z-scored with the checkpoint's normaliser stats stay in range; the
  legacy tiled state does not.

The simulator tests need ``hf-libero`` (``pixi run -e sim-eval pytest ...``).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from surgicalvla.rl.rollout import lift, policy_state

ROOT = Path(__file__).resolve().parents[1]
SMOLVLA_LIBERO = ("lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d")


def _server():
    spec = importlib.util.spec_from_file_location("libero_server", ROOT / "scripts" / "libero_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw_obs(rng):
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    return {
        "robot0_eef_pos": rng.normal(size=3),
        "robot0_eef_quat": quat,
        "robot0_gripper_qpos": np.array([0.03, -0.029]),
    }


# --------------------------------------------------------------------- no simulator
def test_policy_state_selects_8d_agent_pos():
    obs = SimpleNamespace(state=np.arange(7, dtype=np.float32), agent_pos=np.arange(8, dtype=np.float32) + 10)
    np.testing.assert_array_equal(policy_state(obs, 8), obs.agent_pos)
    np.testing.assert_array_equal(policy_state(obs, 7), obs.state)


def test_policy_state_refuses_silent_tiling():
    obs = SimpleNamespace(state=np.arange(7, dtype=np.float32), agent_pos=None)
    with pytest.raises(ValueError, match="Refusing to tile"):
        policy_state(obs, 8)
    with pytest.raises(ValueError, match="Refusing to tile"):
        policy_state(obs, 14)
    np.testing.assert_array_equal(policy_state(obs, 14, lift_state=True), lift(obs.state, 14))


def test_agent_pos_layout():
    server = _server()
    raw = _raw_obs(np.random.default_rng(0))
    pos = server.lerobot_agent_pos(raw)
    assert pos.shape == (8,) and pos.dtype == np.float32
    np.testing.assert_allclose(pos[:3], raw["robot0_eef_pos"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(pos[3:6], server.quat_to_axis_angle(raw["robot0_eef_quat"]), atol=1e-6)
    np.testing.assert_allclose(pos[6:], raw["robot0_gripper_qpos"], atol=1e-7)


# ------------------------------------------------------------------- with simulator
libero_ok = importlib.util.find_spec("libero") is not None
needs_libero = pytest.mark.skipif(not libero_ok, reason="needs hf-libero (pixi -e sim-eval)")


def _lerobot_quat_to_axisangle():
    from lerobot.sim.integrations.libero import _quat_to_axisangle

    return _quat_to_axisangle


@needs_libero
def test_axis_angle_matches_lerobot():
    ref = _lerobot_quat_to_axisangle()
    server = _server()
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q *= np.sign(q[3]) or 1.0  # robosuite keeps w >= 0
        np.testing.assert_allclose(server.quat_to_axis_angle(q), ref(q.astype(np.float32)), atol=2e-5)


def _first_task(suite: str):
    server = _server()
    tasks = server.discover_tasks(suite)
    return server, tasks[0]


@needs_libero
@pytest.mark.parametrize("suite", ["libero_spatial", "libero_10"])
def test_adapter_matches_lerobot_libero_env(suite):
    """Same task, same init state, both settled: identical 8-D state and images."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.sim.integrations.libero import LiberoEnv, _get_suite

    server, task = _first_task(suite)
    size = 128
    sim = server.Simulator(size, size, max_cached_envs=1)
    try:
        ours = sim.reset(suite, task["name"], episode=0, seed=0)
    finally:
        sim.close()

    bench = _get_suite(suite)
    task_id = next(i for i, t in enumerate(bench.tasks) if t.name == task["name"])
    env = LiberoEnv(
        task_suite=bench,
        task_id=task_id,
        task_suite_name=suite,
        obs_type="pixels_agent_pos",
        observation_height=size,
        observation_width=size,
        init_states=True,
    )
    try:
        theirs, _ = env.reset(seed=0)
    finally:
        env.close()

    from lerobot.sim.integrations.libero import canonical_observation

    canon = canonical_observation(theirs)
    state_key = next(k for k in canon if k.endswith("state") or k == "agent_pos")
    np.testing.assert_allclose(ours["agent_pos"], np.asarray(canon[state_key]), atol=1e-5)
    images = {k: v for k, v in canon.items() if np.asarray(v).ndim == 3}
    assert images, f"no images in canonical observation keys {list(canon)}"
    agent = next(v for k, v in images.items() if "image2" not in k and "wrist" not in k)
    np.testing.assert_array_equal(ours["images"]["agentview"], np.asarray(agent))


def _checkpoint_state_stats():
    from huggingface_hub import try_to_load_from_cache
    from safetensors.torch import load_file

    path = try_to_load_from_cache(
        SMOLVLA_LIBERO[0], "policy_preprocessor_step_5_normalizer_processor.safetensors", revision=SMOLVLA_LIBERO[1]
    )
    if not isinstance(path, str):
        pytest.skip("lerobot/smolvla_libero normaliser stats not cached")
    stats = load_file(path)
    return stats["observation.state.mean"].numpy(), stats["observation.state.std"].numpy()


@needs_libero
@pytest.mark.parametrize("suite", ["libero_spatial", "libero_object", "libero_goal", "libero_10"])
def test_adapter_state_in_smolvla_libero_training_distribution(suite):
    os.environ.setdefault("MUJOCO_GL", "egl")
    mean, std = _checkpoint_state_stats()
    server, task = _first_task(suite)
    sim = server.Simulator(64, 64, max_cached_envs=1)
    states, legacy = [], []
    try:
        for episode in range(3):
            obs = sim.reset(suite, task["name"], episode=episode, seed=episode)
            states.append(obs["agent_pos"])
            legacy.append(lift(obs["state"], 8))
            for _ in range(5):
                obs = sim.step([0.0] * 6 + [-1.0])[0]
                states.append(obs["agent_pos"])
    finally:
        sim.close()

    z = np.abs((np.stack(states) - mean) / std)
    assert z.max() < 5.0, f"adapter state outside training distribution: max |z| per dim {z.max(0).round(2)}"
    z_legacy = np.abs((np.stack(legacy) - mean) / std)
    assert z_legacy[:, 6].min() > 20.0, "legacy 7-D tiled state should be far out of distribution in dim 6"
