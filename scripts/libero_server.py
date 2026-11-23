#!/usr/bin/env python
"""LIBERO simulator server, one process, one socket, one environment at a time.

This script runs inside the ``libero`` pixi environment (Python 3.10, numpy 1.22,
robosuite 1.4) and is spoken to by ``src/surgicalvla/rl/env.py`` from the
``default`` environment (Python 3.13, torch, lerobot). The two environments
cannot share an interpreter, so the simulator is a separate process and the
protocol is :mod:`surgicalvla.rl.protocol` -- which this script loads by path,
because the package is not installed here (torch deliberately is not either).

    pixi run -e libero python scripts/libero_server.py --port 0

``--port 0`` binds an ephemeral port; the resolved port is printed on stdout as
``@@READY@@ {"port": 12345}`` once the server is accepting connections.
Everything else is written to stderr, because stdout belongs to that line
until the client has read it.

Commands (JSON per line, request -> response):

``{"cmd": "list_tasks", "suite": "libero_10"}``
    -> ``{"ok": true, "suite": ..., "tasks": [{"name", "language", "index",
    "n_init_states"}]}``

``{"cmd": "reset", "suite": ..., "task": ..., "episode": 0, "seed": 0}``
    -> ``{"ok": true, "obs": {...}}`` -- the episode is placed at
    ``init_states[episode % len(init_states)]``

``{"cmd": "step", "action": [7 floats]}``
    -> ``{"ok": true, "obs": {...}, "reward": float, "done": bool,
    "success": bool}``

``{"cmd": "close"}`` -> closes the episode; the server then waits for a new
connection.

An observation is ``{"images": {camera: uint8 HxWx3}, "state": float32 (7,),
"agent_pos": float32 (8,), "language": str, "success": bool}``:

* ``agent_pos`` is the state LeRobot's LIBERO integration builds and the
  ``lerobot/libero`` / ``HuggingFaceVLA/libero`` datasets store as
  ``observation.state`` (what ``smolvla_libero`` was trained on): end-effector
  position (3), axis-angle rotation (3, robosuite ``quat2axisangle``), gripper
  joint positions (2).
* ``state`` is the legacy 7-D layout of an ``OSC_POSE`` action: end-effector
  position, axis-angle rotation, gripper openness in [-1, 1] (+1 fully open).

By default the server follows LeRobot's LIBERO conventions (``--flip-images``,
``--settle-steps 10``): images are rotated 180 degrees as in the LeRobot
datasets, and after placing the init state the simulator takes 10 no-op steps
so objects settle before the first observation.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import pickle
import re
import socket
import sys
import traceback
import zipfile
from collections import OrderedDict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL_PATH = os.path.join(REPO_ROOT, "src", "surgicalvla", "rl", "protocol.py")

# Full-open finger separation of a Panda gripper, in metres. Only used to map
# the gripper joint state into the same [-1, 1] interval the actions use.
GRIPPER_MAX_WIDTH = 0.04

LANGUAGE_RE = re.compile(r"\(:language\s+(.+?)\)\s*$", re.MULTILINE)


def load_protocol():
    """Import ``protocol.py`` by path (the package is not installed in this env)."""
    spec = importlib.util.spec_from_file_location("surgicalvla_rl_protocol", PROTOCOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- tasks
def task_language(bddl_path: str) -> str:
    with open(bddl_path) as f:
        text = f.read()
    m = LANGUAGE_RE.search(text)
    if m:
        return " ".join(m.group(1).split())
    name = os.path.splitext(os.path.basename(bddl_path))[0]
    return " ".join(name.split("_"))


def load_init_states(path: str) -> np.ndarray:
    """Read a LIBERO ``.pruned_init`` file without torch.

    Those files are ``torch.save`` output wrapping a plain numpy array, so the
    pickle inside the zip archive is all we need. Falls back to ``np.load`` for
    a plain ``.npy``.
    """
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("data.pkl"))
            states = pickle.loads(z.read(name))
    except (zipfile.BadZipFile, KeyError, pickle.UnpicklingError):
        states = np.load(path, allow_pickle=True)
    states = np.asarray(states)
    if states.ndim != 2:
        raise ValueError(f"{path}: expected (n_init_states, sim_state_dim), got {states.shape}")
    return states


def discover_tasks(suite: str) -> list[dict]:
    from libero.libero import get_libero_path

    bddl_dir = os.path.join(get_libero_path("bddl_files"), suite)
    init_dir = os.path.join(get_libero_path("init_states"), suite)
    if not os.path.isdir(bddl_dir):
        raise FileNotFoundError(f"unknown LIBERO suite {suite!r}: {bddl_dir} does not exist")

    tasks = []
    for bddl in sorted(glob.glob(os.path.join(bddl_dir, "*.bddl"))):
        name = os.path.splitext(os.path.basename(bddl))[0]
        init_file = None
        for ext in (".pruned_init", ".init"):
            candidate = os.path.join(init_dir, name + ext)
            if os.path.exists(candidate):
                init_file = candidate
                break
        tasks.append(
            {
                "index": len(tasks),
                "name": name,
                "language": task_language(bddl),
                "bddl": bddl,
                "init_file": init_file,
                "n_init_states": 0 if init_file is None else len(load_init_states(init_file)),
            }
        )
    if not tasks:
        raise FileNotFoundError(f"no .bddl files under {bddl_dir}")
    return tasks


# ----------------------------------------------------------------------- simulator
def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion -> rotation vector, matching robosuite's convention."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w = float(np.clip(q[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-8:
        return np.zeros(3)
    return (q[:3] / s) * angle


def lerobot_agent_pos(obs: dict) -> np.ndarray:
    """8-D ``observation.state`` exactly as LeRobot's LIBERO integration builds it.

    ``[eef_pos (3), axis_angle(eef_quat) (3), gripper_qpos (2)]`` (see
    ``lerobot/sim/integrations/libero.py::canonical_observation``). This is what
    ``lerobot/smolvla_libero`` was trained on; the 7-D :func:`proprio_state`
    replaces the two finger joints with one openness scalar and is a different
    input distribution.
    """
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat_to_axis_angle(obs["robot0_eef_quat"]).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)


# LeRobot's ``get_libero_dummy_action``: zero motion, gripper open.
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def proprio_state(obs: dict) -> np.ndarray:
    qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    openness = float(np.clip((qpos[0] - qpos[1]) / GRIPPER_MAX_WIDTH, 0.0, 1.0)) * 2.0 - 1.0
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat_to_axis_angle(obs["robot0_eef_quat"]).astype(np.float32),
            np.array([openness], dtype=np.float32),
        ]
    ).astype(np.float32)


class Simulator:
    def __init__(
        self,
        camera_height: int = 128,
        camera_width: int = 128,
        max_cached_envs: int = 8,
        flip_images: bool = True,
        settle_steps: int = 10,
    ):
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.max_cached_envs = max_cached_envs
        self.flip_images = bool(flip_images)
        self.settle_steps = int(settle_steps)
        self.envs: OrderedDict[str, object] = OrderedDict()
        self.tasks: dict[str, dict] = {}
        self.init_states: dict[str, np.ndarray] = {}
        self.current = None
        self.current_task = None

    # -- env lifecycle -----------------------------------------------------
    def env(self, task: dict):
        from libero.libero.envs import OffScreenRenderEnv

        name = task["name"]
        if name in self.envs:
            self.envs.move_to_end(name)
            return self.envs[name]

        if len(self.envs) >= self.max_cached_envs:
            old_name, old_env = self.envs.popitem(last=False)
            try:
                old_env.close()
            except Exception:  # noqa: BLE001 - teardown must not kill the server
                log(f"warning: closing cached env {old_name} failed")
        env = OffScreenRenderEnv(
            bddl_file_name=task["bddl"],
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )
        self.envs[name] = env
        return env

    def reset(self, suite: str, name: str, episode: int, seed: int) -> dict:
        task = self.tasks.get(name)
        if task is None:
            for t in discover_tasks(suite):
                self.tasks[t["name"]] = t
            task = self.tasks.get(name)
        if task is None:
            raise KeyError(f"task {name!r} is not in suite {suite!r}")

        if task["init_file"] is None:
            raise FileNotFoundError(f"task {name!r} has no init-state file")
        if name not in self.init_states:
            self.init_states[name] = load_init_states(task["init_file"])
        states = self.init_states[name]

        env = self.env(task)
        env.seed(int(seed))
        env.reset()
        idx = int(episode) % len(states)
        obs = env.set_init_state(states[idx])
        for _ in range(self.settle_steps):
            obs, _, _, _ = env.step(np.asarray(DUMMY_ACTION, dtype=np.float32))
        self.current = env
        self.current_task = task
        return self.observation(obs, task)

    def step(self, action):
        if self.current is None:
            raise RuntimeError("step called before reset")
        env = self.current
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"expected a 7-D OSC_POSE action, got shape {action.shape}")
        obs, reward, done, _info = env.step(np.clip(action, -1.0, 1.0))
        success = bool(env.check_success())
        horizon_done = bool(getattr(env.env, "timestep", 0) >= getattr(env.env, "horizon", 1))
        return self.observation(obs, self.current_task), float(reward), bool(done) or horizon_done, success

    def _image(self, image) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        # LeRobot rotates LIBERO renders by 180 degrees (flip H and W).
        return np.ascontiguousarray(image[::-1, ::-1]) if self.flip_images else image

    def observation(self, obs: dict, task: dict) -> dict:
        return {
            "images": {
                "agentview": self._image(obs["agentview_image"]),
                "eye_in_hand": self._image(obs["robot0_eye_in_hand_image"]),
            },
            "state": proprio_state(obs),
            "agent_pos": lerobot_agent_pos(obs),
            "language": task["language"],
            "success": bool(self.current.check_success()) if self.current is not None else False,
        }

    def close(self):
        for env in self.envs.values():
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        self.envs.clear()
        self.current = None


# -------------------------------------------------------------------------- server
def serve(protocol, args):
    sim = Simulator(
        args.camera_height,
        args.camera_width,
        args.max_cached_envs,
        flip_images=args.flip_images,
        settle_steps=args.settle_steps,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen(1)
    port = listener.getsockname()[1]
    print(f"{protocol.READY_PREFIX} {protocol.dumps({'port': port}).decode().strip()}", flush=True)
    log(f"[libero_server] listening on {args.host}:{port}")

    try:
        while True:
            sock, peer = listener.accept()
            log(f"[libero_server] client {peer} connected")
            channel = protocol.Channel(sock)
            try:
                while True:
                    request = channel.recv()
                    if request is None:
                        log("[libero_server] client disconnected")
                        break
                    try:
                        response = handle(protocol, sim, request)
                    except Exception as exc:  # noqa: BLE001 - report, keep serving
                        response = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                        log(f"[libero_server] {response['error']}")
                    channel.send(response)
                    if request.get("cmd") == "close" and response.get("ok"):
                        log("[libero_server] closed by request")
                        break
            finally:
                channel.close()
    except KeyboardInterrupt:
        log("[libero_server] interrupted")
    finally:
        listener.close()
        sim.close()


def handle(protocol, sim: Simulator, request: dict) -> dict:
    cmd = request.get("cmd")
    if cmd == "list_tasks":
        suite = request["suite"]
        tasks = discover_tasks(suite)
        sim.tasks.update({t["name"]: t for t in tasks})
        return {
            "ok": True,
            "suite": suite,
            "tasks": [
                {k: t[k] for k in ("index", "name", "language", "n_init_states")} for t in tasks
            ],
        }
    if cmd == "reset":
        obs = sim.reset(
            request.get("suite", ""),
            request["task"],
            int(request.get("episode", 0)),
            int(request.get("seed", 0)),
        )
        return {"ok": True, "obs": obs}
    if cmd == "step":
        obs, reward, done, success = sim.step(request["action"])
        return {"ok": True, "obs": obs, "reward": reward, "done": done, "success": success}
    if cmd == "close":
        sim.close()
        return {"ok": True}
    raise ValueError(f"unknown command {cmd!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 binds an ephemeral port")
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--max-cached-envs", type=int, default=8)
    parser.add_argument(
        "--flip-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rotate renders 180 degrees like LeRobot's LIBERO datasets (default: on)",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=10,
        help="no-op steps after placing the init state, as LeRobot's num_steps_wait (default: 10)",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    serve(load_protocol(), args)


if __name__ == "__main__":
    main()
