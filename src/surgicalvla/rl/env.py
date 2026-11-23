"""Client for the LIBERO simulator server (see ``scripts/libero_server.py``).

The simulator cannot run in this process: LIBERO pins ``numpy`` 1.22 /
``robosuite`` 1.4 / Python 3.10 while this package needs ``numpy`` 2.x and
``lerobot`` (Python >= 3.12). So the simulator lives in the ``libero`` pixi
environment as a child process and the two sides talk
:mod:`surgicalvla.rl.protocol` over a loopback socket.

:class:`LiberoClient` owns that child process:

* :meth:`~LiberoClient.spawn` starts ``pixi run -e libero python
  scripts/libero_server.py --port 0``, waits for the ``@@READY@@`` line on
  stdout and connects.
* :meth:`~LiberoClient.connect` attaches to a server someone else started
  (useful when debugging the server by hand).
* every request is one JSON line in, one JSON line out; a server-side
  exception comes back as ``{"ok": false, "error": ...}`` and is re-raised as
  :class:`LiberoError` so a silent simulator bug cannot be mistaken for a
  policy failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from surgicalvla.rl.protocol import READY_PREFIX, Channel, connect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_SCRIPT = REPO_ROOT / "scripts" / "libero_server.py"
MANIFEST = REPO_ROOT / "pixi.toml"

#: Variables the parent pixi environment leaves behind that would make the child
#: ``pixi run -e libero`` activate the wrong interpreter (or import the wrong
#: packages from a user site).
_STRIP_ENV = ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE", "VIRTUAL_ENV")


class LiberoError(RuntimeError):
    """A request reached the simulator but it reported a failure."""

    def __init__(self, message: str, traceback: str | None = None):
        super().__init__(message)
        self.traceback = traceback


@dataclass
class Observation:
    """One simulator observation, in the simulator's native 7-D OSC_POSE layout."""

    images: dict[str, np.ndarray]  # camera name -> uint8 (H, W, 3)
    state: np.ndarray  # float32 (7,): eef xyz, axis-angle, gripper openness in [-1, 1]
    language: str
    success: bool = False
    object_pose_6d: np.ndarray | None = None  # optional [xyz, r1..r6] handover target pose
    # float32 (8,): LeRobot's LIBERO observation.state (eef xyz, axis-angle, 2 finger joints)
    agent_pos: np.ndarray | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> Observation:
        pose = payload.get("object_pose_6d")
        return cls(
            images={k: np.asarray(v) for k, v in payload["images"].items()},
            state=np.asarray(payload["state"], dtype=np.float32),
            language=payload["language"],
            success=bool(payload.get("success", False)),
            object_pose_6d=None if pose is None else np.asarray(pose, dtype=np.float32),
            agent_pos=None if payload.get("agent_pos") is None else np.asarray(payload["agent_pos"], dtype=np.float32),
        )


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    success: bool


@dataclass
class TaskInfo:
    index: int
    name: str
    language: str
    n_init_states: int


@dataclass
class LiberoClient:
    """A connected simulator server."""

    channel: Channel
    process: subprocess.Popen | None = None
    port: int = 0
    cameras: tuple[str, ...] = ()
    _log_lines: list[str] = field(default_factory=list)
    _thread: threading.Thread | None = None

    # ------------------------------------------------------------------ spawning
    @classmethod
    def spawn(
        cls,
        env: str = "libero",
        host: str = "127.0.0.1",
        camera_height: int = 128,
        camera_width: int = 128,
        max_cached_envs: int = 8,
        timeout: float = 240.0,
        extra_args: list[str] | None = None,
        verbose: bool = True,
    ) -> LiberoClient:
        """Start the simulator in its own pixi environment and connect to it."""
        pixi = shutil.which("pixi") or str(Path.home() / ".pixi" / "bin" / "pixi")
        if not Path(pixi).exists():
            raise FileNotFoundError(
                "pixi not found on PATH; install it or run the server manually and use LiberoClient.connect()"
            )
        cmd = [
            pixi,
            "run",
            "--manifest-path",
            str(MANIFEST),
            "-e",
            env,
            "python",
            str(SERVER_SCRIPT),
            "--host",
            host,
            "--port",
            "0",
            "--camera-height",
            str(camera_height),
            "--camera-width",
            str(camera_width),
            "--max-cached-envs",
            str(max_cached_envs),
            *(extra_args or []),
        ]
        environ = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
        environ["PYTHONUNBUFFERED"] = "1"
        environ.setdefault("MUJOCO_GL", "egl")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=environ,
            stdout=subprocess.PIPE,
            stderr=None if verbose else subprocess.DEVNULL,
        )

        ready = threading.Event()
        state: dict[str, object] = {}

        def pump():
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line.startswith(READY_PREFIX):
                    payload = line[len(READY_PREFIX) :].strip()
                    try:
                        state["ready"] = json.loads(payload) if payload else {}
                    except json.JSONDecodeError:
                        state["ready"] = {}
                    ready.set()
                elif line:
                    # stdout belongs to the handshake; echo anything else so a
                    # python traceback or a stray print is not swallowed.
                    print(f"[libero_server] {line}", file=sys.stderr, flush=True)

        thread = threading.Thread(target=pump, name="libero-server-stdout", daemon=True)
        thread.start()

        if not ready.wait(timeout):
            proc.terminate()
            raise RuntimeError(
                f"the LIBERO server did not report {READY_PREFIX} within {timeout:.0f}s; "
                f"check the server output above (is the `{env}` pixi environment installed?)"
            )
        if proc.poll() is not None:
            raise RuntimeError(f"the LIBERO server exited with code {proc.returncode} during startup")

        payload = state.get("ready") or {}
        port = int(payload.get("port", 0))
        client = cls(channel=connect(host, port, timeout=timeout), process=proc, port=port)
        client._thread = thread
        return client

    @classmethod
    def connect(cls, host: str = "127.0.0.1", port: int = 8000, timeout: float | None = 240.0) -> LiberoClient:
        """Attach to an already running server (``scripts/libero_server.py --port N``)."""
        return cls(channel=connect(host, port, timeout=timeout), port=port)

    # ------------------------------------------------------------------ requests
    def _request(self, message: dict) -> dict:
        self.channel.send(message)
        response = self.channel.recv()
        if response is None:
            raise ConnectionError("the LIBERO server closed the connection")
        if not response.get("ok", False):
            raise LiberoError(response.get("error", "unknown simulator error"), response.get("traceback"))
        return response

    def list_tasks(self, suite: str) -> list[TaskInfo]:
        payload = self._request({"cmd": "list_tasks", "suite": suite})
        return [TaskInfo(**t) for t in payload["tasks"]]

    def reset(self, suite: str, task: str, episode: int = 0, seed: int = 0) -> Observation:
        payload = self._request(
            {"cmd": "reset", "suite": suite, "task": task, "episode": int(episode), "seed": int(seed)}
        )
        obs = Observation.from_dict(payload["obs"])
        if not self.cameras:
            self.cameras = tuple(sorted(obs.images))
        return obs

    def step(self, action) -> StepResult:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        payload = self._request({"cmd": "step", "action": action})
        return StepResult(
            observation=Observation.from_dict(payload["obs"]),
            reward=float(payload["reward"]),
            done=bool(payload["done"]),
            success=bool(payload["success"]),
        )

    # -------------------------------------------------------------------- teardown
    def close(self) -> None:
        try:
            if self.channel is not None:
                try:
                    self.channel.send({"cmd": "close"})
                    self.channel.recv()
                except (OSError, ConnectionError):
                    pass
                self.channel.close()
        finally:
            self.channel = None
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass
