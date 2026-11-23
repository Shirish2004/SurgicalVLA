"""RL rollouts in any LeRobot simulator (LIBERO, RoboTwin, VLABench, ...).

:class:`LeRobotSimClient` exposes the ``reset`` / ``step`` / ``list_tasks`` /
``close`` interface :func:`surgicalvla.rl.rollout.rollout_episode` drives, but
builds the environment exactly as ``lerobot-eval`` does (``make_env`` +
``preprocess_observation``), so RL rollouts see the observations the benchmark
evaluation sees. With ``endpoint`` set, the env runs in LeRobot's canonical
Docker image behind ``lerobot-env-server`` (RoboTwin and VLABench need their own
torch / simulator stacks and only run that way).

GRPO groups share one start state:
* LIBERO: the benchmark init state is set explicitly (``init_state_id``);
* RoboTwin / VLABench: the scene is a function of the reset seed, so every
  member of a group is reset with the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from surgicalvla.rl.env import Observation, StepResult, TaskInfo

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SEED_SCENES = 10_000  # "init states" for seed-defined benchmarks


def _env_config(benchmark: str, task: str, task_id: int | None, kwargs: dict, endpoint: str | None):
    from lerobot.envs import configs

    if benchmark == "libero":
        cfg = configs.LiberoEnvConfig(task=task, task_ids=None if task_id is None else [task_id], **kwargs)
    elif benchmark == "robotwin":
        cfg = configs.RoboTwinEnvConfig(task=task, **kwargs)
    elif benchmark == "vlabench":
        cfg = configs.VLABenchEnvConfig(task=task, **kwargs)
    else:
        raise ValueError(f"unsupported benchmark {benchmark!r}")
    if endpoint:
        cfg.server.endpoint = endpoint
    return cfg


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0] if value.ndim else value.item()
    return value


def _success(info: dict) -> bool:
    """``is_success`` as lerobot-eval reads it, with or without autoreset final_info."""
    final = info.get("final_info")
    if isinstance(final, dict) and "is_success" in final:
        return bool(_first(final["is_success"]))
    if isinstance(final, (list, tuple, np.ndarray)):
        for item in final:
            if isinstance(item, dict) and "is_success" in item:
                return bool(item["is_success"])
    if "is_success" in info:
        return bool(_first(info["is_success"]))
    return False


@dataclass
class LeRobotSimClient:
    benchmark: str
    env_kwargs: dict | None = None
    endpoint: str | None = None
    max_cached_envs: int = 4

    def __post_init__(self):
        self._envs: dict[str, object] = {}
        self._current = None
        self._language = ""

    # ------------------------------------------------------------------ tasks
    def list_tasks(self, suite: str) -> list[TaskInfo]:
        if self.benchmark == "libero":
            from libero.libero import benchmark as libero_benchmark

            bench = libero_benchmark.get_benchmark_dict()[suite]()
            return [
                TaskInfo(index=i, name=f"{suite}:{i}", language=t.language, n_init_states=len(bench.get_task_init_states(i)))
                for i, t in enumerate(bench.tasks)
            ]
        names = [t.strip() for t in suite.split(",") if t.strip()]
        return [TaskInfo(index=i, name=n, language=n.replace("_", " "), n_init_states=SEED_SCENES) for i, n in enumerate(names)]

    # ------------------------------------------------------------------ env
    def _env(self, task: str):
        if task in self._envs:
            return self._envs[task]
        from lerobot.envs.factory import make_env

        if len(self._envs) >= self.max_cached_envs:
            _, old = self._envs.popitem()
            old.close()
        if self.benchmark == "libero":
            suite, task_id = task.split(":")
            cfg = _env_config("libero", suite, int(task_id), dict(self.env_kwargs or {}), self.endpoint)
        else:
            cfg = _env_config(self.benchmark, task, None, dict(self.env_kwargs or {}), self.endpoint)
        envs = make_env(cfg, n_envs=1, use_async_envs=False)
        env = next(iter(next(iter(envs.values())).values()))
        self._envs[task] = env
        return env

    def _observation(self, raw: dict, success: bool = False) -> Observation:
        from lerobot.envs.utils import preprocess_observation

        obs = preprocess_observation(raw)
        images = {}
        for key, value in obs.items():
            if key.startswith("observation.images."):
                img = value[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
                images[key.removeprefix("observation.images.")] = img
        state = obs.get("observation.state")
        state = np.zeros(0, np.float32) if state is None else state[0].float().cpu().numpy()
        # state carries the benchmark's own layout (LIBERO 8-D, RoboTwin 14-D, VLABench 7-D)
        return Observation(images=images, state=state, language=self._language, success=success, agent_pos=None)

    def reset(self, suite: str, task: str, episode: int = 0, seed: int = 0) -> Observation:
        env = self._env(task)
        if self.benchmark == "libero":
            env.set_attr("init_state_id", int(episode))
            reset_seed = int(seed)
        else:
            reset_seed = int(episode)  # the scene is the group's shared start state
        raw, _ = env.reset(seed=[reset_seed])
        try:
            self._language = str(_first(env.call("task_description")))
        except Exception:  # noqa: BLE001 - not every env exposes it
            self._language = task.replace("_", " ")
        self._current = env
        return self._observation(raw)

    def step(self, action) -> StepResult:
        if self._current is None:
            raise RuntimeError("step called before reset")
        act = np.asarray(action, dtype=np.float32).reshape(1, -1)
        raw, reward, terminated, truncated, info = self._current.step(act)
        success = _success(info)
        done = bool(_first(terminated)) or bool(_first(truncated))
        return StepResult(self._observation(raw, success), float(_first(reward)), done, success)

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()
        self._current = None


__all__ = ["LIBERO_SUITES", "LeRobotSimClient"]
