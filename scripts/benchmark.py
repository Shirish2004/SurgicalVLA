"""One launcher for training and evaluation across benchmarks.

    # supervised fine-tuning
    python3 scripts/benchmark.py train sft --benchmark libero   --policy smolvla
    python3 scripts/benchmark.py train sft --benchmark robotwin --policy smolvla
    python3 scripts/benchmark.py train sft --benchmark libero   --policy surgicalvla
    python3 scripts/benchmark.py train sft --benchmark robotwin --policy surgicalvla
    # online RL (ReinFlow-GRPO); LIBERO only for now
    python3 scripts/benchmark.py train rl  --benchmark libero
    # evaluation
    python3 scripts/benchmark.py eval --benchmark libero      --checkpoint lerobot/smolvla_libero --revision a24f1f38...
    python3 scripts/benchmark.py eval --benchmark libero_plus --checkpoint lerobot/smolvla_libero --revision a24f1f38...
    python3 scripts/benchmark.py eval --benchmark robotwin    --checkpoint outputs/baselines/smolvla_robotwin_clean50/checkpoints/last/pretrained_model

Add ``--dry-run`` to print the commands without running them. Everything after
``--`` is passed through to the underlying trainer / ``lerobot-eval``.

Each job runs in the pixi environment it belongs to (training: ``dev``,
simulators: ``sim-eval``). LIBERO-Plus and RoboTwin run inside LeRobot's
canonical Docker images behind ``lerobot-env-server`` (started and stopped here);
LIBERO runs locally unless ``--server`` is given.

Evaluation protocols (docs/baselines.md):
* libero       4 suites x 10 tasks x 50 init states x seeds (default 3)
* libero_plus  every registered variant of the 4 perturbed suites, 1 episode each (upstream protocol)
* robotwin     100 episodes per task, demo_clean (Easy). LeRobot's RoboTwin integration hard-codes
               task_config/demo_clean.yml, so the Hard (demo_randomized) setting needs an
               integration change and is not offered here yet.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEROBOT = ROOT / "lerobot"
ENDPOINT = "tcp/127.0.0.1:7448"

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LIBERO_RENAME = {"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}
ROBOTWIN_RENAME = {
    "observation.images.head_camera": "observation.images.camera1",
    "observation.images.left_camera": "observation.images.camera2",
    "observation.images.right_camera": "observation.images.camera3",
}
ROBOTWIN_TASKS = sorted(p.stem for p in (ROOT.parent / "RoboTwin" / "description" / "task_instruction").glob("*.json"))
SMOLVLA_BASE = ("lerobot/smolvla_base", "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
SEEDS = [1000, 2000, 3000]


def pixi(env: str, *cmd: str) -> list[str]:
    return ["pixi", "run", "--manifest-path", str(ROOT / "pixi.toml"), "-e", env, *cmd]


def execute(cmds: list[list[str]], dry_run: bool) -> None:
    for cmd in cmds:
        print("$", shlex.join(cmd), flush=True)
        if not dry_run:
            code = subprocess.call(cmd, cwd=ROOT)
            if code:
                sys.exit(code)


# ----------------------------------------------------------------------------- train
def train_cmds(args, extra: list[str]) -> list[list[str]]:
    out = args.out
    if args.stage == "rl":
        if args.benchmark != "libero":
            sys.exit("RL is implemented for LIBERO only; RoboTwin RL needs a RoboTwin env client and stays 14-D.")
        return [pixi("dev", "python", "scripts/train_reinflow_grpo.py", "--config", args.config or "configs/reinflow_grpo.yaml", *extra)]

    if args.policy == "smolvla":  # unmodified SmolVLA baseline through lerobot-train
        if args.benchmark == "robotwin":
            env = ["env", f"OUT={out or 'outputs/baselines/smolvla_robotwin_clean50'}"]
            return [[*env, *pixi("dev", "bash", "scripts/train_robotwin_baseline.sh", *extra)]]
        # exact recipe of lerobot/smolvla_libero (its train_config.json): full fine-tune
        return [
            pixi(
                "dev",
                "lerobot-train",
                f"--policy.path={SMOLVLA_BASE[0]}",
                f"--policy.pretrained_revision={SMOLVLA_BASE[1]}",
                "--policy.push_to_hub=false",
                "--policy.device=cuda",
                "--policy.freeze_vision_encoder=false",
                "--policy.train_expert_only=false",
                "--policy.scheduler_decay_steps=25000",
                "--dataset.repo_id=lerobot/libero",
                f"--rename_map={json.dumps(LIBERO_RENAME)}",
                "--batch_size=32",
                "--steps=25000",
                "--save_freq=5000",
                "--seed=1000",
                "--num_workers=4",
                f"--output_dir={out or 'outputs/baselines/smolvla_libero_repro'}",
                "--job_name=smolvla_libero_repro",
                "--trackio.enable=true",
                "--trackio.project=surgical-vla",
                *extra,
            )
        ]

    # SurgicalVLA through the repo's trainer
    if args.benchmark == "libero":
        cmd = ["python", "scripts/train.py", "--config", args.config or "configs/train_libero.yaml", "--repo_id", "lerobot/libero"]
    else:
        cmd = ["python", "scripts/train.py", "--config", args.config or "configs/train.yaml", "--repo_id", "lerobot/robotwin_unified"]
    if out:
        cmd.append(f"run.ckpt_dir={out}")
    return [pixi("dev", *cmd, *extra)]


# ----------------------------------------------------------------------------- eval
def server_config(benchmark: str, task: str, out_dir: Path) -> Path:
    """LeRobot's canonical server config for ``benchmark`` with ``task`` and every registered variant."""
    import yaml

    cfg = yaml.safe_load((LEROBOT / "docker" / "sim" / "configs" / f"{benchmark}.yaml").read_text())
    cfg["sim"]["task"] = task
    cfg["sim"]["kwargs"].get("gym_kwargs", {}).pop("task_ids", None)  # all registered tasks / variants
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"server_{benchmark}_{task.replace(',', '_')[:60]}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def server(path: Path, stop: bool = False) -> list[str]:
    return pixi("sim-eval", "lerobot-env-server", f"--config_path={path}", "--docker=true", *(["--stop=true"] if stop else []))


def eval_cmds(args, extra: list[str]) -> list[list[str]]:
    name = args.name or Path(args.checkpoint.rstrip("/")).name.replace("/", "_")
    out = Path(args.out or f"results/{args.benchmark}/{name}")
    seeds = [str(s) for s in (args.seeds or SEEDS)]
    common = ["--policy", args.checkpoint, "--revision", args.revision or "", "--out", str(out), "--seeds", *seeds]
    cmds: list[list[str]] = []

    if args.benchmark in ("libero", "libero_plus"):
        suites = args.suites or LIBERO_SUITES
        plus = args.benchmark == "libero_plus"
        use_server = plus or args.server
        for suite in suites:
            run = ["python", "scripts/eval_libero_baseline.py", "run", *common, "--suites", suite, "--env-type", args.benchmark]
            if plus:  # upstream protocol: one episode per registered variant
                run += ["--n-episodes", "1", "--batch-size", "1"]
            if use_server:
                cfg = server_config(args.benchmark, suite, out)
                run += ["--server-endpoint", ENDPOINT]
                cmds += [server(cfg), pixi("sim-eval", *run, *extra), server(cfg, stop=True)]
            else:
                cmds.append(pixi("sim-eval", *run, *extra))
        cmds.append(pixi("sim-eval", "python", "scripts/eval_libero_baseline.py", "aggregate", "--root", str(out)))
        return cmds

    # RoboTwin: 100 episodes per task through the canonical RoboTwin image
    tasks = args.tasks or ROBOTWIN_TASKS
    cfg = server_config("robotwin", ",".join(tasks), out)
    cmds.append(server(cfg))
    for seed in seeds:
        cmds.append(
            pixi(
                "sim-eval",
                "lerobot-eval",
                f"--policy.path={args.checkpoint}",
                *([f"--policy.pretrained_revision={args.revision}"] if args.revision else []),
                "--policy.device=cuda",
                "--env.type=robotwin",
                f"--env.task={','.join(tasks)}",
                "--eval.n_episodes=100",
                "--eval.batch_size=1",
                f"--seed={seed}",
                f"--rename_map={json.dumps(ROBOTWIN_RENAME)}",
                f"--env.server.endpoint={ENDPOINT}",
                f"--output_dir={out / 'easy' / f'seed{seed}'}",
                *extra,
            )
        )
    cmds.append(server(cfg, stop=True))
    return cmds


def main() -> None:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("stage", choices=["sft", "rl"])
    t.add_argument("--benchmark", required=True, choices=["libero", "robotwin"])
    t.add_argument("--policy", default="surgicalvla", choices=["smolvla", "surgicalvla"])
    t.add_argument("--config", help="override the default config for this benchmark/stage")
    t.add_argument("--out", help="output / checkpoint directory")
    t.add_argument("--dry-run", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--benchmark", required=True, choices=["libero", "libero_plus", "robotwin"])
    e.add_argument("--checkpoint", required=True, help="Hub repo id or local pretrained_model dir")
    e.add_argument("--revision", help="Hub revision to pin (recommended for Hub checkpoints)")
    e.add_argument("--name", help="results sub-directory name")
    e.add_argument("--out", help="results directory (default results/<benchmark>/<name>)")
    e.add_argument("--seeds", nargs="+", type=int)
    e.add_argument("--suites", nargs="+", help="LIBERO / LIBERO-Plus suites (default: all four)")
    e.add_argument("--tasks", nargs="+", help="RoboTwin tasks (default: all 50)")
    e.add_argument("--server", action="store_true", help="LIBERO: use LeRobot's canonical Docker image")
    e.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    cmds = train_cmds(args, extra) if args.cmd == "train" else eval_cmds(args, extra)
    execute(cmds, args.dry_run)


if __name__ == "__main__":
    main()
