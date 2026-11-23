"""One launcher for training and evaluation on LIBERO, RoboTwin 2.0 and VLABench.

Every benchmark has its own model, configs and outputs:

    configs/benchmarks/<benchmark>/sft.yaml   ->  outputs/<benchmark>/sft
    configs/benchmarks/<benchmark>/rl.yaml    ->  outputs/<benchmark>/rl   (final/ = trained model)

    # supervised fine-tuning (SurgicalVLA, or the unmodified SmolVLA baseline)
    python3 scripts/benchmark.py train sft --benchmark {libero,robotwin,vlabench} [--policy smolvla]
    # online RL (ReinFlow-GRPO; training.mode sequential | joint | interleaved in rl.yaml)
    python3 scripts/benchmark.py train rl  --benchmark {libero,robotwin,vlabench}
    # evaluation of any checkpoint (Hub id or local dir; local dirs resolve to their newest policy)
    python3 scripts/benchmark.py eval --benchmark {libero,libero_plus,robotwin,vlabench} --checkpoint ...
    # everything: SFT then RL on every benchmark, then evaluate every model (SFT and RL) on its benchmark
    python3 scripts/benchmark.py pipeline [--benchmarks libero robotwin vlabench] [--libero-plus]
    # Step-4 pilot: ablation arms at matched compute on one LIBERO suite, 3 seeds, then evaluation
    python3 scripts/benchmark.py pilot --suite libero_spatial --steps 20000 [--arms baseline vision_aux ...]
    # pseudo-labels, reasoning traces and their validation
    python3 scripts/benchmark.py labels --suite libero_spatial

``--dry-run`` prints the commands; arguments after ``--`` go to the trainer / lerobot-eval.

Simulators: evaluation always uses LeRobot's canonical Docker images behind
``lerobot-env-server`` (started and stopped here); ``--local`` runs LIBERO in the
``sim-eval`` env instead, which resolves newer simulator versions and is not the protocol.
RL rollouts on LIBERO run locally; RoboTwin and VLABench rollouts use the Docker images.

Evaluation protocols (docs/baselines.md):
* libero       4 suites x 10 tasks x 50 init states x 3 seeds, checkpoint's n_action_steps (50)
* libero_plus  every registered variant of the 4 perturbed suites, 1 episode each
* robotwin     50 tasks x 100 episodes, Easy (LeRobot's integration hard-codes demo_clean)
* vlabench     21 primitive + 22 composite tasks x 10 episodes x 3 seeds (VLABench paper protocol)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEROBOT = ROOT / "lerobot"
ENDPOINT = "tcp/127.0.0.1:7448"
SEEDS = [1000, 2000, 3000]
SMOLVLA_BASE = ("lerobot/smolvla_base", "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
ROBOTWIN_TASKS = sorted(p.stem for p in (ROOT.parent / "RoboTwin" / "description" / "task_instruction").glob("*.json"))
VLABENCH_PRIMITIVE = (
    "select_fruit select_toy select_chemistry_tube add_condiment select_book select_painting select_drink "
    "insert_flower select_billiards select_ingredient select_mahjong select_poker density_qa friction_qa "
    "magnetism_qa reflection_qa simple_cuestick_usage simple_seesaw_usage sound_speed_qa thermal_expansion_qa weight_qa"
).split()
VLABENCH_COMPOSITE = (
    "cluster_billiards cluster_book cluster_drink cluster_toy cook_dishes cool_drink find_unseen_object get_coffee "
    "hammer_nail heat_food make_juice play_mahjong play_math_game play_poker play_snooker rearrange_book "
    "rearrange_chemistry_tube set_dining_table set_study_table store_food take_chemistry_experiment use_seesaw_complex"
).split()

RENAME = {  # simulator camera -> policy camera (all SurgicalVLA and SmolVLA checkpoints use camera1..3)
    "libero": {"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"},
    "robotwin": {
        "observation.images.head_camera": "observation.images.camera1",
        "observation.images.left_camera": "observation.images.camera2",
        "observation.images.right_camera": "observation.images.camera3",
    },
    "vlabench": {
        "observation.images.image": "observation.images.camera1",
        "observation.images.second_image": "observation.images.camera2",
        "observation.images.wrist_image": "observation.images.camera3",
    },
}
RENAME["libero_plus"] = RENAME["libero"]
DATASET = {"libero": "lerobot/libero", "robotwin": "lerobot/robotwin_unified", "vlabench": "lerobot/vlabench_unified"}


def pixi(env: str, *cmd: str) -> list[str]:
    return ["pixi", "run", "--manifest-path", str(ROOT / "pixi.toml"), "-e", env, *cmd]


def execute(cmds: list[list[str]], dry_run: bool) -> None:
    for cmd in cmds:
        print("$", shlex.join(cmd), flush=True)
        if not dry_run:
            code = subprocess.call(cmd, cwd=ROOT)
            if code:
                sys.exit(code)


def resolve_policy(ref: str) -> str:
    """Local training output -> its newest saved policy dir (Hub ids pass through)."""
    path = ROOT / ref if not Path(ref).is_absolute() else Path(ref)
    if not path.exists():
        return ref
    from surgicalvla.utils.checkpoints import latest_pretrained

    try:
        return str(latest_pretrained(path))
    except FileNotFoundError:
        print(f"# note: no saved policy under {ref} yet", file=sys.stderr)
        return ref


# ----------------------------------------------------------------------------- servers
def server_config(benchmark: str, task: str, out_dir: Path) -> Path:
    """LeRobot's canonical server config for ``benchmark`` serving ``task`` (all variants)."""
    import yaml

    cfg = yaml.safe_load((LEROBOT / "docker" / "sim" / "configs" / f"{benchmark}.yaml").read_text())
    cfg["sim"]["task"] = task
    cfg["sim"]["kwargs"].get("gym_kwargs", {}).pop("task_ids", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(task.encode()).hexdigest()[:10]
    path = out_dir / f"server_{benchmark}_{digest}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def served(benchmark: str, task: str, out_dir: Path, inner: list[list[str]]) -> list[list[str]]:
    cfg = server_config(benchmark, task, out_dir)
    start = pixi("sim-eval", "lerobot-env-server", f"--config_path={cfg}", "--docker=true")
    return [start, *inner, [*start, "--stop=true"]]


# ----------------------------------------------------------------------------- train
def train_cmds(args, extra: list[str]) -> list[list[str]]:
    b = args.benchmark
    if args.stage == "rl":
        cfg_path = args.config or f"configs/benchmarks/{b}/rl.yaml"
        rl = [pixi("sim-eval", "python", "scripts/train_reinflow_grpo.py", "--config", cfg_path, *extra)]
        if b == "libero":
            return rl
        import yaml

        suite = yaml.safe_load((ROOT / cfg_path).read_text())["env"]["suite"]
        return served(b, suite, ROOT / f"outputs/{b}/servers", rl)

    if args.policy == "smolvla":  # unmodified SmolVLA baselines through lerobot-train
        out = args.out or f"outputs/baselines/smolvla_{b}"
        if b == "robotwin":
            return [["env", f"OUT={out}", *pixi("dev", "bash", "scripts/train_robotwin_baseline.sh", *extra)]]
        recipe = {  # the recipes of the released lerobot/smolvla_<benchmark> checkpoints
            "libero": ["--policy.freeze_vision_encoder=false", "--policy.train_expert_only=false",
                       "--policy.scheduler_decay_steps=25000", "--batch_size=32", "--steps=25000"],
            "vlabench": ["--batch_size=32", "--steps=20000"],
        }[b]
        return [
            pixi(
                "dev", "lerobot-train",
                f"--policy.path={SMOLVLA_BASE[0]}", f"--policy.pretrained_revision={SMOLVLA_BASE[1]}",
                "--policy.push_to_hub=false", "--policy.device=cuda",
                f"--dataset.repo_id={DATASET[b]}", f"--rename_map={json.dumps(RENAME[b])}",
                *recipe, "--save_freq=5000", "--seed=1000", "--num_workers=4",
                f"--output_dir={out}", f"--job_name=smolvla_{b}",
                "--trackio.enable=true", "--trackio.project=surgical-vla",
                *extra,
            )
        ]

    cmd = ["python", "scripts/train.py", "--config", args.config or f"configs/benchmarks/{b}/sft.yaml", "--repo_id", DATASET[b]]
    if b == "robotwin":  # local copy of robotwin_unified (root set in the config)
        cmd += ["--root", "/media/protomate/Seagate/shirish/robotwin_data"]
    if args.out:
        cmd.append(f"run.ckpt_dir={args.out}")
    return [pixi("dev", *cmd, *extra)]


# ----------------------------------------------------------------------------- eval
def eval_cmds(args, extra: list[str]) -> list[list[str]]:
    b = args.benchmark
    checkpoint = resolve_policy(args.checkpoint)
    name = args.name or Path(args.checkpoint.rstrip("/")).name.replace("/", "_")
    out = Path(args.out or f"results/{b}/{name}")
    seeds = [str(s) for s in (args.seeds or SEEDS)]
    revision = ["--revision", args.revision] if args.revision else ["--revision", ""]

    if b in ("libero", "libero_plus"):
        cmds: list[list[str]] = []
        for suite in args.suites or LIBERO_SUITES:
            run = ["python", "scripts/eval_libero_baseline.py", "run", "--policy", checkpoint, *revision,
                   "--out", str(out), "--seeds", *(seeds[:1] if b == "libero_plus" else seeds),
                   "--suites", suite, "--env-type", b]
            if b == "libero_plus":  # upstream protocol: one episode per registered variant
                run += ["--n-episodes", "1", "--batch-size", "1"]
            if b == "libero_plus" or not args.local:
                run += ["--server-endpoint", ENDPOINT]
                cmds += served(b, suite, out, [pixi("sim-eval", *run, *extra)])
            else:
                cmds.append(pixi("sim-eval", *run, *extra))
        return cmds + [pixi("sim-eval", "python", "scripts/eval_libero_baseline.py", "aggregate", "--root", str(out))]

    if b == "robotwin":
        suites = {"easy": args.tasks or ROBOTWIN_TASKS}
        n_episodes = 100
    else:  # vlabench
        suites = {"primitive": args.tasks or VLABENCH_PRIMITIVE} if args.tasks else {"primitive": VLABENCH_PRIMITIVE, "composite": VLABENCH_COMPOSITE}
        n_episodes = 10
    cmds = []
    for suite, tasks in suites.items():
        inner = [
            pixi(
                "sim-eval", "python", "scripts/lerobot_eval.py",
                f"--policy.path={checkpoint}",
                *([f"--policy.pretrained_revision={args.revision}"] if args.revision else []),
                "--policy.device=cuda", f"--env.type={b}", f"--env.task={','.join(tasks)}",
                f"--eval.n_episodes={n_episodes}", "--eval.batch_size=1", "--eval.use_async_envs=false",
                f"--seed={seed}", f"--rename_map={json.dumps(RENAME[b])}", f"--env.server.endpoint={ENDPOINT}",
                f"--output_dir={out / suite / f'seed{seed}'}",
                *extra,
            )
            for seed in (seeds[:1] if b == "robotwin" else seeds)
        ]
        cmds += served(b, ",".join(tasks), out, inner)
    return cmds


# ----------------------------------------------------------------------------- pilot
SIDECAR = "pseudolabels"
_NO_COT = ["smolvla_moe.use_cot=false"]
ARMS = {
    "baseline": _NO_COT,
    "vision_aux": [*_NO_COT, "smolvla_moe.aux_vision=true", f"aux.sidecar_root={SIDECAR}"],
    "cot_latent": [*_NO_COT, "smolvla_moe.latent_cot=true"],
    "cot_explicit": ["smolvla_moe.use_cot=true", f"aux.sidecar_root={SIDECAR}", "aux.cot_traces=true"],
    "both": ["smolvla_moe.use_cot=true", "smolvla_moe.aux_vision=true", "smolvla_moe.latent_cot=true",
             f"aux.sidecar_root={SIDECAR}", "aux.cot_traces=true"],
    "moe_matched": [*_NO_COT, "smolvla_moe.moe_variant=matched"],
    "dense": [*_NO_COT, "smolvla_moe.moe_variant=dense"],
    "dense_param_matched": [*_NO_COT, "smolvla_moe.moe_variant=dense_param_matched"],
}


def suite_episodes(suite: str) -> Path:
    path = ROOT / "configs" / "data" / f"libero_{suite}_episodes.json"
    if not path.exists():
        sys.exit(f"{path} missing: run `pixi run -e sim-eval python scripts/libero_suite_episodes.py --suite {suite}`")
    return path


def pilot(args, extra: list[str]) -> None:
    """Each arm x seed: SFT on one LIBERO suite with identical steps, batch and data, then evaluation."""
    episodes = suite_episodes(args.suite).relative_to(ROOT)
    for arm in args.arms:
        for seed in args.seeds:
            out = f"outputs/pilot/{args.suite}/{arm}/seed{seed}"
            overrides = [*ARMS[arm], f"run.ckpt_dir={out}", f"run.seed={seed}", f"run.steps={args.steps}",
                         f"dataset.episodes={episodes}", "dataset.val_fraction=0"]
            if arm == "both":
                overrides.append(f"smolvla_moe.cot_curriculum_steps={args.steps // 2}")
            train = pixi("dev", "python", "scripts/train.py", "--config", "configs/benchmarks/libero/sft.yaml",
                         "--repo_id", "lerobot/libero", *overrides, *extra)
            execute([train], args.dry_run)
            ev = argparse.Namespace(benchmark="libero", checkpoint=out, revision=None, name=f"{arm}_seed{seed}",
                                    out=f"results/pilot/{args.suite}/{arm}/seed{seed}", seeds=[1000],
                                    suites=[args.suite], tasks=None, local=False)
            execute(eval_cmds(ev, []), args.dry_run)


def labels(args, extra: list[str]) -> None:
    """Pseudo-labels for one suite's demonstrations, bootstrap reasoning traces, validation videos."""
    episodes = ",".join(map(str, json.loads(suite_episodes(args.suite).read_text())[: args.limit or None]))
    cmds = [
        pixi("dev", "python", "scripts/make_pseudolabels.py", "--repo-id", "lerobot/libero", "--episodes", episodes,
             "--gripper-dims", "6", "--out", SIDECAR, "--stride", str(args.stride),
             "--save-videos", f"results/pseudolabels/videos/{args.suite}", "--report", f"results/pseudolabels/{args.suite}_cost.json"),
        pixi("dev", "python", "scripts/make_cot_traces.py", "--repo-id", "lerobot/libero", "--episodes", episodes,
             "--sidecar", SIDECAR, "--checkpoint", "lerobot/smolvla_libero",
             "--revision", "a24f1f380461b665158d52acd140152c79e8b36d", "--bootstrap"),
    ]
    execute(cmds, args.dry_run)


# ----------------------------------------------------------------------------- pipeline
def pipeline(args, extra: list[str]) -> None:
    """SFT -> RL per benchmark, then evaluate every benchmark's SFT and RL model on its benchmark."""
    ns = argparse.Namespace
    for b in args.benchmarks:
        if not args.skip_sft:
            execute(train_cmds(ns(stage="sft", benchmark=b, policy="surgicalvla", config=None, out=None), extra), args.dry_run)
        if not args.skip_rl:
            execute(train_cmds(ns(stage="rl", benchmark=b, policy="surgicalvla", config=None, out=None), extra), args.dry_run)
    for b in args.benchmarks:
        for stage in ("sft", "rl"):
            targets = [b, "libero_plus"] if b == "libero" and args.libero_plus else [b]
            for target in targets:
                ev = ns(benchmark=target, checkpoint=f"outputs/{b}/{stage}", revision=None, name=f"surgicalvla_{stage}",
                        out=None, seeds=None, suites=None, tasks=None, local=False)
                if args.dry_run and not (ROOT / f"outputs/{b}/{stage}").exists():
                    print(f"# (eval of outputs/{b}/{stage} once it exists)")
                execute(eval_cmds(ev, []), args.dry_run)


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
    t.add_argument("--benchmark", required=True, choices=["libero", "robotwin", "vlabench"])
    t.add_argument("--policy", default="surgicalvla", choices=["smolvla", "surgicalvla"])
    t.add_argument("--config")
    t.add_argument("--out")
    t.add_argument("--dry-run", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--benchmark", required=True, choices=["libero", "libero_plus", "robotwin", "vlabench"])
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--revision")
    e.add_argument("--name")
    e.add_argument("--out")
    e.add_argument("--seeds", nargs="+", type=int)
    e.add_argument("--suites", nargs="+")
    e.add_argument("--tasks", nargs="+")
    e.add_argument("--local", action="store_true", help="LIBERO: local simulator instead of LeRobot's canonical image (not the protocol)")
    e.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("pipeline")
    p.add_argument("--benchmarks", nargs="+", default=["libero", "robotwin", "vlabench"], choices=["libero", "robotwin", "vlabench"])
    p.add_argument("--libero-plus", action="store_true", help="also evaluate the LIBERO models on LIBERO-Plus")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-rl", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    pl = sub.add_parser("pilot")
    pl.add_argument("--suite", default="libero_spatial")
    pl.add_argument("--arms", nargs="+", default=["baseline", "vision_aux", "cot_latent", "cot_explicit", "both"], choices=list(ARMS))
    pl.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    pl.add_argument("--steps", type=int, default=20000, help="identical for every arm (matched compute)")
    pl.add_argument("--dry-run", action="store_true")

    lb = sub.add_parser("labels")
    lb.add_argument("--suite", default="libero_spatial")
    lb.add_argument("--stride", type=int, default=2)
    lb.add_argument("--limit", type=int, default=0, help="first N episodes only (0 = all)")
    lb.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "pilot":
        pilot(args, extra)
    elif args.cmd == "labels":
        labels(args, extra)
    elif args.cmd == "pipeline":
        pipeline(args, extra)
    else:
        execute(train_cmds(args, extra) if args.cmd == "train" else eval_cmds(args, extra), args.dry_run)


if __name__ == "__main__":
    main()
