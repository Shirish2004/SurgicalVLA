"""One launcher for training and evaluation on LIBERO, RoboTwin 2.0 and VLABench.

Every benchmark has its own model, configs and outputs:

    configs/benchmarks/<benchmark>/sft.yaml   ->  outputs/<benchmark>/sft
    configs/benchmarks/<benchmark>/rl.yaml    ->  outputs/<benchmark>/rl   (final/ = trained model)

    # supervised fine-tuning (SurgicalVLA, or the unmodified SmolVLA baseline)
    python3 scripts/benchmark.py train sft --benchmark {libero,libero_plus,robotwin,vlabench} [--policy smolvla]
    # online RL (ReinFlow-GRPO; training.mode sequential | joint | interleaved in rl.yaml)
    python3 scripts/benchmark.py train rl  --benchmark {libero,libero_plus,robotwin,vlabench}
    # evaluation of any checkpoint (Hub id or local dir; local dirs resolve to their newest policy)
    python3 scripts/benchmark.py eval --benchmark {libero,libero_plus,robotwin,vlabench} --checkpoint ...
    # everything: SFT then RL on every benchmark, then evaluate every model (SFT and RL) on its benchmark
    python3 scripts/benchmark.py pipeline [--benchmarks libero robotwin vlabench] [--libero-plus]
    # ablation arms (configs/ablations/arms.yaml): shared protocol, 3 seeds, then evaluation
    python3 scripts/benchmark.py ablate --suite libero_spatial [--arms smolvla vision ...]
    # feature/label cache, reasoning traces, overlay videos (--workers N: N GPU processes on one cache)
    python3 scripts/benchmark.py cache --suite libero_spatial [--workers 4]

Data locations (environment): SURGICALVLA_DATA_ROOT holds local copies of the LeRobot datasets
(<root>/libero, <root>/vlabench_unified, <root>/libero_plus; each used when its meta/info.json
exists, otherwise the dataset is read through the Hugging Face cache); ROBOTWIN_ROOT the RoboTwin copy.

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
import os
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
DATASET = {"libero": "lerobot/libero", "libero_plus": "lerobot/libero_plus", "robotwin": "lerobot/robotwin_unified", "vlabench": "lerobot/vlabench_unified"}


def pixi(env: str, *cmd: str) -> list[str]:
    return ["pixi", "run", "--manifest-path", str(ROOT / "pixi.toml"), "-e", env, *cmd]


def execute(cmds: list[list[str]], dry_run: bool) -> None:
    for cmd in cmds:
        print("$", shlex.join(cmd), flush=True)
        if not dry_run:
            code = subprocess.call(cmd, cwd=ROOT)
            if code:
                sys.exit(code)


def execute_parallel(cmds: list[list[str]], dry_run: bool, logs: list[Path], threads: int) -> None:
    """Run ``cmds`` at once, each logging to its file with ``threads`` CPU threads; exit if any fails."""
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    for cmd, log in zip(cmds, logs, strict=True):
        print("$", shlex.join(cmd), ">", log, "&", flush=True)
    if dry_run:
        return
    procs = []
    for cmd, log in zip(cmds, logs, strict=True):
        log.parent.mkdir(parents=True, exist_ok=True)
        procs.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT), log))
    failed = [(log, code) for proc, log in procs if (code := proc.wait())]
    for log, code in failed:
        print(f"# worker failed (exit {code}): see {log}", file=sys.stderr, flush=True)
    if failed:
        sys.exit(failed[0][1])


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
def baseline_hub_args(b: str) -> list[str]:
    """lerobot-train pushes the baseline to <namespace>/surgical-vla-<b>-smolvla-baseline-s1000 (hub: in sft.yaml)."""
    import yaml

    hub = yaml.safe_load((ROOT / f"configs/benchmarks/{b}/sft.yaml").read_text()).get("hub") or {}
    if not hub.get("push"):
        return ["--policy.push_to_hub=false"]
    repo = f"{hub.get('namespace', 'Shirish24')}/surgical-vla-{b}-smolvla-baseline-s1000"
    return ["--policy.push_to_hub=true", f"--policy.repo_id={repo}", f"--policy.private={str(hub.get('private', True)).lower()}"]


def train_cmds(args, extra: list[str]) -> list[list[str]]:
    b = args.benchmark
    if args.stage == "rl":
        cfg_path = args.config or f"configs/benchmarks/{b}/rl.yaml"
        resume = ["run.resume=true"] if getattr(args, "resume", False) else []
        rl = [pixi("sim-eval", "python", "scripts/train_reinflow_grpo.py", "--config", cfg_path, *resume, *extra)]
        if b in ("libero", "libero_plus"):  # LIBERO-plus rolls out in the local LIBERO simulator
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
        }
        recipe["libero_plus"] = recipe["libero"] + ([f"--dataset.root={local_root('libero_plus')}"] if local_root("libero_plus") else [])
        recipe = recipe[b]
        return [
            pixi(
                "dev", "lerobot-train",
                f"--policy.path={SMOLVLA_BASE[0]}", f"--policy.pretrained_revision={SMOLVLA_BASE[1]}",
                *baseline_hub_args(b), "--policy.device=cuda",
                f"--dataset.repo_id={DATASET[b]}", f"--rename_map={json.dumps(RENAME[b])}",
                *recipe, "--save_freq=5000", "--seed=1000", "--num_workers=4",
                f"--output_dir={out}", f"--job_name=smolvla_{b}",
                "--trackio.enable=true", "--trackio.project=surgical-vla",
                *extra,
            )
        ]

    cmd = ["python", "scripts/train.py", "--config", args.config or f"configs/benchmarks/{b}/sft.yaml", "--repo_id", DATASET[b]]
    if b == "robotwin":  # local copy of robotwin_unified
        cmd += ["--root", ROBOTWIN_ROOT, f"dataset.root={ROBOTWIN_ROOT}"]
    else:  # the local copy under SURGICALVLA_DATA_ROOT when there is one, else the Hugging Face cache
        cmd.append(f"dataset.root={local_root(b) or 'null'}")
    if getattr(args, "resume", False):  # newest checkpoints/step_N under the output directory
        cmd += ["--resume", "latest"]
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
def suite_episodes(suite: str) -> Path:
    path = ROOT / "configs" / "data" / f"libero_{suite}_episodes.json"
    if not path.exists():
        sys.exit(f"{path} missing: run `pixi run -e sim-eval python scripts/libero_suite_episodes.py --suite {suite}`")
    return path


SIDECAR = "pseudolabels"
ARMS_FILE = ROOT / "configs" / "ablations" / "arms.yaml"


def load_arms() -> dict:
    """``configs/ablations/arms.yaml``; every arm must set every switch."""
    import yaml

    spec = yaml.safe_load(ARMS_FILE.read_text())
    for name, arm in spec["arms"].items():
        missing = set(spec["switches"]) - set(arm["set"])
        if missing:
            raise ValueError(f"arm {name} does not set {sorted(missing)}")
    return spec


def arm_overrides(spec: dict, arm: str, suite: str, seed: int, steps: int | None = None) -> tuple[str, list[str]]:
    """Output directory and trainer overrides of one (arm, suite, seed) run."""
    proto, a = spec["protocol"], spec["arms"][arm]
    out = f"outputs/ablations/{suite}/{arm}/seed{seed}"
    over = [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in a["set"].items()]
    over += [f"run.ckpt_dir={out}", f"run.seed={seed}", f"run.steps={steps or proto['steps']}",
             f"model.scheduler_decay_steps={steps or proto['steps']}", f"run.batch_size={proto['batch_size']}",
             f"run.grad_accum={proto['grad_accum']}", f"dataset.episodes={suite_episodes(suite).relative_to(ROOT)}",
             f"dataset.val_fraction={proto['val_fraction']}", f"aux.sidecar_root={SIDECAR}",
             f"hub.model_name=libero-{suite.removeprefix('libero_')}-{arm.replace('_', '-')}-s{seed}",
             f"project.group={suite}/{arm}", f"project.run_name={suite}-{arm}-s{seed}", "project.job_type=sft"]
    if a.get("requires_cache"):
        over.append(f"aux.cache_name={a['requires_cache']}")
    return out, over


def ablate(args, extra: list[str]) -> None:
    """Each arm x seed: SFT on one LIBERO suite under the shared protocol, then evaluation.

    Safe to re-run: finished training runs and evaluations are skipped, interrupted training
    resumes from its newest checkpoint, arms whose cache does not exist yet are skipped, and a
    failed run is reported without stopping the others.
    """
    spec = load_arms()
    arms = args.arms or (spec["ladder"] if args.suite == "libero_spatial" else spec["per_suite"])
    steps = args.steps or spec["protocol"]["steps"]
    failed = []
    for arm in arms:
        need = spec["arms"][arm].get("requires_cache")
        if need and not args.dry_run and not list((ROOT / SIDECAR).glob(f"*/{need}/manifest.json")):
            print(f"# skip {arm}: needs the '{need}' cache, not built yet", flush=True)
            continue
        for seed in args.seeds or spec["protocol"]["seeds"]:
            out, over = arm_overrides(spec, arm, args.suite, seed, args.steps)
            ckpts = ROOT / out / "checkpoints"
            done = (ckpts / f"step_{steps}" / "model.safetensors").exists()
            resume = ["--resume", "latest"] if ckpts.exists() and any(ckpts.glob("step_*")) and not done else []
            train = pixi("dev", "python", "scripts/train.py", "--config", spec["protocol"]["config"],
                         "--repo_id", spec["protocol"]["repo_id"], *resume, *over, *extra)
            res = ROOT / f"results/ablations/{args.suite}/{arm}/seed{seed}"
            ev = argparse.Namespace(benchmark="libero", checkpoint=out, revision=None, name=f"{arm}_seed{seed}",
                                    out=str(res.relative_to(ROOT)), seeds=spec["protocol"]["eval_seeds"],
                                    suites=[args.suite], tasks=None, local=False)
            try:
                if not done:
                    execute([train], args.dry_run)
                if not (res / "summary.json").exists() or args.dry_run:
                    execute(eval_cmds(ev, []), args.dry_run)
            except SystemExit as exc:
                failed.append(f"{arm}/seed{seed} (exit {exc.code})")
                print(f"# FAILED {arm} seed {seed}: exit {exc.code}; continuing", flush=True)
    if failed:
        print("# failed runs: " + ", ".join(failed), flush=True)
        sys.exit(1)


ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/media/protomate/Seagate/shirish/robotwin_data")
# local copies of the LeRobot datasets: <DATA_ROOT>/libero, <DATA_ROOT>/vlabench_unified, <DATA_ROOT>/libero_plus
DATA_ROOT = os.environ.get("SURGICALVLA_DATA_ROOT", "/media/protomate/Seagate/shirish/lerobot_data")
LOCAL_DIR = {"libero": "libero", "vlabench": "vlabench_unified", "libero_plus": "libero_plus"}


def local_root(benchmark: str) -> str | None:
    """Local copy of the benchmark's dataset (``hf download ... --local-dir``) if complete enough to read, else None."""
    if benchmark == "robotwin":
        return ROBOTWIN_ROOT
    path = Path(DATA_ROOT) / LOCAL_DIR[benchmark]
    return str(path) if (path / "meta" / "info.json").exists() else None

# per benchmark: dataset (read from local_root() when present), labelled camera (the frontal view), gripper action dims, grippers as objects,
# strides (targets, inputs) for its fps
CACHE = {
    "libero": dict(repo="lerobot/libero", camera="observation.images.image", dims=["6"], grippers=1, stride=4, input_stride=1,
                   ckpt=("lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"),
                   rename={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}),
    "robotwin": dict(repo="lerobot/robotwin_unified", camera="observation.images.cam_high", dims=["6", "13"], grippers=2, stride=12, input_stride=3,
                     ckpt=("lerobot/smolvla_robotwin", "eb55d3a74f6cd26c6388aecd116b37b63fac144d"),
                     rename={"observation.images.cam_high": "observation.images.camera1",
                             "observation.images.cam_left_wrist": "observation.images.camera2",
                             "observation.images.cam_right_wrist": "observation.images.camera3"}),
    "vlabench": dict(repo="lerobot/vlabench_unified", camera="observation.images.image", dims=["6"],
                     grippers=1, stride=4, input_stride=1, ckpt=("lerobot/smolvla_vlabench", "7ef370813eb35823caa94088b50f5389a51e4c47"),
                     rename=RENAME["vlabench"]),
    "libero_plus": dict(repo="lerobot/libero_plus", camera="observation.images.front", dims=["6"],
                        grippers=1, stride=8, input_stride=2,  # 20 fps: the same target and input rates as LIBERO at 10 fps
                        ckpt=("lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"),
                        rename={"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}),
}


def cache(args, extra: list[str]) -> None:
    """Feature and label cache for one episode set (a LIBERO suite or a benchmark subset), bootstrap traces, overlay videos."""
    c = CACHE[args.benchmark]
    if args.benchmark == "libero":
        path, name = suite_episodes(args.suite), args.suite
    else:
        path, name = ROOT / "configs" / "data" / f"{args.benchmark}_{args.suite}_episodes.json", f"{args.benchmark}_{args.suite}"
        if not path.exists():
            maker = "robotwin_subset_episodes.py" if args.benchmark == "robotwin" else f"subset_episodes.py --benchmark {args.benchmark}"
            sys.exit(f"{path} missing: run `python3 scripts/{maker}`")
    episodes = str(path.relative_to(ROOT))
    if args.limit:
        episodes = ",".join(map(str, json.loads(path.read_text())[: args.limit]))
    stride = args.stride or c["stride"]
    input_stride = args.input_stride or c["input_stride"]
    root = ["--root", local_root(args.benchmark)] if local_root(args.benchmark) else []
    cmds = [
        pixi("dev", "python", "scripts/make_pseudolabels.py", "--repo-id", c["repo"], *root, "--episodes", episodes, "--camera", c["camera"],
             "--gripper-dims", *c["dims"], "--num-grippers", str(c["grippers"]), "--out", SIDECAR,
             "--stride", str(stride), "--input-stride", str(input_stride),
             "--save-videos", f"results/pseudolabels/videos/{name}", "--report", f"results/pseudolabels/{name}_cost.json",
             "--offload-video", *(["--gpu-mem-gb", str(args.gpu_mem_gb)] if args.gpu_mem_gb else []),
             *(["--upload-repo", args.upload_repo] if args.upload_repo else []), *extra),
        pixi("dev", "python", "scripts/make_cot_traces.py", "--repo-id", c["repo"], *root, "--episodes", episodes,
             "--sidecar", SIDECAR, "--checkpoint", c["ckpt"][0], "--revision", c["ckpt"][1], "--bootstrap",
             "--rename", json.dumps(c["rename"])),  # dataset camera -> policy camera
    ]
    cmds = cmds if not args.no_traces else cmds[:1]
    workers = getattr(args, "workers", 1) or 1
    if workers == 1:
        execute(cmds, args.dry_run)
        return
    # N processes per stage, each on every N-th episode, all writing the same cache (it locks its
    # manifest); traces start once every labelling worker has finished
    threads = max(1, (os.cpu_count() or workers) // workers)
    for stage, cmd in zip(("labels", "traces"), cmds):
        shards = [[*cmd, "--shard", f"{i}/{workers}"] for i in range(workers)]
        logs = [ROOT / "results" / "plan" / f"{name}_{stage}_w{i}.log" for i in range(workers)]
        execute_parallel(shards, args.dry_run, logs, threads)


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
    t.add_argument("--benchmark", required=True, choices=["libero", "libero_plus", "robotwin", "vlabench"])
    t.add_argument("--policy", default="surgicalvla", choices=["smolvla", "surgicalvla"])
    t.add_argument("--config")
    t.add_argument("--out")
    t.add_argument("--resume", action="store_true",
                   help="continue from the newest checkpoint in the output directory (weights, optimizer, data position, logs)")
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
    p.add_argument("--benchmarks", nargs="+", default=["libero", "robotwin", "vlabench"], choices=["libero", "libero_plus", "robotwin", "vlabench"])
    p.add_argument("--libero-plus", action="store_true", help="also evaluate the LIBERO models on LIBERO-Plus")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-rl", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    pl = sub.add_parser("ablate", aliases=["pilot"], help="ablation arms of configs/ablations/arms.yaml")
    pl.add_argument("--suite", default="libero_spatial")
    pl.add_argument("--arms", nargs="+", help="default: the ladder on libero_spatial, per_suite elsewhere")
    pl.add_argument("--seeds", nargs="+", type=int)
    pl.add_argument("--steps", type=int, help="override protocol.steps (all arms alike)")
    pl.add_argument("--dry-run", action="store_true")

    lb = sub.add_parser("cache", aliases=["labels"], help="feature/label cache for one LIBERO suite or benchmark subset")
    lb.add_argument("--benchmark", default="libero", choices=sorted(CACHE))
    lb.add_argument("--suite", default="libero_spatial", help="LIBERO suite, or the subset name (RoboTwin subset10; VLABench and LIBERO-plus subset)")
    lb.add_argument("--stride", type=int, help="targets every k-th frame (default: CACHE[benchmark]['stride'])")
    lb.add_argument("--limit", type=int, default=0, help="first N episodes only (0 = all)")
    lb.add_argument("--gpu-mem-gb", type=float, help="cap labelling's GPU memory, e.g. 8 when training shares the GPU")
    lb.add_argument("--input-stride", type=int, help="object inputs every k-th frame (default: CACHE[benchmark]['input_stride'])")
    lb.add_argument("--upload-repo", help="Hub dataset for finished shards (bounded local disk)")
    lb.add_argument("--no-traces", action="store_true", help="skip the bootstrap reasoning traces")
    lb.add_argument("--workers", type=int, default=1,
                    help="parallel GPU processes on one cache, each on every N-th episode (A100 80 GB: 4-5 with --gpu-mem-gb 14)")
    lb.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd in ("ablate", "pilot"):
        ablate(args, extra)
    elif args.cmd in ("cache", "labels"):
        cache(args, extra)
    elif args.cmd == "pipeline":
        pipeline(args, extra)
    else:
        execute(train_cmds(args, extra) if args.cmd == "train" else eval_cmds(args, extra), args.dry_run)


if __name__ == "__main__":
    main()
