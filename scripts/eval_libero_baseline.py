"""Seeded LIBERO / LIBERO-Plus benchmark runs through ``lerobot-eval`` + aggregation.

    pixi run -e sim-eval python scripts/eval_libero_baseline.py run \
        --policy lerobot/smolvla_libero --revision a24f1f38... \
        --suites libero_spatial libero_object libero_goal libero_10 --seeds 1000 2000 3000
    pixi run -e sim-eval python scripts/eval_libero_baseline.py aggregate --root results/baselines/smolvla_libero

Protocol (docs/baselines.md): every task x all 50 benchmark init states x each
seed. ``batch_size`` divides ``n_episodes`` and vector environment ``j`` uses init states
``j, j + b, j + 2b, ...``, so every init state is used exactly once per seed and episode
``i`` of a task runs init state ``i``.
The seed changes the policy's flow noise and the env RNG, not the init states.

Statistics. Episodes that share an init state across seeds are correlated, so a
Wilson interval over the pooled episodes would be too narrow. We report

* mean +- std of the per-seed success rate (n = #seeds),
* a 95 % cluster-bootstrap CI that resamples (task, init state) clusters with
  all their seeds (the unit the benchmark actually randomises over),
* the pooled Wilson interval, for comparison with ``lerobot-eval``'s own output.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
RENAME = {"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def cluster_bootstrap(success: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """``success``: (clusters, seeds) 0/1 array. Resample clusters with replacement."""
    rng = np.random.default_rng(seed)
    per_cluster = success.mean(axis=1)
    idx = rng.integers(0, len(per_cluster), size=(n_boot, len(per_cluster)))
    means = per_cluster[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ----------------------------------------------------------------------------- run
def run(args) -> None:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol.json").write_text(json.dumps(vars(args), indent=2, default=str))
    for suite in args.suites:
        for seed in args.seeds:
            out = root / suite / f"seed{seed}"
            if (out / "eval_info.json").exists() and not args.overwrite:
                print(f"[skip] {out} exists")
                continue
            cmd = [
                sys.executable,
                str(Path(__file__).with_name("lerobot_eval.py")),
                f"--policy.path={args.policy}",
                f"--policy.device={args.device}",
                f"--env.type={args.env_type}",
                f"--env.task={suite}",
                f"--env.observation_height={args.resolution}",
                f"--env.observation_width={args.resolution}",
                f"--eval.n_episodes={args.n_episodes}",
                f"--eval.batch_size={args.batch_size}",
                f"--seed={seed}",
                f"--rename_map={json.dumps(RENAME)}",
                f"--output_dir={out}",
            ]
            if args.revision:
                cmd.insert(2, f"--policy.pretrained_revision={args.revision}")
            if args.server_endpoint:
                cmd.append(f"--env.server.endpoint={args.server_endpoint}")
            if args.task_ids:
                cmd.append(f"--env.task_ids={json.dumps(args.task_ids)}")
            cmd += [f"--policy.{kv}" for kv in args.policy_arg or []]
            print("[run]", " ".join(cmd), flush=True)
            t0 = time.time()
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "stdout.log", "w") as log:
                code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(f"[done] {suite} seed={seed} exit={code} in {time.time() - t0:.0f}s", flush=True)
            if code != 0:
                sys.exit(f"lerobot-eval failed, see {out / 'stdout.log'}")


# ----------------------------------------------------------------------------- aggregate
def _episodes(info: dict) -> dict[str, list[bool]]:
    """task name -> list of per-episode successes (episode order == init-state order)."""
    out: dict[str, list[bool]] = {}
    per_task = info.get("per_task") or []
    for entry in per_task:
        name = str(entry.get("task_group", "")) + ":" + str(entry.get("task_id", entry.get("task", "")))
        metrics = entry.get("metrics", entry)
        succ = metrics.get("successes")
        if succ is None and "per_episode" in metrics:
            succ = [e.get("success", e.get("is_success")) for e in metrics["per_episode"]]
        if succ is None:
            raise KeyError(f"no per-episode successes in eval_info entry keys {list(metrics)}")
        out[name] = [bool(s) for s in succ]
    if not out:
        raise KeyError(f"unrecognised eval_info.json layout, top-level keys {list(info)}")
    return out


def init_state_of(episode_ix: int, n_init_states: int = 50) -> int:
    """Benchmark init state of the ``episode_ix``-th episode of a task.

    LeRobot's LIBERO environment ``j`` of ``b`` starts at init state ``j`` and advances by
    ``b`` on every reset (``sim/integrations/libero.py``), and episodes are recorded rollout by
    rollout, so episode ``r * b + j`` uses init state ``r * b + j``: the episode index itself.
    """
    return int(episode_ix) % n_init_states


def episode_records(root: Path, suite: str, seed_file: Path, info: dict) -> list[dict]:
    """One record per episode: (suite, seed, task, episode_ix, init_state, success, length, outcome)."""
    lengths_path = seed_file.parent / "episode_lengths.json"
    lengths = json.loads(lengths_path.read_text()) if lengths_path.exists() else None
    eps = _episodes(info)
    total = sum(len(v) for v in eps.values())
    if lengths is not None and len(lengths) != total:
        lengths = None  # tasks ran in parallel or the file is from another run: lengths unknown
    out, k = [], 0
    for task in eps:  # eval_info keeps LeRobot's task order
        for ix, success in enumerate(eps[task]):
            length = None if lengths is None else int(lengths[k])
            out.append({"suite": suite, "seed": seed_file.parent.name, "task": task, "episode_ix": ix,
                        "init_state": init_state_of(ix), "success": bool(success),
                        "length": length, "outcome": "success" if success else "timeout"})
            k += 1
    return out


def aggregate(args) -> dict:
    root = Path(args.root)
    report: dict = {"suites": {}}
    records: list[dict] = []
    for suite_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seeds = sorted(suite_dir.glob("seed*/eval_info.json"))
        if not seeds:
            continue
        infos = [json.loads(p.read_text()) for p in seeds]
        runs = [_episodes(i) for i in infos]
        for p, i in zip(seeds, infos, strict=True):
            records += episode_records(root, suite_dir.name, p, i)
        tasks = sorted(runs[0])
        n_ep = min(len(r[t]) for r in runs for t in tasks)
        # (task, init_state) clusters x seeds
        grid = np.array([[[r[t][e] for r in runs] for e in range(n_ep)] for t in tasks], dtype=float)
        clusters = grid.reshape(-1, len(runs))
        per_seed = clusters.mean(axis=0)
        k, n = int(clusters.sum()), clusters.size
        report["suites"][suite_dir.name] = {
            "n_tasks": len(tasks),
            "episodes_per_task_per_seed": n_ep,
            "seeds": [p.parent.name for p in seeds],
            "success_mean": float(per_seed.mean()),
            "success_std_over_seeds": float(per_seed.std(ddof=1)) if len(runs) > 1 else 0.0,
            "per_seed": per_seed.tolist(),
            "ci95_cluster_bootstrap": cluster_bootstrap(clusters),
            "ci95_wilson_pooled": wilson(k, n),
            "per_task": {t: float(grid[i].mean()) for i, t in enumerate(tasks)},
        }
    suites = report["suites"]
    if suites:
        report["average"] = float(np.mean([s["success_mean"] for s in suites.values()]))
    (root / "summary.json").write_text(json.dumps(report, indent=2))
    if records:
        (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for name, s in suites.items():
        lo, hi = s["ci95_cluster_bootstrap"]
        print(
            f"{name:16s} {100 * s['success_mean']:5.1f} +- {100 * s['success_std_over_seeds']:.1f}  "
            f"CI95 [{100 * lo:.1f}, {100 * hi:.1f}]  ({s['n_tasks']} tasks x {s['episodes_per_task_per_seed']} x {len(s['seeds'])} seeds)"
        )
    if suites:
        print(f"{'average':16s} {100 * report['average']:5.1f}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--policy", default="lerobot/smolvla_libero")
    r.add_argument("--revision", default="a24f1f380461b665158d52acd140152c79e8b36d")
    r.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    r.add_argument("--seeds", nargs="+", type=int, default=[1000, 2000, 3000])
    r.add_argument("--n-episodes", type=int, default=50)
    r.add_argument("--batch-size", type=int, default=10)
    r.add_argument("--resolution", type=int, default=360, help="LeRobot LiberoEnvConfig default")
    r.add_argument("--task-ids", nargs="*", type=int)
    r.add_argument("--env-type", default="libero", choices=["libero", "libero_plus"])
    r.add_argument("--server-endpoint", help="drive a Docker-served simulator (lerobot-env-server), e.g. tcp/127.0.0.1:7448")
    r.add_argument("--device", default="cuda")
    r.add_argument("--policy-arg", action="append", metavar="KEY=VALUE",
                   help="policy config override passed as --policy.KEY=VALUE (repeatable), e.g. object_token_ablation=zero")
    r.add_argument("--out", default="results/baselines/smolvla_libero")
    r.add_argument("--overwrite", action="store_true")
    a = sub.add_parser("aggregate")
    a.add_argument("--root", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else aggregate(args)


if __name__ == "__main__":
    main()
