"""Collect every evaluated model into one results file (``results/results.json``).

    pixi run -e dev python scripts/collect_results.py

Sources:

* ``results/ablations/<suite>/<arm>/seed<k>/`` (our arms; ``episodes.jsonl`` per training seed);
* ``results/baselines/<name>/`` (official checkpoints evaluated in our harness);
* ``configs/reporting/published.yaml`` (other VLAs, as reported; reference only).

For our arms, the success rate pools every training seed, evaluation seed, task and init
state. The 95 % CI is a cluster bootstrap over (task, init state) clusters, each carrying all
of its seeds. Every entry is checked against ``configs/reporting/schema.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "configs/reporting/schema.json").read_text())
TYPES = {"string": str, "number": (int, float), "integer": int, "array": list, "object": dict, "null": type(None)}


def validate(entry: dict) -> None:
    """Required keys, enums and JSON types of ``configs/reporting/schema.json``."""
    for key in SCHEMA["required"]:
        if key not in entry:
            raise ValueError(f"result entry misses {key!r}: {entry}")
    for key, spec in SCHEMA["properties"].items():
        if key not in entry:
            continue
        v = entry[key]
        if "enum" in spec and v not in spec["enum"]:
            raise ValueError(f"{key}={v!r} not in {spec['enum']}")
        if "type" in spec:
            types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
            allowed: list[type] = []
            for name in types:
                t = TYPES[name]
                allowed += list(t) if isinstance(t, tuple) else [t]
            if not isinstance(v, tuple(allowed)):
                raise ValueError(f"{key}={v!r} is not {types}")


def cluster_ci(records: list[dict], n_boot: int = 10_000, seed: int = 0) -> tuple[float, list[float]]:
    clusters: dict[tuple, list[float]] = {}
    for r in records:
        clusters.setdefault((r["task"], r["init_state"]), []).append(float(r["success"]))
    groups = [np.array(v) for v in clusters.values()]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), (n_boot, len(groups)))
    boots = sums[idx].sum(1) / counts[idx].sum(1)
    return float(sums.sum() / counts.sum()), [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def ours(root: Path, arms_spec: dict) -> list[dict]:
    out = []
    for arm_dir in sorted(p for p in (root / "results/ablations").glob("*/*") if p.is_dir()):
        suite, arm = arm_dir.parent.name, arm_dir.name
        seeds = sorted(arm_dir.glob("seed*/episodes.jsonl"))
        if not seeds:
            continue
        per_seed, records = [], []
        for f in seeds:
            rows = [json.loads(line) for line in f.read_text().splitlines()]
            rows = [r for r in rows if r["suite"] == suite]
            per_seed.append(float(np.mean([r["success"] for r in rows])) if rows else None)
            records += rows
        success, ci = cluster_ci(records)
        proto = arms_spec["protocol"]
        out.append({"arm": arm, "benchmark": "libero", "suite": suite, "source": "ours",
                    "model": f"outputs/ablations/{suite}/{arm}", "success": success, "ci95": ci, "per_seed": per_seed,
                    "training": {"data": f"lerobot/libero, {suite} episodes", "steps": proto["steps"],
                                 "batch": proto["batch_size"] * proto["grad_accum"], "seeds": [f.parent.name for f in seeds]},
                    "eval": {"protocol": "canonical LIBERO image, 50 init states x 10 tasks, 50-step horizon",
                             "eval_seeds": proto["eval_seeds"], "episodes": len(records)},
                    "citation": None, "date": None,
                    "notes": arms_spec["arms"].get(arm, {}).get("what", "")})
    return out


def official(root: Path) -> list[dict]:
    out = []
    for summary in sorted((root / "results/baselines").glob("*/summary.json")):
        rep = json.loads(summary.read_text())
        for suite, s in rep.get("suites", {}).items():
            out.append({"arm": summary.parent.name, "benchmark": "libero", "suite": suite, "source": "official_checkpoint",
                        "model": summary.parent.name, "success": s["success_mean"], "ci95": s["ci95_cluster_bootstrap"],
                        "per_seed": s.get("per_seed", []),
                        "eval": {"protocol": "canonical LIBERO image, 50-step horizon",
                                 "eval_seeds": s.get("seeds", []), "episodes": s["n_tasks"] * s["episodes_per_task_per_seed"] * len(s["seeds"])},
                        "citation": None, "date": None,
                        "notes": f"{s['n_tasks']} tasks x {s['episodes_per_task_per_seed']} episodes x {len(s['seeds'])} seeds"})
    return out


def published(root: Path) -> list[dict]:
    spec = yaml.safe_load((root / "configs/reporting/published.yaml").read_text())
    out = []
    for bench, rows in spec.items():
        for r in rows:
            for suite, val in r["success"].items():
                out.append({"arm": r["model"], "benchmark": bench, "suite": suite, "source": "published", "model": r["model"],
                            "success": val / 100.0, "ci95": None, "citation": r["citation"], "date": str(r["date"]),
                            "eval": {"protocol": r["protocol"]}, "notes": r["caveats"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="results/results.json")
    args = ap.parse_args()
    root = Path(args.root)
    arms_spec = yaml.safe_load((root / "configs/ablations/arms.yaml").read_text())
    entries = ours(root, arms_spec) + official(root) + published(root)
    for e in entries:
        validate(e)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2))
    print(f"{len(entries)} entries -> {out} ({sum(e['source'] == 'ours' for e in entries)} ours)")


if __name__ == "__main__":
    main()
