"""Paired comparison of two evaluated models on the same starting states.

    pixi run -e dev python scripts/compare_arms.py results/ablations/libero_spatial/smolvla \
        results/ablations/libero_spatial/vision [--out results/ablations/libero_spatial/vision_vs_smolvla.json]

Each argument is an evaluation root, or a directory of per-seed roots, holding
``episodes.jsonl`` (``eval_libero_baseline.py aggregate``). Episodes are matched on
(suite, eval seed, task, init state); only matched episodes count. Reports per suite:

* success of A and B, and the paired difference B − A;
* a 95 % cluster-bootstrap CI of the difference, resampling (task, init state) clusters with
  all their seeds and training seeds (episodes sharing a starting state are correlated);
* McNemar's exact test on the discordant pairs.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def load(root: Path) -> list[dict]:
    files = [root / "episodes.jsonl"] if (root / "episodes.jsonl").exists() else sorted(root.glob("*/episodes.jsonl"))
    if not files:
        raise FileNotFoundError(f"no episodes.jsonl under {root}")
    out = []
    for f in files:
        train_seed = f.parent.name if f.parent != root else ""
        for line in f.read_text().splitlines():
            r = json.loads(line)
            r["train_seed"] = train_seed
            out.append(r)
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def compare(a: list[dict], b: list[dict], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired statistics per suite; training seeds of each arm are matched in sorted order."""
    def key(r, ts):
        return (r["suite"], ts, r["seed"], r["task"], r["init_state"])

    seeds_a = sorted({r["train_seed"] for r in a})
    seeds_b = sorted({r["train_seed"] for r in b})
    pair = dict(zip(seeds_a, range(len(seeds_a)), strict=False))
    pair_b = dict(zip(seeds_b, range(len(seeds_b)), strict=False))
    ia = {key(r, pair.get(r["train_seed"])): r["success"] for r in a}
    ib = {key(r, pair_b.get(r["train_seed"])): r["success"] for r in b}
    report = {}
    for suite in sorted({k[0] for k in ia} & {k[0] for k in ib}):
        keys = sorted(k for k in ia if k[0] == suite and k in ib and k[1] is not None)
        if not keys:
            continue
        xa = np.array([ia[k] for k in keys], float)
        xb = np.array([ib[k] for k in keys], float)
        clusters: dict[tuple, list[int]] = {}
        for i, k in enumerate(keys):
            clusters.setdefault((k[3], k[4]), []).append(i)
        groups = list(clusters.values())
        rng = np.random.default_rng(seed)
        diffs = np.empty(n_boot)
        for t in range(n_boot):
            idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            diffs[t] = xb[idx].mean() - xa[idx].mean()
        disc_b = int(((xb == 1) & (xa == 0)).sum())
        disc_a = int(((xa == 1) & (xb == 0)).sum())
        report[suite] = {"episodes": len(keys), "success_a": float(xa.mean()), "success_b": float(xb.mean()),
                         "diff_b_minus_a": float(xb.mean() - xa.mean()),
                         "ci95_cluster_bootstrap": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
                         "b_only_success": disc_b, "a_only_success": disc_a, "mcnemar_p": mcnemar_exact(disc_b, disc_a)}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out")
    args = ap.parse_args()
    report = compare(load(Path(args.a)), load(Path(args.b)))
    for suite, r in report.items():
        lo, hi = r["ci95_cluster_bootstrap"]
        print(f"{suite:16s} A {100 * r['success_a']:5.1f}  B {100 * r['success_b']:5.1f}  B-A {100 * r['diff_b_minus_a']:+5.1f} "
              f"[{100 * lo:+.1f}, {100 * hi:+.1f}]  McNemar p={r['mcnemar_p']:.3g}  (n={r['episodes']})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
