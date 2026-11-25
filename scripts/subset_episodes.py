"""Write a fixed per-task episode subset of a local LeRobot dataset (feature-cache episodes).

    python3 scripts/subset_episodes.py --benchmark vlabench      # 2 episodes per instruction
    python3 scripts/subset_episodes.py --benchmark libero_plus   # 12 episodes per task

The subset is fixed by rule before any result: for every task string in the dataset's episode
metadata, ``--per-task`` episodes drawn with ``--seed`` (all of them if the task has fewer).
VLABench stores one task string per instruction (295, prefixed ``primitive:`` or
``composite:``); LIBERO-plus stores the 40 LIBERO tasks, each with its perturbed variants.
Writes ``configs/data/<benchmark>_subset_episodes.json`` (sorted episode indices).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PER_TASK = {"vlabench": 2, "libero_plus": 12}


def episode_tasks(root: Path) -> dict[int, str]:
    """Episode index -> its (first) task string, from ``meta/episodes``."""
    import pyarrow.parquet as pq

    out = {}
    for f in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "tasks"]).to_pydict()
        out.update({int(e): str(ts[0]) for e, ts in zip(t["episode_index"], t["tasks"])})
    return out


def subset(tasks: dict[int, str], per_task: int, seed: int) -> list[int]:
    by_task = defaultdict(list)
    for ep, task in sorted(tasks.items()):
        by_task[task].append(ep)
    rng = random.Random(seed)
    chosen = [e for task in sorted(by_task) for e in rng.sample(by_task[task], min(per_task, len(by_task[task])))]
    return sorted(chosen)


def main() -> None:
    from scripts.benchmark import CACHE

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(PER_TASK))
    ap.add_argument("--per-task", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    per_task = args.per_task or PER_TASK[args.benchmark]
    tasks = episode_tasks(Path(CACHE[args.benchmark]["root"]))
    episodes = subset(tasks, per_task, args.seed)
    path = ROOT / "configs" / "data" / f"{args.benchmark}_subset_episodes.json"
    path.write_text(json.dumps(episodes))
    print(f"{len(set(tasks.values()))} tasks x {per_task} -> {len(episodes)} episodes: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
