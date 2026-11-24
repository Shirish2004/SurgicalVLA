"""Write the RoboTwin task subset and its protocol episodes (50 clean demonstrations per task).

    python3 scripts/robotwin_subset_episodes.py [--n 10] [--seed 0]

The subset is fixed by rule before any result: both handover tasks (closest to instrument
handover) plus ``n - 2`` tasks drawn with ``--seed`` from the other 48. The local
``robotwin_unified`` copy stores the 50 tasks as alphabetical blocks of 550 episodes whose first
50 are ``demo_clean``, so task ``k`` uses episodes ``550 k + j``, ``j < 50``.
Writes ``configs/data/robotwin_subset<n>_tasks.json`` and ``..._episodes.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark import ROBOTWIN_TASKS  # noqa: E402

BLOCK, CLEAN = 550, 50


def subset(n: int, seed: int) -> list[str]:
    tasks = sorted(ROBOTWIN_TASKS)
    handover = [t for t in tasks if t.startswith("handover")]
    rest = [t for t in tasks if t not in handover]
    return handover + sorted(random.Random(seed).sample(rest, n - len(handover)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tasks = sorted(ROBOTWIN_TASKS)
    chosen = subset(args.n, args.seed)
    episodes = [BLOCK * tasks.index(t) + j for t in chosen for j in range(CLEAN)]
    out = ROOT / "configs" / "data"
    (out / f"robotwin_subset{args.n}_tasks.json").write_text(json.dumps(chosen, indent=1))
    (out / f"robotwin_subset{args.n}_episodes.json").write_text(json.dumps(episodes))
    print(f"{len(chosen)} tasks, {len(episodes)} episodes: {chosen}")


if __name__ == "__main__":
    main()
