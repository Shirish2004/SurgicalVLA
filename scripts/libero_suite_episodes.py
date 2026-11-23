"""Write the episode indices of one LIBERO suite in a LeRobot LIBERO dataset (for suite-level SFT).

    pixi run -e sim-eval python scripts/libero_suite_episodes.py --suite libero_spatial

Episodes are matched by instruction: each suite's task languages come from the LIBERO
benchmark registry, each episode's from the dataset metadata. Writes
``configs/data/<dataset>_<suite>_episodes.json``, usable as ``dataset.episodes``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def norm(text: str) -> str:
    return " ".join(text.lower().replace(".", " ").split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--repo-id", default="lerobot/libero")
    ap.add_argument("--out-dir", default="configs/data")
    args = ap.parse_args()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from libero.libero import benchmark

    languages = {norm(t.language) for t in benchmark.get_benchmark_dict()[args.suite]().tasks}
    meta = LeRobotDatasetMetadata(args.repo_id)
    task_text = {int(row.task_index): norm(str(text)) for text, row in meta.tasks.iterrows()}
    # episode -> task via the per-frame task_index (frame tables are small; images live in videos)
    from datasets import load_dataset

    frames = load_dataset(args.repo_id, data_files="data/*/*.parquet", split="train", columns=["episode_index", "task_index"])
    ep_task = dict(zip(frames["episode_index"], frames["task_index"], strict=True))
    episodes = sorted(int(e) for e, t in ep_task.items() if task_text[int(t)] in languages)
    out = Path(args.out_dir) / f"{args.repo_id.split('/')[-1]}_{args.suite}_episodes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(episodes))
    print(f"{len(episodes)} episodes of {args.suite} ({len(languages)} tasks) -> {out}")


if __name__ == "__main__":
    main()
