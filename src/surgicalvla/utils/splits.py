import json
import os
import random

from torch.utils.data import Subset


def block_episodes(total: int, block: int, take: int) -> list[int]:
    """First ``take`` episodes of every ``block``-sized block (RoboTwin: 50 demo_clean of 550)."""
    if not 0 < take <= block:
        raise ValueError(f"need 0 < take <= block, got take={take} block={block}")
    return [b * block + j for b in range(total // block) for j in range(take)]


def split_episodes(total: int, val_fraction: float, seed: int = 0, pool: list[int] | None = None):
    """Random train/val split of ``pool`` (default: every episode)."""
    ids = list(range(total)) if pool is None else list(pool)
    random.Random(seed).shuffle(ids)
    # the fraction is of the pool being split, and at least one episode stays in training
    n = min(max(1, int(round(len(ids) * val_fraction))), len(ids) - 1)
    return sorted(ids[n:]), sorted(ids[:n])


def save_split(path: str, train: list, val: list):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"train": train, "val": val}, f)


def load_split(path: str):
    with open(path) as f:
        d = json.load(f)
    return d["train"], d["val"]


def strided_subset(dataset, n: int):
    total = len(dataset)
    if n >= total:
        return dataset
    step = total / n
    return Subset(dataset, [int(i * step) for i in range(n)])