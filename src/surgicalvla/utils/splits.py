import json
import os
import random

from torch.utils.data import Subset


def split_episodes(total: int, val_fraction: float, seed: int = 0):
    ids = list(range(total))
    random.Random(seed).shuffle(ids)
    n = max(1, int(round(total * val_fraction)))
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