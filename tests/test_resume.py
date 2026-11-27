"""Resuming SFT: data order, RNG, logs and the launcher flag."""

from __future__ import annotations

import argparse
import itertools

import scripts.benchmark as bench
from surgicalvla.training.sampler import ResumableRandomSampler
from surgicalvla.utils.logrun import truncate_csv


def test_sampler_resumes_exactly_where_it_stopped():
    full = list(itertools.islice(iter(ResumableRandomSampler(7, seed=3)), 30))
    resumed = ResumableRandomSampler(7, seed=3)
    resumed.set_position(17)
    assert list(itertools.islice(iter(resumed), 13)) == full[17:]


def test_sampler_epochs_are_permutations_with_different_orders():
    s = list(itertools.islice(iter(ResumableRandomSampler(50, seed=0)), 100))
    assert sorted(s[:50]) == sorted(s[50:]) == list(range(50))
    assert s[:50] != s[50:]


def test_csv_truncated_to_resume_step(tmp_path):
    p = tmp_path / "metrics.csv"
    p.write_text("step,key,value\n10,loss,1\n20,loss,0.9\n30,loss,0.8\n")
    truncate_csv(str(p), 20)
    assert p.read_text() == "step,key,value\n10,loss,1\n20,loss,0.9\n"


def test_rng_state_round_trip():
    import numpy as np
    import torch

    from surgicalvla.training.trainer import get_rng_state, set_rng_state

    st = get_rng_state()
    a = (torch.rand(3), np.random.rand(3))
    set_rng_state(st)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def _train(**kw):
    ns = argparse.Namespace(**{"stage": "sft", "benchmark": "libero", "policy": "surgicalvla", "config": None, "out": None, **kw})
    return " ".join(bench.train_cmds(ns, [])[0])


def test_launcher_resume_flag():
    assert "--resume latest" in _train(resume=True, out="results/libero_train")
    assert "--resume" not in _train()
    assert "run.resume=true" in _train(stage="rl", resume=True)


def test_val_fraction_is_of_the_episode_pool():
    from surgicalvla.utils.splits import split_episodes

    train, val = split_episodes(1693, 0.01, pool=list(range(432)))
    assert len(val) == 4 and len(train) == 428 and set(train) | set(val) == set(range(432))
    train, val = split_episodes(1693, 0.5, pool=[7, 8])
    assert len(train) == 1 and len(val) == 1


def test_train_loader_keeps_workers_and_follows_the_sampler():
    import torch
    from torch.utils.data import TensorDataset

    from surgicalvla.training.trainer import ResumableRandomSampler, make_train_loader

    ds = TensorDataset(torch.arange(12))
    loader = make_train_loader(ds, 4, ResumableRandomSampler(len(ds), 0), num_workers=2)
    assert loader.num_workers == 2 and loader.persistent_workers and loader.prefetch_factor == 4
    serial = make_train_loader(ds, 4, ResumableRandomSampler(len(ds), 0), num_workers=0)  # 0 workers must not raise
    from itertools import islice

    first = lambda dl: [b[0].tolist() for b in islice(dl, 5)]  # noqa: E731  (the sampler is endless)
    assert first(loader) == first(serial)
