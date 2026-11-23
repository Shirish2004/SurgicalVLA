"""scripts/benchmark.py builds the right command for every benchmark x stage."""

from __future__ import annotations

import argparse

import pytest

import scripts.benchmark as bench


def _train(**kw):
    ns = argparse.Namespace(stage="sft", benchmark="libero", policy="surgicalvla", config=None, out=None, dry_run=True)
    vars(ns).update(kw)
    return [" ".join(c) for c in bench.train_cmds(ns, [])]


def _eval(tmp_path, **kw):
    ns = argparse.Namespace(
        benchmark="libero", checkpoint="lerobot/smolvla_libero", revision="abc", name=None, out=str(tmp_path),
        seeds=[1000], suites=None, tasks=None, server=False, dry_run=True,
    )
    vars(ns).update(kw)
    return [" ".join(c) for c in bench.eval_cmds(ns, [])]


def test_train_routes():
    assert "lerobot-train" in _train(policy="smolvla")[0] and "lerobot/libero" in _train(policy="smolvla")[0]
    assert "train_robotwin_baseline.sh" in _train(policy="smolvla", benchmark="robotwin")[0]
    assert "configs/train_libero.yaml" in _train()[0]
    assert "configs/train.yaml" in _train(benchmark="robotwin")[0]
    assert "train_reinflow_grpo.py" in _train(stage="rl")[0]
    # the LIBERO SmolVLA baseline must use smolvla_libero's recipe: full fine-tune
    cmd = _train(policy="smolvla")[0]
    assert "--policy.train_expert_only=false" in cmd and "--steps=25000" in cmd and "--batch_size=32" in cmd


def test_robotwin_rl_is_refused():
    with pytest.raises(SystemExit, match="LIBERO only"):
        _train(stage="rl", benchmark="robotwin")


def test_libero_eval_runs_every_suite_then_aggregates(tmp_path):
    cmds = _eval(tmp_path)
    runs = [c for c in cmds if "eval_libero_baseline.py run" in c]
    assert len(runs) == 4 and all("--env-type libero" in c for c in runs)
    assert "aggregate" in cmds[-1]
    assert not any("lerobot-env-server" in c for c in cmds)


def test_libero_plus_uses_server_and_one_episode_per_variant(tmp_path):
    cmds = _eval(tmp_path, benchmark="libero_plus", suites=["libero_spatial"])
    assert "lerobot-env-server" in cmds[0] and "--stop=true" in cmds[2]
    assert "--n-episodes 1" in cmds[1] and "--server-endpoint" in cmds[1]
    import yaml

    cfg = yaml.safe_load(next(tmp_path.glob("server_libero_plus_*.yaml")).read_text())
    assert cfg["sim"]["task"] == "libero_spatial"
    assert "task_ids" not in cfg["sim"]["kwargs"]["gym_kwargs"], "full protocol evaluates every variant"


def test_robotwin_eval_is_100_episodes_per_task(tmp_path):
    cmds = _eval(tmp_path, benchmark="robotwin", checkpoint="outputs/x", revision=None, tasks=["beat_block_hammer"])
    ev = [c for c in cmds if "lerobot-eval" in c]
    assert ev and all("--eval.n_episodes=100" in c and "--env.type=robotwin" in c for c in ev)
    assert len(bench.ROBOTWIN_TASKS) == 50
