"""scripts/benchmark.py builds the right command for every benchmark x stage."""

from __future__ import annotations

import argparse

import pytest

import scripts.benchmark as bench

BENCHES = ["libero", "robotwin", "vlabench"]


def _train(**kw):
    ns = argparse.Namespace(stage="sft", benchmark="libero", policy="surgicalvla", config=None, out=None, dry_run=True)
    vars(ns).update(kw)
    return [" ".join(c) for c in bench.train_cmds(ns, [])]


def _eval(tmp_path, **kw):
    ns = argparse.Namespace(
        benchmark="libero", checkpoint="lerobot/smolvla_libero", revision="abc", name=None, out=str(tmp_path),
        seeds=[1000], suites=None, tasks=None, local=False, dry_run=True,
    )
    vars(ns).update(kw)
    return [" ".join(c) for c in bench.eval_cmds(ns, [])]


@pytest.mark.parametrize("b", BENCHES)
def test_each_benchmark_has_its_own_sft_model_and_config(b):
    cmd = _train(benchmark=b)[0]
    assert f"configs/benchmarks/{b}/sft.yaml" in cmd and bench.DATASET[b] in cmd


@pytest.mark.parametrize("b", BENCHES)
def test_rl_runs_for_every_benchmark(b):
    cmds = _train(stage="rl", benchmark=b)
    rl = [c for c in cmds if "train_reinflow_grpo.py" in c]
    assert rl and f"configs/benchmarks/{b}/rl.yaml" in rl[0] and "-e sim-eval" in rl[0]
    served = any("lerobot-env-server" in c for c in cmds)
    assert served == (b != "libero"), "RoboTwin / VLABench rollouts run in LeRobot's Docker images"
    if served:
        assert "--stop=true" in cmds[-1]


def test_smolvla_baselines_use_the_released_recipes():
    lib = _train(policy="smolvla")[0]
    assert "--policy.train_expert_only=false" in lib and "--steps=25000" in lib and "--batch_size=32" in lib
    assert "lerobot/vlabench_unified" in _train(policy="smolvla", benchmark="vlabench")[0]
    assert "train_robotwin_baseline.sh" in _train(policy="smolvla", benchmark="robotwin")[0]


def test_libero_eval_runs_every_suite_in_the_canonical_image_then_aggregates(tmp_path):
    cmds = _eval(tmp_path)
    runs = [c for c in cmds if "eval_libero_baseline.py run" in c]
    assert len(runs) == 4 and all("--env-type libero" in c and "--server-endpoint" in c for c in runs)
    assert "aggregate" in cmds[-1] and sum("lerobot-env-server" in c and "--stop" not in c for c in cmds) == 4
    local = [c for c in _eval(tmp_path, local=True) if "eval_libero_baseline.py run" in c]
    assert local and not any("--server-endpoint" in c for c in local)


def test_libero_plus_uses_server_and_one_episode_per_variant(tmp_path):
    cmds = _eval(tmp_path, benchmark="libero_plus", suites=["libero_spatial"])
    assert "lerobot-env-server" in cmds[0] and "--stop=true" in cmds[2]
    assert "--n-episodes 1" in cmds[1] and "--server-endpoint" in cmds[1]
    import yaml

    cfg = yaml.safe_load(next(tmp_path.glob("server_libero_plus_*.yaml")).read_text())
    assert cfg["sim"]["task"] == "libero_spatial"
    assert "task_ids" not in cfg["sim"]["kwargs"]["gym_kwargs"], "full protocol evaluates every variant"


def test_robotwin_eval_is_100_episodes_per_task(tmp_path):
    cmds = _eval(tmp_path, benchmark="robotwin", checkpoint="outputs/none", revision=None, tasks=["beat_block_hammer"])
    ev = [c for c in cmds if "lerobot_eval.py" in c]
    assert ev and all("--eval.n_episodes=100" in c and "--env.type=robotwin" in c for c in ev)
    assert len(bench.ROBOTWIN_TASKS) == 50


def test_vlabench_eval_covers_both_suites_at_10_episodes(tmp_path):
    cmds = _eval(tmp_path, benchmark="vlabench", checkpoint="lerobot/smolvla_vlabench", seeds=[1000, 2000, 3000])
    ev = [c for c in cmds if "lerobot_eval.py" in c]
    assert len(ev) == 6, "2 suites x 3 seeds"
    assert all("--eval.n_episodes=10" in c and "--env.type=vlabench" in c for c in ev)
    assert len(bench.VLABENCH_PRIMITIVE) == 21 and len(bench.VLABENCH_COMPOSITE) == 22
    assert sum("lerobot-env-server" in c and "--stop=true" not in c for c in cmds) == 2


def test_pipeline_trains_every_benchmark_before_any_eval(capsys):
    args = argparse.Namespace(benchmarks=BENCHES, libero_plus=True, skip_sft=False, skip_rl=False, dry_run=True)
    bench.pipeline(args, [])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("$")]
    train = [i for i, line in enumerate(lines) if "scripts/train.py" in line or "train_reinflow_grpo.py" in line]
    evals = [i for i, line in enumerate(lines) if "lerobot_eval.py" in line or "eval_libero_baseline.py run" in line]
    assert len(train) == 6 and evals and max(train) < min(evals)
    assert any("--env-type libero_plus" in line for line in lines)


def test_every_arm_sets_every_switch_and_shares_the_protocol():
    spec = bench.load_arms()
    outs = {}
    for arm in spec["arms"]:
        out, over = bench.arm_overrides(spec, arm, "libero_spatial", 0)
        keys = {o.split("=", 1)[0] for o in over}
        assert set(spec["switches"]) <= keys
        shared = sorted(o for o in over if o.split("=", 1)[0] in {"run.steps", "run.batch_size", "run.grad_accum", "dataset.episodes"})
        outs[arm] = shared
    assert len({tuple(v) for v in outs.values()}) == 1  # identical steps, batch and data for every arm


def test_each_comparison_differs_only_in_the_factor_under_test():
    spec = bench.load_arms()
    expected = {"aux_only": {"smolvla_moe.aux_vision"}, "objtok_only": {"smolvla_moe.object_input"},
                "frozen_control": {"smolvla_moe.object_input"}, "full": {"smolvla_moe.moe_variant"},
                "full_moe_matched": {"smolvla_moe.moe_variant"}, "cot_latent": {"smolvla_moe.latent_cot"}}
    for arm, factor in expected.items():
        a, b = spec["arms"][arm]["set"], spec["arms"][spec["arms"][arm]["compare_to"]]["set"]
        assert {k for k in a if a[k] != b[k]} == factor, arm


def test_final_architecture_is_the_sft_default():
    import yaml

    spec = bench.load_arms()
    for b in ("libero", "robotwin", "vlabench"):
        cfg = yaml.safe_load(open(f"configs/benchmarks/{b}/sft.yaml"))
        for key, want in spec["arms"]["full"]["set"].items():
            section, name = key.split(".")
            if key == "smolvla_moe.cot_curriculum_steps":
                assert cfg[section][name] > 0
            else:
                assert cfg[section][name] == want, (b, key)


def test_ablate_dry_run_trains_then_evaluates(capsys):
    bench.ablate(argparse.Namespace(suite="libero_object", arms=None, seeds=[0], steps=None, dry_run=True), [])
    out = capsys.readouterr().out
    assert "outputs/ablations/libero_object/smolvla/seed0" in out and "outputs/ablations/libero_object/full/seed0" in out
    assert "smolvla_moe.object_input=none" in out and "smolvla_moe.object_input=objects" in out
