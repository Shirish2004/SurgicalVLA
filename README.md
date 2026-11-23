# SurgicalVLA

A Vision-Language-Action model for precision manipulation (target: surgical
instrument handover), built on [LeRobot](https://github.com/huggingface/lerobot)'s
SmolVLA and trained in two stages: supervised flow matching, then online RL with
GRPO over the flow's SDE (ReinFlow-GRPO). Evaluated on **LIBERO** (+ LIBERO-Plus),
**RoboTwin 2.0** and **VLABench**, with one model per benchmark.

> **Status (2026-11-23).** Research code, no results yet. The LIBERO baseline is
> **not reproduced** yet (66 % vs ~90 % published on LIBERO-Spatial with LeRobot's
> canonical simulator stack; see [`docs/baselines.md`](docs/baselines.md)), so no
> number in this repository is a claim. Several model branches are
> **experimental** because they have no training signal yet; see
> [`docs/limitations.md`](docs/limitations.md).

## What trains today, and what is experimental

| Component | Status |
|---|---|
| SmolVLA backbone (SmolVLM2-500M, 16 layers) + flow-matching action expert | trained; up-cycled model reproduces `smolvla_base` at init (rel. error < 2e-3, tested on the real weights) |
| Mixture-of-experts action FFNs (1 shared + top-2 of 4 routed, Switch load balancing) | trained. **Caveat:** 3× the dense FFN FLOPs; compute-matched ablation pending |
| ReinFlow noise head + GRPO (per-step ratios, k3 KL to the SFT reference, executed-step log-probs) | implemented and unit-tested; not yet run to convergence |
| Joint / interleaved SFT + RL (`training.mode`) | implemented and unit-tested; pilot pending |
| Chain-of-thought, phase-conditioned routing, object-centric 6-D tokens, pose auxiliary loss | **experimental**: no labels in any dataset, so no supervision yet |

## Install (pixi)

```bash
pixi install -e dev        # training + tests (Python 3.13, torch, LeRobot, editable)
pixi install -e sim-eval   # + LIBERO and the client for Docker-served simulators
pixi run -e dev test       # test suite
```

`lerobot/` is a vendored, unmodified LeRobot checkout (upstream `8ec1a474d`, git-ignored).
RoboTwin, VLABench and LIBERO-Plus run in LeRobot's canonical Docker images
(`lerobot-env-server --docker=true`); the launcher builds/starts/stops them. Docker with
the NVIDIA runtime is required for those.

## Quickstart

```bash
# one benchmark: SFT, then RL from that SFT model
python3 scripts/benchmark.py train sft --benchmark libero      # libero | robotwin | vlabench
python3 scripts/benchmark.py train rl  --benchmark libero

# everything: SFT -> RL on every benchmark, then evaluate each model on its benchmark
python3 scripts/benchmark.py pipeline --libero-plus
python3 scripts/benchmark.py pipeline --dry-run                 # just print the commands

# evaluate any checkpoint (Hub id at a pinned revision, or a local training dir)
python3 scripts/benchmark.py eval --benchmark libero \
    --checkpoint lerobot/smolvla_libero --revision a24f1f380461b665158d52acd140152c79e8b36d

# unmodified SmolVLA baselines with the released recipes
python3 scripts/benchmark.py train sft --benchmark robotwin --policy smolvla
```

Per-benchmark configs live in `configs/benchmarks/<benchmark>/{sft,rl}.yaml` and write to
`outputs/<benchmark>/{sft,rl}`. See [`docs/training.md`](docs/training.md).

## Results

Each benchmark has its own model; numbers will be reported per benchmark with the
protocols in [`docs/baselines.md`](docs/baselines.md) (multiple seeds, cluster-bootstrap
CIs). Nothing is reported until the baseline protocol is reproduced.

| Benchmark | SmolVLA (reference) | SurgicalVLA SFT | SurgicalVLA SFT + RL |
|---|---|---|---|
| LIBERO (4 suites) | protocol gap under investigation | — | — |
| LIBERO-Plus | — | — | — |
| RoboTwin 2.0 (Easy) | baseline training queued | — | — |
| VLABench (primitive / composite) | — | — | — |

## Documentation

- [`docs/audit.md`](docs/audit.md): Phase-0 audit, bugs found and fixed (each with a regression test)
- [`docs/baselines.md`](docs/baselines.md): evaluation protocols, versions, commands, reproduction status
- [`docs/training.md`](docs/training.md): per-benchmark pipeline, RL stage, joint SFT + RL
- [`docs/limitations.md`](docs/limitations.md): experimental components and open issues
- [`docs/legacy_skill_routed_moe.md`](docs/legacy_skill_routed_moe.md): the earlier skill-routed MoE stack (`policy_type: surgical`) that produced the pre-audit `checkpoints*/`
- [`ARCH.md`](ARCH.md): target architecture sketch (includes experimental parts)

## Repository layout

```
src/surgicalvla/
  modules/   configuration_smolvla_moe.py, modeling_smolvla_moe.py   (the SurgicalVLA model)
             moe_layers.py, object_encoder.py                       (MoE FFN, ReinFlow head, object tokens)
             policy.py, moe.py, skills.py                           (legacy skill-routed stack)
  rl/        benchmarks.py (benchmark contracts), lerobot_sim.py (rollout client),
             reinflow_grpo.py (GRPO math), joint.py (joint SFT+RL), rollout.py
  training/  trainer.py (SFT)
scripts/     benchmark.py (launcher), train.py (SFT), train_reinflow_grpo.py (RL),
             eval_libero_baseline.py, lerobot_eval.py, train_robotwin_baseline.sh
configs/     benchmarks/<benchmark>/{sft,rl}.yaml
tests/       unit + regression tests (simulator tests run in pixi -e sim-eval)
```

## Citation

Paper in preparation; citation to follow.

## Licence

MIT.
