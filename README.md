# SurgicalVLA

A Vision-Language-Action model for precision manipulation (target: surgical
instrument handover), built on [LeRobot](https://github.com/huggingface/lerobot)'s
SmolVLA and trained in two stages: supervised flow matching, then online RL with
GRPO over the flow's SDE (ReinFlow-GRPO). Evaluated on **LIBERO** (+ LIBERO-Plus),
**RoboTwin 2.0** and **VLABench**, with one model per benchmark.

> **Status (2026-11-23).** The full architecture is built and passes an end-to-end check on
> real weights; **no ablation or benchmark result exists yet**, so nothing here is a claim.
> LIBERO evaluation follows a fixed protocol (LeRobot's canonical simulator image, 50-step
> execution horizon; the SmolVLA baseline scores ~66 % on LIBERO-Spatial under it, against ~90 %
> published, see [`docs/baselines.md`](docs/baselines.md)). All new components are
> **experimental** until their ablation passes ([`docs/limitations.md`](docs/limitations.md)).

## Architecture in one table

Details and measured costs: [`docs/architecture.md`](docs/architecture.md).

| Component | Switch | Supervision |
|---|---|---|
| SmolVLA backbone + flow-matching action expert | — | demonstrations; reproduces `smolvla_base` at init (rel. error < 2e-3) |
| Action-expert MoE | `moe_variant`: `upcycled` / `matched` (compute-matched) / `dense` / `dense_param_matched` | flow matching + load balancing |
| Auxiliary vision heads (training only) | `aux_vision` | label-free pseudo-labels from SAM 3, DINOv2, TAPNext++, Depth Anything V2, confidence-weighted |
| Latent chain of thought | `latent_cot` | future visual latent from an EMA target encoder |
| Explicit chain of thought + curriculum | `use_cot`, `aux.cot_traces`, `cot_curriculum_steps` | traces written from pseudo-labels, filtered by action-likelihood gain (bootstrapped) |
| Phase head | always | pseudo phase segments from gripper events and speed minima |
| ReinFlow-GRPO RL, joint SFT+RL | `configs/benchmarks/<b>/rl.yaml`, `training.mode` | task success (+ shaping) on LIBERO, RoboTwin 2.0, VLABench |

All auxiliary targets are generated offline by frozen models from raw demonstrations; simulator
ground truth is used only to validate them. The deployed policy runs no tracker.

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

# label-free pseudo-labels (+ tracking videos) and reasoning traces for one suite
pixi run -e dev labels --suite libero_spatial --limit 3
# full-architecture smoke test on real weights (params, VRAM, step time, latency per MoE variant)
pixi run -e dev smoke
# Step-4 ablation pilot on one LIBERO suite, 3 seeds, matched compute
pixi run -e dev pilot --suite libero_spatial --dry-run

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
CIs). Nothing is reported before the ablation pilots finish.

| Benchmark | SmolVLA (reference) | SurgicalVLA SFT | SurgicalVLA SFT + RL |
|---|---|---|---|
| LIBERO (4 suites) | protocol fixed; full run pending | — | — |
| LIBERO-Plus | — | — | — |
| RoboTwin 2.0 (Easy) | baseline training queued | — | — |
| VLABench (primitive / composite) | — | — | — |

## Documentation

- [`docs/audit.md`](docs/audit.md): Phase-0 audit, bugs found and fixed (each with a regression test)
- [`docs/baselines.md`](docs/baselines.md): evaluation protocols, versions, commands, reproduction status
- [`docs/architecture.md`](docs/architecture.md): components, switches, MoE variants with measured cost
- [`docs/training.md`](docs/training.md): per-benchmark pipeline, auxiliary supervision, pilot, RL stage, joint SFT + RL
- [`docs/pseudolabels.md`](docs/pseudolabels.md) / [`docs/pseudolabel_quality.md`](docs/pseudolabel_quality.md): label-free targets, cost, validation against simulator ground truth
- [`docs/related_work.md`](docs/related_work.md): related work and component licenses (all components release-safe)
- [`docs/limitations.md`](docs/limitations.md): experimental components and open issues
- [`docs/legacy_skill_routed_moe.md`](docs/legacy_skill_routed_moe.md): the earlier skill-routed MoE stack (`policy_type: surgical`) that produced the pre-audit `checkpoints*/`
- [`ARCH.md`](ARCH.md): target architecture sketch (includes experimental parts)

## Repository layout

```
src/surgicalvla/
  modules/   configuration_smolvla_moe.py, modeling_smolvla_moe.py   (the SurgicalVLA model)
             moe_layers.py, object_encoder.py                       (MoE FFN, ReinFlow head, object tokens)
             policy.py, moe.py, skills.py                           (legacy skill-routed stack)
             aux_heads.py, latent_cot.py                            (auxiliary vision heads, latent CoT)
  pseudolabels/  vision.py, tracks.py, phases.py, nounphrase.py, cot.py, sidecar.py, reader.py
  third_party/   tapnet (TAPNext++ PyTorch inference, Apache-2.0)
  rl/        benchmarks.py (benchmark contracts), lerobot_sim.py (rollout client),
             reinflow_grpo.py (GRPO math), joint.py (joint SFT+RL), rollout.py
  training/  trainer.py (SFT)
scripts/     benchmark.py (launcher: train / eval / pipeline / pilot / labels), train.py (SFT),
             train_reinflow_grpo.py (RL), make_pseudolabels.py, make_cot_traces.py,
             validate_pseudolabels.py, smoke_full_arch.py, eval_libero_baseline.py, lerobot_eval.py
configs/     benchmarks/<benchmark>/{sft,rl}.yaml
tests/       unit + regression tests (simulator tests run in pixi -e sim-eval)
```

## Citation

Paper in preparation; citation to follow.

## Licence

MIT.
