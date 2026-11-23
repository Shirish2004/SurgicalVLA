# Architecture

SurgicalVLA is SmolVLA (SmolVLM2-500M truncated to 16 layers + a flow-matching action expert)
with switchable additions. Every addition is a config flag in `smolvla_moe` (per-benchmark SFT
configs, `configs/benchmarks/<benchmark>/sft.yaml`) and stays **experimental** until its
ablation passes (`docs/limitations.md`). With all flags off the model reproduces
`lerobot/smolvla_base` at initialisation (rel. action error < 2e-3, tested on the real weights).

```
images ─ SigLIP + connector ─┐
instruction tokens ──────────┤ bidirectional block
(object tokens, optional) ───┘
state ───────────────────────── causal
explicit CoT text (optional) ── causal, decoded by the LM head at inference every N chunks
latent CoT queries (optional) ─ causal, K learned tokens, no decoding
        │  SmolVLM2 layers 0..15 → prefix KV cache
        ▼
noisy action chunk + flow time ─ action expert (16 layers, cross-attends to the prefix)
                                  FFN = dense | MoE (variant below)
                                  └─ velocity → Euler (SFT / eval) or SDE + noise head (RL)

training-only heads on the pooled prefix: auxiliary vision targets, latent-CoT predictor, phase
```

## Components

| Flag | What it adds | Supervision | Inference cost |
|---|---|---|---|
| `moe_variant` | action-expert FFN: `upcycled`, `matched`, `dense`, `dense_param_matched` | flow matching + Switch load balancing | see table below |
| `aux_vision` | heads predicting the target's DINOv2 feature, motion, point-track displacement and gripper offset from the pooled VLM prefix | label-free pseudo-labels (`docs/pseudolabels.md`), weighted by their confidence | none (training only) |
| `latent_cot` | `latent_cot_tokens` learned queries appended to the prefix; their mean output predicts the embedding of the frame `latent_cot_horizon` ahead | EMA target encoder (the vision encoder itself when frozen), stop-gradient | K prefix tokens |
| `use_cot` + `aux.cot_traces` | explicit reasoning text in the prefix, CE-trained, decoded at inference | traces written from pseudo-labels, filtered by action-likelihood gain (`scripts/make_cot_traces.py`) | up to `cot_max_tokens` VLM passes every `cot_refresh_interval` chunks |
| `cot_curriculum_steps` | probability of feeding explicit text decays 1 → 0; latent tokens remain | — | as `latent_cot` |
| phase head | phase from the CoT phase token, else from pooled perception | pseudo phase segments (gripper events + speed minima) | none |
| object tokens / pose head | object-centric 6-D tokens and pose auxiliary loss | **none yet** (no labels; off by default) | — |
| ReinFlow noise head | learned per-step σ for SDE rollouts and exact transition log-probs | RL only | none at evaluation (ODE) |

Design constraint: every auxiliary target is produced offline by frozen models from raw
demonstrations; simulator ground truth is used only to validate pseudo-labels and for oracle
ablations. The deployed policy runs no tracker.

## MoE variants (measured, `scripts/smoke_full_arch.py`)

All components on, warm-started from `lerobot/smolvla_libero`, batch 2, RTX A4500.

| Variant | Experts | Active FFN compute | Params | Train step | Peak VRAM | Chunk latency |
|---|---|---|---|---|---|---|
| `upcycled` | 1 shared + top-2 of 4, full width | 3 × dense | 893 M | 0.25 s | 7.3 GiB | 252 ms |
| `matched` | width dense/3, random row-slices | 1 × dense | 657 M | 0.26 s | 5.6 GiB | 258 ms |
| `dense` | none | 1 × dense | 609 M | 0.21 s | 5.5 GiB | 122 ms |
| `dense_param_matched` | one FFN widened to the MoE's total params | 5 × dense | 892 M | 0.22 s | 7.9 GiB | 124 ms |

`matched` vs `dense` is the compute-matched comparison; `upcycled` vs `dense_param_matched` the
parameter-matched one. The MoE latency overhead at equal FLOPs comes from the per-expert
dispatch loop, not arithmetic; grouped expert matmuls would remove it.

## Training data flow

`scripts/train.py` → `Trainer._prep`: processor → (latent CoT) split the future frame off the
observation history → keep the current frame → attach `aux.*` targets, pseudo phase ids and
tokenised traces from the sidecar store (`pseudolabels.reader.SidecarTargets`) by
`(episode_index, frame_index)`. After each optimiser step the latent-CoT EMA encoder advances.
