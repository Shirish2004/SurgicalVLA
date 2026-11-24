# VLM changes: ranked shortlist

The VLM (SmolVLM2, 16 of 32 decoder layers + SigLIP, 350 M parameters) is frozen in every
recipe so far. Each change below is a config flag with a matched ablation arm
(`configs/ablations/arms.yaml`), a forgetting check, and a measured cost.

## Evidence that motivates a change

1. **Auxiliary losses cannot reach visual features through a frozen VLM.** Measured parameter
   counts: with `train_expert_only`, only the action expert, the heads, the object projector,
   the state projection and the latent queries train. Losses read from the prefix reach about
   10 M parameters. The `aux_source: expert` workaround routes them through the expert instead
   (`docs/results_analysis.md`).
2. **The released LIBERO checkpoint fine-tuned the whole VLM.** `lerobot/smolvla_libero` was
   trained with `train_expert_only: false` and `freeze_vision_encoder: false`. That
   checkpoint's published score (90 % Spatial) therefore comes from a recipe that adapts the
   VLM.
3. **Simulator rendering differs from the VLM's pretraining images.** SAM 3 text scores for
   instruction colours are 0.2–0.5, and LIBERO's "black bowl" is rendered silver
   (`docs/pseudolabels.md`). That is a visual domain gap a frozen encoder cannot adapt to.

## Shortlist (cheapest first)

| Rank | Change | Flag | Params | Measured cost (batch 2, A4500) | Arm | Risk |
|---|---|---|---|---|---|---|
| **1** | LoRA r = 16 on the 16 decoder layers (q, k, v, o, gate, up, down) | `vlm_lora_rank: 16`, `vlm_lora_parts: [text]` | 4.3 M (1.2 % of the VLM) | step 0.22 s (unchanged), peak 2.85 GiB (unchanged) | `vision_lora`, `full_lora` | low: identical to the frozen model at step 0 |
| 2 | 1 + LoRA on the SigLIP vision encoder | `vlm_lora_parts: [text, vision]` | 9.7 M | step 0.42 s (**2×**), peak 4.6 GiB (+1.7 GiB) | `vision_lora_vis` | visual-feature drift; doubles training time |
| 3 | Unfreeze the vision connector only (low LR) | not implemented | ≈ 12 M | not measured | — | small |
| 4 | Full VLM fine-tune at low LR (the released recipe) | `model.train_expert_only: false`, `model.freeze_vision_encoder: false` | 350 M (> 10 % of the model) | earlier smoke with the VLM trainable: step 0.21 s, peak 5.5 GiB at batch 2; at batch 24 expect about 2× the frozen run's memory | not in the ladder | largest forgetting risk; most compute |

Chosen path (2026-11-24): LoRA, starting with option 1 and its matched arms. Option 2 runs as
one arm (`vision_lora_vis`), because it doubles step time. Option 4 is deferred until options 1
and 2 have been measured.

## Forgetting checks

- **Feature drift.** Every validation logs `val/vlm_feature_cos`, the cosine between the pooled
  VLM prefix with and without the adapters on the same batch (`policy.vlm_feature_drift`).
  After 3 steps at LR 1e-4 it was 1.000 (decoder only) and 0.994 (decoder + vision).
- **Pseudo-label agreement.** Explicit CoT traces are decoded from the adapted VLM; their
  token accuracy (`cot_token_acc`) is logged.
- **Cross-suite success.** Evaluate each suite's LoRA model on the other three suites, together
  with the frozen-VLM model trained on the same suite.

## Matched ablations

| Arm | Compared with | Differs only in |
|---|---|---|
| `vision_lora` | `vision` | `vlm_lora_rank` 0 → 16 |
| `vision_lora_vis` | `vision_lora` | `vlm_lora_parts` [text] → [text, vision] |
| `full_lora` | `full` | `vlm_lora_rank` 0 → 16 |

Same data, steps (8,000), batch, LR schedule and evaluation as every other arm.
