# Pseudo-label pipeline

`scripts/make_pseudolabels.py` produces label-free auxiliary targets offline from raw
demonstrations with frozen models, and stores them as a sidecar next to the dataset
(`src/surgicalvla/pseudolabels/`). No simulator state is read; the only dataset-specific
input is which action dimensions command grippers (`--gripper-dims`).

```bash
pixi run -e dev python scripts/make_pseudolabels.py --repo-id lerobot/libero \
    --episodes 84,169,1608 --gripper-dims 6 --out pseudolabels \
    --report results/pseudolabels/libero_cost_3eps.json
```

## Targets (per episode, one camera, every `--stride`-th frame)

| Target | Source | Confidence |
|---|---|---|
| target noun phrase | rule: object of the first manipulation verb; cross-checked by a grounded SmolVLM2 answer (rule wins on disagreement) | 1.0 agree / 0.5 rule only / 0.4 disagree / 0.7 VLM only |
| target and gripper masks, boxes | SAM 3 video tracking; prompts: phrase, its head noun, "robot end effector", "robot hand" | per-frame SAM 3 score |
| target identity | among object candidates, maximise 0.5 · text score + 0.5 · carried score (target moves with the gripper after the gripper closes) | `obj_select_confidence` |
| object feature | DINOv2-S patch features pooled in the target mask | target score |
| gripper-relative displacement | centroid offset (image units) + relative-depth difference (Depth Anything V2 Small) | min(target, gripper score) |
| target motion | centroid step to the next labelled frame | min of the two scores |
| point tracks | TAPNext++ on a grid inside the first confident target mask | per-point visibility |
| phase boundaries | gripper command events + arm-speed minima | step sharpness / minimum depth |

Storage: one compressed `.npz` per episode with a manifest (resumable, atomic writes, disk cap
`--max-gb`, eviction only of episodes already pushed to the Hub). Frames are read with
`StreamingLeRobotDataset`; episodes may arrive interleaved and are assembled by frame count.

## Design findings from development (LIBERO)

- SAM 3 never detects "gripper" / "robot gripper" on LIBERO or RoboTwin frames; "robot end
  effector" scores 0.5-0.8 on both. Its masks include part of the forearm.
- Simulator renderings often contradict the instruction's colour words (LIBERO's "black bowl"
  is rendered silver), giving text scores of 0.2-0.5. With SAM 3's default video thresholds
  (detect 0.5, new track 0.7) such objects never become tracks: 14 of 20 sampled LIBERO
  episodes had no target. Lowering the thresholds (0.15 / 0.2) and choosing the candidate that
  is carried by the gripper fixed all three failure cases re-checked (visually verified),
  including one where the text score alone preferred a different bowl.
- The rule-based parser must keep coordinated adjectives ("yellow and white mug").

## Cost (baseline: 3 LIBERO episodes 84, 169, 1608; RTX A4500)

| | Value |
|---|---|
| time per episode | 81 s (vision 77 s, point tracks 2.5 s, reading 1.5 s) |
| per 1,000 episodes | **22.6 GPU-hours, 0.08 GB disk** |
| peak VRAM | 11.3 GiB (SAM 3 video with low thresholds dominates) |
| model load | ~7 s |

Extrapolated to full datasets at stride 2: LIBERO (1,693 episodes) ≈ 38 GPU-h; RoboTwin
clean-50 (2,500) ≈ 55-60 GPU-h (longer episodes); VLABench (10,977) ≈ 250 GPU-h, which needs
subsampling or the speed-ups below. Estimates rest on 3 episodes.

Speed-ups not yet applied: stride 4 (about half the time), a single end-effector prompt,
running SAM 3 on the object prompts at a lower internal resolution, and the `kernels`
package (better SAM 3 post-processing; may change speed either way).

Quality against simulator ground truth: `docs/pseudolabel_quality.md`. Tracking videos:
`--save-videos <dir>` (target mask green, gripper box red, point tracks yellow).

## Reasoning traces (`scripts/make_cot_traces.py`)

Candidates per phase segment are written from the pseudo-labels, e.g.
`<cot> phase: grasp | object: black bowl | motion: right </cot>`, and scored by the paired
flow-matching-loss gain on the segment's middle frame. Measured with `lerobot/smolvla_libero` on
2 episodes (19 segments): **every candidate had negative gain** (-0.01 to -0.35); a policy never
trained with traces in its prefix treats them as distractors. Round 1 therefore runs with
`--bootstrap` (unfiltered); filtering applies from round 2, scored by the trace-trained policy.
