# Feature and pseudo-label cache

`scripts/make_pseudolabels.py` (`pixi run -e dev cache --suite <suite>`) produces, offline and
with frozen models only, two kinds of data per LeRobot episode:

- **Policy inputs.** Object tokens on every frame (`--input-stride 1`). They are
  **past-only**: computed from that frame alone (`surgicalvla/perception/object_inputs.py`),
  exactly as the policy computes them live at evaluation.
- **Training targets.** Every 4th frame (`--stride 4`). They may use the whole episode and
  enter only losses.

No simulator state is read; the only dataset-specific input is which action dimensions
command grippers (`--gripper-dims`).

## Storage (`src/surgicalvla/pseudolabels/cache.py`)

Layout:

```
pseudolabels/<dataset>/<cache-name>/manifest.json
pseudolabels/<dataset>/<cache-name>/shards/{frames,episodes,cot}-NNNNN.parquet
```

- **Rows.** One row per frame, keyed by (episode, frame) as in the LeRobot dataset. Masks are
  stored as full-resolution RLE and features as fp16. Raw frames are never stored.
- **Manifest.**
  - module ids, Hub revisions and weight SHA-256 (read from the Hugging Face cache blob names);
  - the TAPNext++ checkpoint hash;
  - prompts, thresholds, strides, camera, causal mode;
  - library versions and git commit;
  - creation time;
  - a config hash over everything that changes the values.
- **Mismatch.** A training run states what it expects (`aux.expect`, plus what the model
  needs). A mismatched cache is refused with the differing field named, e.g.
  `labeler.stride: training config expects 4, cache has 2`.
- **Integrity.** Shards are written atomically and recorded with a SHA-256 checksum. A
  damaged shard is detected on read, and an interrupted run resumes after the last complete
  shard.
- **Bounded disk.** With `--upload-repo` (a private Hub dataset), finished shards are
  uploaded, and uploaded shards are evicted locally, least recently used first, beyond
  `--max-gb`. Readers download evicted shards on demand.
- **Augmentation.** None: the training data has no geometric augmentation (tested), and the
  policy's resize-with-padding is deterministic and identical in training and evaluation, so
  cached boxes, masks and tracks stay aligned with the images.

### Inputs per frame (K = 4 objects: 3 phrase candidates + end effector; plus the grid control)

| Column | Shape | Content |
|---|---|---|
| `in_obj_feat` | (4, 384) fp16 | DINOv2-S patch features pooled in each SAM 3 instance mask |
| `in_obj_pose` | (4, 8) fp16 | box xyxy, mask centroid, median relative depth, SAM 3 score |
| `in_obj_valid` | (4,) bool | instance present (score ≥ 0.15; target candidates not covered by the end effector; duplicates merged) |
| `in_grid_feat`, `in_grid_pose` | (4, 384), (4, 8) | control: the same encoders pooled over a fixed 2×2 grid |

SAM 3 runs in image mode with the image features computed once and shared by the prompts
(0.22 s per frame instead of 0.65 s, identical scores).

```bash
pixi run -e dev python scripts/make_pseudolabels.py --repo-id lerobot/libero \
    --episodes 84,169,1608 --camera observation.images.image --gripper-dims 6 --out pseudolabels \
    --report results/pseudolabels/libero_cost_3eps.json
```

## Targets (per episode, the frontal camera, every `--stride`-th frame)

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

Frames are read with `StreamingLeRobotDataset`; episodes may arrive interleaved and are
assembled by frame count.

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

## Cost

Current cache (inputs on every frame + targets every 4th frame), measured on 2 LIBERO-Spatial
episodes (1261, 1262), RTX A4500:

| | Value |
|---|---|
| time per episode | 57 s: inputs 30 s, targets 24 s, point tracks 0.8 s |
| per 1,000 episodes | **15.9 GPU-hours, 0.75 GB disk** |
| peak VRAM | 7.3 GiB (with `--offload-video`) |
| LIBERO-Spatial (432 episodes) | ≈ 6.9 GPU-h, 0.33 GB |
| all four LIBERO suites (1,693) | ≈ 27 GPU-h, 1.3 GB |

The pipeline's hyperparameters (prompts, thresholds, stride) are frozen at the values in the
manifest. They were chosen on 3 LIBERO training demonstrations (84, 169, 1608: Goal, Goal,
Spatial), never on evaluation episodes, and are applied unchanged to every suite and benchmark.

### Earlier measurement (targets only, stride 2, 3 episodes 84, 169, 1608)

| | Value |
|---|---|
| time per episode | 81 s (vision 77 s, point tracks 2.5 s, reading 1.5 s) |
| per 1,000 episodes | **22.6 GPU-hours, 0.08 GB disk** |
| peak VRAM | 11.3 GiB (SAM 3 video with low thresholds dominates) |
| model load | ~7 s |

Extrapolated to full datasets at stride 2: LIBERO (1,693 episodes) ≈ 38 GPU-h; RoboTwin
clean-50 (2,500) ≈ 55-60 GPU-h (longer episodes); VLABench (10,977) ≈ 250 GPU-h, which needs
subsampling or the speed-ups below. Estimates rest on 3 episodes.

With `--offload-video` (used by `benchmark.py labels`) SAM 3 keeps the decoded frames in host
memory: peak VRAM falls to 5.3 GiB on a LIBERO episode, and `--gpu-mem-gb` caps the process so
labelling can share the GPU with training. LIBERO-Spatial (432 episodes) is labelled at stride 4
(`benchmark.py labels` default).

Speed-ups not yet applied: a single end-effector prompt,
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
