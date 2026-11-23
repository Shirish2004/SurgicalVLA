# Pseudo-label quality against simulator ground truth

Simulator ground truth is used here **only to measure** pseudo-label quality, never as a
training signal.

**Setup.** LIBERO-Spatial task "pick up the black bowl on the cookie box and place it on the
plate", original LIBERO demonstrations 0, 1, 2 (`yifengzhu-hf/LIBERO-datasets`). Recorded
simulator states are replayed and rendered at 256×256 (agentview); the unchanged pseudo-labeler
runs on those frames at stride 2 (`scripts/validate_pseudolabels.py`). Ground truth: target mask
from geom segmentation; target and grip-site image positions by pinhole projection (projection
checked: projected body origin lies 9.8 px from the mask centroid, consistent with a bowl's
origin at its base); grasp = lift onset (target 1 cm above its start), place = last frame the
target moves faster than 2 mm/step. Overlay videos: `results/pseudolabels/validation/`.
Small sample (3 demos, one task): treat as a working baseline, not a final number.

| Target | Demo 0 | Demo 1 | Demo 2 | Verdict |
|---|---|---|---|---|
| target mask IoU (mean over frames where the target is visible) | 0.88 | 0.93 | 0.81 | **keep** |
| frames with IoU > 0.5 | 100 % | 100 % | 100 % | |
| target detected when visible | 100 % | 100 % | 100 % | |
| target centroid error | 1.5 px | 0.8 px | 1.6 px | **keep** |
| target-motion direction (cosine, frames where it moves) | 0.9999 | 0.9997 | 0.9997 | **keep** |
| place boundary error | 2 frames | 5 | 4 | **keep** |
| grasp boundary error | 13 frames | 0 | 15 | keep; see note |
| gripper-relative displacement, end-effector **centroid** | 134 px / cos 0.66 | 115 / 0.67 | 78 / 0.72 | replaced |
| gripper-relative displacement, **nearest** end-effector pixel (default) | 93 px / cos 0.77 | 54 / 0.69 | 18 / 0.73 | **weak** |
| gripper-relative displacement, lowest end-effector pixels | 99 px / cos 0.82 | 59 / 0.71 | 12 / 0.98 | weak |

Notes:
- **Grasp offset is definitional.** The pseudo boundary is the gripper-close event; ground truth
  is lift onset, which follows the close by 0.5-0.75 s in these demos. Demo 1's 0-frame error is
  a speed minimum coinciding with lift.
- **Gripper-relative displacement is the weakest target.** SAM 3's "robot end effector" mask
  covers hand and forearm, so no single pixel of it is the grip point; the image-plane offset is
  in the right direction (cosine ~0.7-0.8) but 20-90 px off on a 256-px image. It is kept with a
  low default loss weight and confidence weighting, and Gate 2 of the ablation decides whether it
  stays. Candidate improvement (not done): track the fingertips with points seeded at the
  end-effector mask's extremity at the first grasp.
- Point tracks and DINOv2 features have no direct simulator counterpart; point-track quality is
  indirectly supported by the target-motion agreement.
