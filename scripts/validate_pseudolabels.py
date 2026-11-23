"""Validate pseudo-labels against LIBERO simulator ground truth (validation only, never training).

    MUJOCO_GL=egl pixi run -e sim-eval python scripts/validate_pseudolabels.py \
        --hdf5 <LIBERO demo file> --demos 0 1 2 --out results/pseudolabels/validation

For each demonstration the recorded simulator states are replayed; RGB frames and geom
segmentation are rendered from the same state with ``mujoco.Renderer``. Ground truth:

* target mask: pixels whose geom belongs to the task's first object of interest;
* target / end-effector image position: pinhole projection of the body / eef position;
* events: grasp = lift onset (target 1 cm above its start height), place = last frame the
  target moves faster than 2 mm per step.

The pseudo-labeler runs unchanged on the rendered frames (it never sees the ground truth).
Metrics: mask IoU, target centroid error, gripper-relative displacement error and direction
agreement, target-motion direction agreement, and the frame error of the nearest pseudo phase
boundary to each ground-truth event. Overlay videos are written alongside.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def project(model, data, cam: str, points: np.ndarray, h: int, w: int) -> np.ndarray:
    """World points ``(N, 3)`` -> normalised image ``[x, y]`` for ``mujoco.Renderer`` output."""
    import mujoco

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    pos, rot = data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3)
    pc = (np.asarray(points) - pos) @ rot  # camera frame: x right, y up, looking along -z
    f = 0.5 * h / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
    u = w / 2 + f * pc[:, 0] / -pc[:, 2]
    v = h / 2 - f * pc[:, 1] / -pc[:, 2]
    return np.stack([u / w, v / h], axis=1)


def _patch_robosuite_segmentation() -> None:
    """robosuite 1.4 combines the uint8 id colour channels without widening (overflows on NumPy 2)."""
    import mujoco
    from robosuite.utils import binding_utils

    def read_pixels(self, width, height, depth=False, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        dep = np.empty((height, width), dtype=np.float32) if depth else None
        mujoco.mjr_readPixels(rgb=rgb, depth=dep, viewport=viewport, con=self.con)
        out = rgb
        if segmentation:
            c = rgb.astype(np.int32)
            seg = c[..., 0] + c[..., 1] * 256 + c[..., 2] * 65536
            seg[seg >= self.scn.ngeom + 1] = 0
            ids = np.full((self.scn.ngeom + 1, 2), -1, dtype=np.int32)
            for i in range(self.scn.ngeom):
                g = self.scn.geoms[i]
                if g.segid != -1:
                    ids[g.segid + 1] = (g.objtype, g.objid)
            out = ids[seg]
        return (out, dep) if depth else out

    binding_utils.MjRenderContext.read_pixels = read_pixels


def replay(env, states: np.ndarray, target: str, cam: str, size: int):
    import mujoco

    _patch_robosuite_segmentation()
    sim = env.env.sim
    model, data = sim.model._model, sim.data._data
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    target_bodies = {b for b, n in enumerate(body_names) if n.startswith(target)}
    target_geoms = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in target_bodies])
    root = min(target_bodies)
    eef_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    frames, masks, obj_px, eef_px, obj_z, obj_pos = [], [], [], [], [], []
    for s in states:
        sim.set_state_from_flattened(s)
        sim.forward()
        # robosuite renders bottom-up (OpenGL origin); flip to top-down to match project()
        frames.append(np.flipud(sim.render(camera_name=cam, height=size, width=size)).copy())
        seg = np.flipud(sim.render(camera_name=cam, height=size, width=size, segmentation=True))
        masks.append((seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 1], target_geoms))
        p = data.xpos[root].copy()
        obj_pos.append(p)
        obj_z.append(p[2])
        obj_px.append(project(model, data, cam, p[None], size, size)[0])
        eef_px.append(project(model, data, cam, data.site_xpos[eef_site][None], size, size)[0])
    return np.stack(frames), np.stack(masks), np.stack(obj_px), np.stack(eef_px), np.array(obj_z), np.stack(obj_pos)


def events(obj_z: np.ndarray, obj_pos: np.ndarray) -> dict[str, int | None]:
    lift = np.flatnonzero(obj_z > obj_z[0] + 0.01)
    speed = np.linalg.norm(np.diff(obj_pos, axis=0), axis=1)
    moving = np.flatnonzero(speed > 0.002)
    return {"grasp": int(lift[0]) if len(lift) else None, "place": int(moving[-1] + 1) if len(moving) else None}


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--demos", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--out", default="results/pseudolabels/validation")
    args = ap.parse_args()

    import h5py
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from surgicalvla.pseudolabels.nounphrase import extract_phrase
    from surgicalvla.pseudolabels.phases import gripper_events, phase_boundaries
    from surgicalvla.pseudolabels.sidecar import mask_to_bits
    from surgicalvla.pseudolabels.vision import VisionLabeler, gripper_point
    from surgicalvla.pseudolabels.viz import overlay_frames, write_mp4

    stem = Path(args.hdf5).name.removesuffix("_demo.hdf5")
    bench = benchmark.get_benchmark_dict()[args.suite]()
    task = next(t for t in bench.tasks if t.name == stem)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        camera_heights=args.size, camera_widths=args.size, camera_names=[args.camera],
    )
    env.reset()
    target = env.env.obj_of_interest[0]
    phrase, phrase_conf = extract_phrase(task.language)
    labeler = VisionLabeler()
    out = Path(args.out)
    results = []
    with h5py.File(args.hdf5, "r") as f:
        for k in args.demos:
            demo = f["data"][f"demo_{k}"]
            states, actions = demo["states"][()], demo["actions"][()]
            proprio = np.concatenate([demo["obs"]["ee_states"][()], demo["obs"]["gripper_states"][()]], axis=1)
            frames, gt_masks, gt_obj, gt_eef, obj_z, obj_pos = replay(env, states, target, args.camera, args.size)
            idx = np.arange(0, len(frames), args.stride)
            grasp_events = gripper_events(actions[:, 6])[0]
            grasp = int(np.searchsorted(idx, grasp_events[0])) if len(grasp_events) else None
            lab = labeler.label(frames[idx], phrase, grasp)
            masks = lab.pop("_obj_masks_full")
            grip_masks = lab.pop("_grip_masks_full")
            valid = lab["obj_score"].astype(float) > 0
            gtm = gt_masks[idx]
            inter = (masks & gtm).sum((1, 2))
            union = (masks | gtm).sum((1, 2))
            iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
            visible = gtm.reshape(len(idx), -1).any(1)
            pseudo_c = np.array([[np.nan, np.nan] if not m.any() else [np.nonzero(m)[1].mean() / args.size, np.nonzero(m)[0].mean() / args.size] for m in masks])
            gt_c = np.array([[np.nan, np.nan] if not m.any() else [np.nonzero(m)[1].mean() / args.size, np.nonzero(m)[0].mean() / args.size] for m in gtm])
            cen_err = np.linalg.norm(pseudo_c - gt_c, axis=1)
            rel_gt = gt_obj[idx] - gt_eef[idx]
            rel_ok = lab["rel_confidence"].astype(float) > 0
            rel_modes = {}
            for mode in ("centroid", "nearest", "lowest"):
                ps = np.array([
                    (pseudo_c[j] - g) if (g := gripper_point(grip_masks[j], pseudo_c[j], mode)) is not None and not np.isnan(pseudo_c[j]).any() else [np.nan, np.nan]
                    for j in range(len(idx))
                ])
                ok = rel_ok & ~np.isnan(ps).any(1)
                rel_modes[mode] = {
                    "err_px": float(np.median(np.linalg.norm(ps[ok] - rel_gt[ok], axis=1)) * args.size) if ok.any() else None,
                    "cosine": float(np.median(cosine(ps[ok], rel_gt[ok]))) if ok.any() else None,
                }
            rel_ps = lab["rel_disp"][:, :2].astype(float)
            mot_gt = np.diff(gt_c, axis=0, append=gt_c[-1:])
            mot_ps = lab["obj_motion"].astype(float)
            moving = np.linalg.norm(mot_gt, axis=1) > 0.005
            mot_ok = (lab["motion_confidence"].astype(float) > 0) & moving & ~np.isnan(mot_gt).any(1)
            pb = phase_boundaries(proprio, actions, [6])
            ev = events(obj_z, obj_pos)
            ev_err = {name: (None if t is None or not len(pb.frames) else int(np.min(np.abs(pb.frames - t)))) for name, t in ev.items()}
            row = {
                "demo": k,
                "frames": int(len(frames)),
                "phrase": phrase,
                "target_body": target,
                "select_confidence": float(lab["obj_select_confidence"]),
                "mask_iou_mean_visible": float(np.nanmean(iou[visible])) if visible.any() else None,
                "mask_iou_median_visible": float(np.nanmedian(iou[visible])) if visible.any() else None,
                "frames_iou_gt_0.5": float(np.mean(iou[visible] > 0.5)) if visible.any() else None,
                "target_detected_when_visible": float(np.mean(valid[visible])) if visible.any() else None,
                "centroid_err_px": float(np.nanmedian(cen_err) * args.size),
                # sanity check of the projection: projected body origin vs ground-truth mask centroid
                "gt_projection_vs_mask_px": float(np.nanmedian(np.linalg.norm(gt_obj[idx] - gt_c, axis=1)) * args.size),
                "rel_disp_err_px": float(np.median(np.linalg.norm(rel_ps[rel_ok] - rel_gt[rel_ok], axis=1)) * args.size) if rel_ok.any() else None,
                "rel_disp_cosine": float(np.median(cosine(rel_ps[rel_ok], rel_gt[rel_ok]))) if rel_ok.any() else None,
                "rel_disp_by_gripper_point": rel_modes,
                "motion_cosine_when_moving": float(np.median(cosine(mot_ps[mot_ok], mot_gt[mot_ok]))) if mot_ok.any() else None,
                "gt_events": ev,
                "phase_boundary_err_frames": ev_err,
                "n_pseudo_boundaries": int(len(pb.frames)),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            lab["obj_mask"] = np.stack([mask_to_bits(m) for m in masks])
            lab["target_phrase"] = np.array(phrase)
            write_mp4(overlay_frames(frames[idx], lab), out / f"{stem}_demo{k}.mp4", fps=10)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps({"task": stem, "phrase_confidence": phrase_conf, "demos": results}, indent=2))


if __name__ == "__main__":
    main()
