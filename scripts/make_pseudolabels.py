"""Feature and label cache for any LeRobot dataset (Parquet shards, ``surgicalvla.pseudolabels.cache``).

    pixi run -e dev python scripts/make_pseudolabels.py --repo-id lerobot/libero \
        --episodes 0:20 --camera observation.images.image --gripper-dims 6 --out pseudolabels \
        --report results/pseudolabels/libero_cost.json

Two kinds of data, from frozen models only:

* **Inputs** (every ``--input-stride``-th frame; past-only): per-frame SAM 3 image detections of
  the target phrase and the end effector, DINOv2 features pooled in each mask, box, centroid,
  relative depth and score, plus the 2×2-grid control (``surgicalvla.perception.object_inputs``).
  Each frame's inputs depend on that frame alone, which a test checks.
* **Targets** (every ``--stride``-th frame; may use the whole episode): the noun phrase (rule
  + grounded SmolVLM2 answer), SAM 3 video tracks of target and gripper (RLE masks), the
  target's DINOv2 feature, relative depth, gripper-relative offset and motion
  (``pseudolabels.vision``), TAPNext++ point tracks (``pseudolabels.tracks``), and phase
  boundaries from gripper events and speed minima (``pseudolabels.phases``), each with a
  confidence.

Frames are read with LeRobot's ``StreamingLeRobotDataset`` in episode order. The cache is
resumable (finished shards are skipped), bounded by ``--max-gb``, optionally uploaded to a Hub
dataset (``--upload-repo``), and its manifest records every model id, revision, weight hash,
prompt, threshold and setting. Nothing here reads simulator state; the only dataset-specific
input is which action dimensions are grippers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from surgicalvla.pseudolabels.nounphrase import VLMPhraseExtractor, extract_phrase
from surgicalvla.pseudolabels.phases import gripper_events, phase_boundaries


def parse_episodes(spec: str, total: int) -> list[int]:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a or 0), min(int(b or total), total)))
    return [int(x) for x in spec.split(",") if x.strip()]


def iter_episodes(args, episodes: list[int], lengths: dict[int, int]):
    """Yield ``(episode_index, frames uint8 (T,H,W,3), state (T,S), action (T,A), task)``.

    Episodes are read one at a time (one stream per episode), so host memory holds a single
    episode; within it, frames are ordered by ``frame_index``.
    """
    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

    camera = args.camera
    buffers: dict[int, dict[int, dict]] = {}

    def compact(item: dict) -> dict:
        # keep only what labelling needs: the labelled camera as uint8 (other cameras and float
        # copies of large frames would exhaust host memory while episodes are buffered)
        return {"frame": (item[camera].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                "state": item["observation.state"].numpy(),
                "action": item["action"].numpy().reshape(-1, item["action"].shape[-1])[0],
                "task": str(item.get("task", ""))}

    def assemble(ep: int):
        buf = buffers.pop(ep)
        items = [buf[k] for k in sorted(buf)]
        return (ep, np.stack([b["frame"] for b in items]), np.stack([b["state"] for b in items]),
                np.stack([b["action"] for b in items]), items[0]["task"])

    for want in episodes:
        ds = StreamingLeRobotDataset(
            args.repo_id, root=args.root, revision=args.revision, episodes=[want], streaming=args.root is None, shuffle=False
        )
        if camera not in ds.meta.camera_keys:
            raise SystemExit(f"--camera {camera!r} is not a camera of {args.repo_id}: {ds.meta.camera_keys}")
        for item in ds:
            ep, fi = int(item["episode_index"]), int(item["frame_index"])
            if ep == want:
                buffers.setdefault(ep, {})[fi] = compact(item)
        if want in buffers:
            if len(buffers[want]) != lengths[want]:
                print(f"warning: episode {want} incomplete ({len(buffers[want])}/{lengths[want]} frames)", file=sys.stderr)
            yield assemble(want)


def grasp_frame(action: np.ndarray, gripper_dims: list[int], labelled: np.ndarray) -> int | None:
    """Index into ``labelled`` of the first frame after the gripper command first closes.

    "Closing" is the direction of the first gripper event: commands are rescaled per episode,
    and a demonstration starts with the gripper open, so the first transition is a close.
    """
    events = [f for d in gripper_dims for f in gripper_events(action[:, d])[0][:1]]
    if not events:
        return None
    return int(np.searchsorted(labelled, min(events)))


def provenance(args, vision_cfg, input_cfg, tracker_ckpt: str | None) -> dict:
    """Labeler settings and module provenance for the cache manifest."""
    import dataclasses
    import subprocess

    import transformers

    from surgicalvla.pseudolabels.cache import config_hash, file_sha256, hub_weights

    labeler = {
        "stride": args.stride, "input_stride": args.input_stride, "camera": args.camera,
        "gripper_dims": list(args.gripper_dims), "tracks": not args.no_tracks, "vlm": not args.no_vlm,
        "causal_inputs": True, "targets_use_future": True,
        "vision": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(vision_cfg).items() if k != "offload_video"},
        "object_inputs": input_cfg.as_dict(),
        "track_points": 32,
    }
    modules = {
        "sam3": hub_weights(vision_cfg.sam_id), "dinov2": hub_weights(vision_cfg.dino_id),
        "depth": hub_weights(vision_cfg.depth_id),
    }
    if not args.no_vlm:
        modules["phrase_vlm"] = hub_weights("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    if tracker_ckpt:
        modules["tapnextpp"] = {"id": "TAPNext++ (vendored, Apache-2.0)", "path": tracker_ckpt, "sha256": file_sha256(tracker_ckpt)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    return {"revision": args.revision, "labeler": labeler, "modules": modules,
            "versions": {"transformers": transformers.__version__, "torch": torch.__version__, "git_commit": commit},
            "config_hash": config_hash(labeler, modules)}


def episode_rows(n: int, stride: int, input_stride: int, vis: dict | None, tracks: dict | None, inputs: dict | None,
                 k: int, c: int, p: int) -> dict:
    """Per-frame columns for one episode: union of input frames and target frames."""
    t_idx = np.arange(0, n, stride)
    i_idx = np.arange(0, n, input_stride)
    frames = np.union1d(t_idx, i_idx)
    rows = len(frames)
    is_t = np.isin(frames, t_idx)
    is_i = np.isin(frames, i_idx)
    tpos = np.searchsorted(t_idx, frames)  # row -> labelled-frame position (valid where is_t)
    ipos = np.searchsorted(i_idx, frames)
    f16 = np.float16
    cols = {"frame": frames.astype(np.int32), "has_target": is_t, "has_input": is_i}

    def put(name, src, shape, dtype, where, pos):
        arr = np.zeros((rows, *shape), dtype)
        if src is not None:
            arr[where] = np.asarray(src)[pos[where]].astype(dtype)
        cols[name] = arr

    for name, shape in (("obj_feat", (k, c)), ("obj_pose", (k, 8)), ("grid_feat", (4, c)), ("grid_pose", (4, 8))):
        put(f"in_{name}", None if inputs is None else inputs[name], shape, f16, is_i, ipos)
    put("in_obj_valid", None if inputs is None else inputs["obj_valid"], (k,), bool, is_i, ipos)
    for name, shape in (("obj_box", (4,)), ("obj_feat", (c,)), ("grip_box", (4,)), ("rel_disp", (3,)), ("obj_motion", (2,))):
        put(name, None if vis is None else vis[name], shape, f16, is_t, tpos)
    for name in ("obj_score", "grip_score", "obj_depth", "grip_depth", "rel_confidence", "motion_confidence"):
        put(name, None if vis is None else vis[name], (), f16, is_t, tpos)
    tr = np.zeros((len(t_idx), p, 2), f16)
    tv = np.zeros((len(t_idx), p), bool)
    if tracks is not None and len(tracks["tracks"]):
        q = min(p, tracks["tracks"].shape[0])
        tr[:, :q] = np.transpose(tracks["tracks"][:q], (1, 0, 2))
        tv[:, :q] = tracks["track_visible"][:q].T
    put("track_xy", tr, (p, 2), f16, is_t, tpos)
    put("track_visible", tv, (p,), bool, is_t, tpos)
    from surgicalvla.pseudolabels.cache import rle_encode

    empty = np.zeros(1, np.int32)
    for name in ("obj", "grip"):
        full = None if vis is None else vis[f"_{name}_masks_full"]
        cols[f"{name}_mask_rle"] = [rle_encode(full[tpos[r]]) if (full is not None and is_t[r]) else empty for r in range(rows)]
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", help="local dataset root (default: stream from the Hub)")
    ap.add_argument("--revision")
    ap.add_argument("--episodes", default="0:10", help="'a:b' range, comma list, or a JSON file with a list")
    ap.add_argument("--camera", required=True, help="camera key to label: the frontal view (scripts/benchmark.py CACHE)")
    ap.add_argument("--stride", type=int, default=4, help="targets on every k-th frame")
    ap.add_argument("--input-stride", type=int, default=1, help="policy inputs on every k-th frame")
    ap.add_argument("--gripper-dims", type=int, nargs="+", required=True, help="action dimensions that command grippers")
    ap.add_argument("--num-grippers", type=int, default=1, help="end effectors given as policy-input objects (2 for bimanual)")
    ap.add_argument("--no-tracks", action="store_true", help="skip TAPNext++ point tracks")
    ap.add_argument("--no-vlm", action="store_true", help="noun phrase from the rule only")
    ap.add_argument("--out", default="pseudolabels", help="cache root")
    ap.add_argument("--name", help="cache name (default: v3-<first 12 hex of the config hash>)")
    ap.add_argument("--relabel", help="episodes to label again although cached (same forms as --episodes); "
                                      "their traces are re-scored too")
    ap.add_argument("--relabel-reason", help="why, recorded in the manifest's relabel log")
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--upload-repo", help="Hub dataset for finished shards, e.g. Shirish24/surgical-vla-cache")
    ap.add_argument("--report", help="write timing / disk cost per episode and per 1,000 episodes as JSON")
    ap.add_argument("--save-videos", help="directory for per-episode overlay videos of the tracked target")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offload-video", action="store_true", help="keep video frames in host memory (lower VRAM)")
    ap.add_argument("--gpu-mem-gb", type=float, help="cap this process's GPU memory (e.g. when sharing the GPU with training)")
    ap.add_argument("--shard", help="'i/n': label only worker i's share of --episodes (parallel workers on one cache)")
    args = ap.parse_args()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    from surgicalvla.perception.object_inputs import ObjectInputConfig, ObjectInputExtractor
    from surgicalvla.pseudolabels.cache import FeatureCache
    from surgicalvla.pseudolabels.tracks import DEFAULT_CKPT
    from surgicalvla.pseudolabels.vision import VisionConfig, VisionLabeler

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root, revision=args.revision)
    spec = args.episodes
    episodes = [int(e) for e in json.loads(Path(spec).read_text())] if spec.endswith(".json") else parse_episodes(spec, meta.total_episodes)
    from surgicalvla.pseudolabels.cache import take_shard

    episodes = take_shard(episodes, args.shard)
    vision_cfg, input_cfg = VisionConfig(offload_video=args.offload_video), ObjectInputConfig(num_grippers=args.num_grippers)
    manifest = provenance(args, vision_cfg, input_cfg, None if args.no_tracks else str(DEFAULT_CKPT))
    cache = FeatureCache(args.out, args.repo_id, args.name or f"v3-{manifest['config_hash'][:12]}", manifest=manifest,
                         max_gb=args.max_gb, upload_repo=args.upload_repo)
    if args.relabel:
        spec = args.relabel
        again = [int(e) for e in json.loads(Path(spec).read_text())] if spec.endswith(".json") else parse_episodes(spec, meta.total_episodes)
        gone = cache.forget(again, args.relabel_reason or "relabel requested")
        print(f"relabelling {len(gone)} cached episodes", flush=True)
    todo = [e for e in episodes if not cache.done(e)]
    print(f"{len(todo)} of {len(episodes)} episodes to label ({len(episodes) - len(todo)} already in {cache.dir})", flush=True)
    if not todo:
        return

    if args.gpu_mem_gb and args.device.startswith("cuda"):
        dev = torch.device(args.device)
        dev = torch.device("cuda", dev.index if dev.index is not None else torch.cuda.current_device())
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gb / total), dev)
    t0 = time.time()
    vision = VisionLabeler(device=args.device, config=vision_cfg)
    inputs_x = ObjectInputExtractor(input_cfg, device=args.device, shared={
        "dino": vision.dino, "dino_proc": vision.dino_proc, "depth": vision.depth, "depth_proc": vision.depth_proc})
    tracker = None
    if not args.no_tracks:
        from surgicalvla.pseudolabels.tracks import PointTracker

        tracker = PointTracker(device=args.device)
    vlm = None if args.no_vlm else VLMPhraseExtractor(device=args.device)
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    k, c, p = input_cfg.num_candidates + input_cfg.num_grippers, inputs_x.feat_dim, 32

    import signal

    def _interrupt(sig, _frame):  # kill / hang-up: leave through `finally`, which saves finished work
        raise KeyboardInterrupt(f"signal {sig}")

    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _interrupt)
    rows = []
    t_read = time.time()
    lengths = {e: int(meta.episodes[e]["length"]) for e in todo}
    try:
        for ep, frames, state, action, task in iter_episodes(args, todo, lengths):
            timing = {"read_s": time.time() - t_read}
            t = time.time()
            phrase, phrase_conf = extract_phrase(task, vlm)
            timing["phrase_s"] = time.time() - t
            idx = np.arange(0, len(frames), args.stride)
            sub = frames[idx]
            t = time.time()
            pb = phase_boundaries(state, action, args.gripper_dims)
            timing["phase_s"] = time.time() - t
            t = time.time()
            vis = vision.label(sub, phrase, grasp_frame(action, args.gripper_dims, idx)) if phrase else None
            timing["vision_s"] = time.time() - t
            t = time.time()
            tracks = None
            if tracker is not None and vis is not None:
                tracks = tracker.track(sub, vis["_obj_masks_full"], vis["obj_score"].astype(np.float32))
            timing["tracks_s"] = time.time() - t
            t = time.time()
            inputs = inputs_x(frames[:: args.input_stride], phrase)
            timing["inputs_s"] = time.time() - t
            cols = episode_rows(len(frames), args.stride, args.input_stride, vis, tracks, inputs, k, c, p)
            h, w = frames.shape[1:3]
            cache.add({"episode": int(ep), "length": int(len(frames)), "instruction": task, "phrase": phrase or "",
                       "phrase_confidence": float(phrase_conf), "height": int(h), "width": int(w),
                       "obj_select_confidence": float(vis["obj_select_confidence"]) if vis is not None else 0.0,
                       "phase_frames": pb.frames.astype(int).tolist(), "phase_kinds": pb.kinds.astype(int).tolist(),
                       "phase_confidence": pb.confidence.astype(float).tolist()}, cols)
            if args.save_videos and vis is not None:
                from surgicalvla.pseudolabels.viz import overlay_frames, write_mp4

                labels = {**vis, "target_phrase": phrase}
                if tracks is not None:
                    labels.update(tracks)
                fps = max(1, int(round(meta.fps / args.stride)))
                write_mp4(overlay_frames(sub, labels), Path(args.save_videos) / f"ep_{ep:06d}.mp4", fps=fps)
            row = {"episode": ep, "frames": int(len(frames)), "labelled": int(len(idx)), "input_frames": int(len(frames[:: args.input_stride])),
                   "phrase": phrase, "obj_valid": float((vis["obj_score"] > 0).mean()) if vis is not None else 0.0,
                   "inputs_valid": float(inputs["obj_valid"][:, : input_cfg.num_candidates].any(axis=1).mean()),
                   "grippers_found": float(inputs["obj_valid"][:, input_cfg.num_candidates :].sum(axis=1).mean()), **timing}
            rows.append(row)
            print(json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in row.items()}), flush=True)
            t_read = time.time()
    finally:
        cache.flush()

    if args.report and rows:
        keys = ("frames", "labelled", "input_frames", "read_s", "phrase_s", "vision_s", "tracks_s", "inputs_s", "phase_s")
        per = {kk: float(np.mean([r[kk] for r in rows])) for kk in keys}
        total_s = sum(per[kk] for kk in keys if kk.endswith("_s"))
        bytes_per_ep = cache.disk_bytes() / max(len(cache.episodes()), 1)
        report = {
            "dataset": args.repo_id, "cache": str(cache.dir), "episodes_measured": len(rows), "model_load_s": load_s,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "per_episode_mean": per,
            "seconds_per_episode": total_s, "bytes_per_episode": bytes_per_ep,
            "per_1000_episodes": {"gpu_hours": total_s * 1000 / 3600, "disk_gb": bytes_per_ep * 1000 / 1024**3},
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(json.dumps(report["per_1000_episodes"]), f"peak VRAM {report['peak_vram_gib']:.1f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
