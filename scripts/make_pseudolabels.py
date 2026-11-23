"""Offline pseudo-labels for any LeRobot dataset, written to a bounded, resumable sidecar store.

    pixi run -e dev python scripts/make_pseudolabels.py --repo-id lerobot/libero \
        --episodes 0:20 --gripper-dims 6 --out pseudolabels --report results/pseudolabels/libero_cost.json

Per episode (one camera, frames subsampled by ``--stride``):
  instruction -> target noun phrase (rule + grounded SmolVLM2 answer)
  SAM 3 target and gripper tracks, mask-pooled DINOv2 features, relative depth,
  gripper-relative displacement, target motion        (surgicalvla.pseudolabels.vision)
  TAPNext++ point tracks on the target                  (surgicalvla.pseudolabels.tracks)
  phase boundaries from gripper events and speed minima (surgicalvla.pseudolabels.phases)
  a confidence value for every target

Frames are read with LeRobot's ``StreamingLeRobotDataset`` (Hub streaming, or a local
``--root``), in episode order. Episodes already in the store's manifest are skipped, so
an interrupted run resumes. ``--max-gb`` bounds the store's disk use. Nothing here reads
simulator state; the only dataset-specific input is which action dimensions are grippers.
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
from surgicalvla.pseudolabels.sidecar import SidecarStore


def parse_episodes(spec: str, total: int) -> list[int]:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a or 0), min(int(b or total), total)))
    return [int(x) for x in spec.split(",") if x.strip()]


def iter_episodes(args, episodes: list[int], lengths: dict[int, int]):
    """Yield ``(episode_index, frames uint8 (T,H,W,3), state (T,S), action (T,A), task)``.

    The stream may interleave frames of several episodes (prefetching), so frames are
    buffered per episode and an episode is emitted once all ``lengths[ep]`` frames have
    arrived, sorted by ``frame_index``.
    """
    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

    ds = StreamingLeRobotDataset(
        args.repo_id, root=args.root, revision=args.revision, episodes=episodes, streaming=args.root is None, shuffle=False
    )
    camera = args.camera or ds.meta.camera_keys[0]
    buffers: dict[int, dict[int, dict]] = {}

    def assemble(ep: int):
        buf = buffers.pop(ep)
        items = [buf[k] for k in sorted(buf)]
        frames = np.stack([(b[camera].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for b in items])
        state = np.stack([b["observation.state"].numpy() for b in items])
        action = np.stack([b["action"].numpy().reshape(-1, b["action"].shape[-1])[0] for b in items])
        return ep, frames, state, action, str(items[0].get("task", ""))

    for item in ds:
        ep, fi = int(item["episode_index"]), int(item["frame_index"])
        buffers.setdefault(ep, {})[fi] = item
        if len(buffers[ep]) == lengths[ep]:
            yield assemble(ep)
    for ep in list(buffers):  # incomplete episodes (should not happen); label what arrived
        print(f"warning: episode {ep} incomplete ({len(buffers[ep])}/{lengths[ep]} frames)", file=sys.stderr)
        yield assemble(ep)


def grasp_frame(action: np.ndarray, gripper_dims: list[int], labelled: np.ndarray) -> int | None:
    """Index into ``labelled`` of the first frame after the gripper command first closes.

    "Closing" is the direction of the first gripper event: commands are rescaled per episode,
    and a demonstration starts with the gripper open, so the first transition is a close.
    """
    events = [f for d in gripper_dims for f in gripper_events(action[:, d])[0][:1]]
    if not events:
        return None
    return int(np.searchsorted(labelled, min(events)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", help="local dataset root (default: stream from the Hub)")
    ap.add_argument("--revision")
    ap.add_argument("--episodes", default="0:10", help="'a:b' range or comma list")
    ap.add_argument("--camera", help="camera key for the vision targets (default: the dataset's first camera)")
    ap.add_argument("--stride", type=int, default=2, help="label every k-th frame")
    ap.add_argument("--gripper-dims", type=int, nargs="+", required=True, help="action dimensions that command grippers")
    ap.add_argument("--no-tracks", action="store_true", help="skip TAPNext++ point tracks")
    ap.add_argument("--no-vlm", action="store_true", help="noun phrase from the rule only")
    ap.add_argument("--out", default="pseudolabels")
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--report", help="write timing / disk cost per episode and per 1,000 episodes as JSON")
    ap.add_argument("--save-videos", help="directory for per-episode overlay videos of the tracked target")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root, revision=args.revision)
    episodes = parse_episodes(args.episodes, meta.total_episodes)
    config = {"stride": args.stride, "camera": args.camera, "gripper_dims": args.gripper_dims, "tracks": not args.no_tracks, "vlm": not args.no_vlm}
    store = SidecarStore(args.out, args.repo_id, revision=args.revision, config=config, max_gb=args.max_gb)
    todo = [e for e in episodes if not store.done(e)]
    print(f"{len(todo)} of {len(episodes)} episodes to label ({len(episodes) - len(todo)} already in {store.dir})", flush=True)
    if not todo:
        return

    t0 = time.time()
    from surgicalvla.pseudolabels.vision import VisionLabeler

    vision = VisionLabeler(device=args.device)
    tracker = None
    if not args.no_tracks:
        from surgicalvla.pseudolabels.tracks import PointTracker

        tracker = PointTracker(device=args.device)
    vlm = None if args.no_vlm else VLMPhraseExtractor(device=args.device)
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()

    rows = []
    t_read = time.time()
    lengths = {e: int(meta.episodes[e]["length"]) for e in todo}
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
        grasp = grasp_frame(action, args.gripper_dims, idx)
        t = time.time()
        vis = vision.label(sub, phrase, grasp) if phrase else None
        timing["vision_s"] = time.time() - t
        arrays = {
            "frame_index": idx.astype(np.int32),
            "camera": np.array(args.camera or "default"),
            "instruction": np.array(task),
            "target_phrase": np.array(phrase or ""),
            "phrase_confidence": np.array(phrase_conf, np.float16),
        }
        masks = None
        if vis is not None:
            masks = vis.pop("_obj_masks_full")
            vis.pop("_grip_masks_full")
            arrays.update(vis)
        t = time.time()
        if tracker is not None and masks is not None:
            arrays.update(tracker.track(sub, masks, arrays["obj_score"].astype(np.float32)))
        timing["tracks_s"] = time.time() - t
        arrays.update(
            phase_frames=pb.frames.astype(np.int32), phase_kinds=pb.kinds.astype(np.int8), phase_confidence=pb.confidence.astype(np.float16)
        )
        if args.save_videos:
            from surgicalvla.pseudolabels.viz import overlay_frames, write_mp4

            fps = max(1, int(round(meta.fps / args.stride)))
            write_mp4(overlay_frames(sub, arrays), Path(args.save_videos) / f"ep_{ep:06d}.mp4", fps=fps)
        size = store.write(ep, arrays, stats={"frames": int(len(frames)), "labelled": int(len(idx)), **{k: round(v, 3) for k, v in timing.items()}})
        row = {"episode": ep, "frames": int(len(frames)), "labelled": int(len(idx)), "bytes": size, "phrase": phrase,
               "obj_valid": float((arrays.get("obj_score", np.zeros(1)) > 0).mean()), **timing}
        rows.append(row)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        t_read = time.time()

    if args.report and rows:
        per = {k: float(np.mean([r[k] for r in rows])) for k in ("frames", "labelled", "bytes", "read_s", "phrase_s", "vision_s", "tracks_s", "phase_s")}
        total_s = sum(per[k] for k in ("read_s", "phrase_s", "vision_s", "tracks_s", "phase_s"))
        report = {
            "dataset": args.repo_id, "episodes_measured": len(rows), "config": config, "model_load_s": load_s,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "per_episode_mean": per, "seconds_per_episode": total_s,
            "per_1000_episodes": {"hours": total_s * 1000 / 3600, "disk_gb": per["bytes"] * 1000 / 1024**3},
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(json.dumps(report["per_1000_episodes"]), f"peak VRAM {report['peak_vram_gib']:.1f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
