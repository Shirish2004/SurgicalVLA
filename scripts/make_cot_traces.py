"""Write action-predictive reasoning traces for a pseudo-labelled LeRobot dataset.

    pixi run -e dev python scripts/make_cot_traces.py --repo-id lerobot/libero --episodes 84,169,1608 \
        --sidecar pseudolabels --checkpoint lerobot/smolvla_libero --revision a24f1f38...

For each episode and each phase segment (from the sidecar's phase boundaries), candidate
traces are built from the pseudo-labels and scored on the segment's middle frame by the
policy's flow-matching-loss gain (``surgicalvla.pseudolabels.cot``). The best trace with a
positive gain is stored per segment in ``<sidecar>/<dataset>/cot_<episode>.json``; segments
where no trace helps get none. The scoring policy is the SFT policy the traces are for (a
SmolVLA checkpoint is up-cycled to SurgicalVLA first).

Bootstrapping (R&B-EnCoRe): a policy never trained with traces in its prefix finds every
trace harmful (all gains negative on LIBERO, docs/pseudolabel_quality.md). Round 1 therefore
uses ``--bootstrap``, which stores the full trace for every segment unfiltered; after SFT on
those traces, round 2 re-scores with the trained policy and keeps only positive-gain traces.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch


def load_policy(ref: str, revision: str | None, device: str):
    from huggingface_hub import snapshot_download
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    import surgicalvla.modules.configuration_smolvla_moe  # noqa: F401
    from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig
    from surgicalvla.modules.modeling_smolvla_moe import SmolVLAMoEPolicy

    path = Path(ref) if Path(ref).is_dir() else Path(snapshot_download(ref, revision=revision))
    if json.loads((path / "config.json").read_text()).get("type") == "smolvla_moe":
        policy = SmolVLAMoEPolicy.from_pretrained(str(path))
    else:
        base = SmolVLAConfig.from_pretrained(str(path))
        fields = {f.name for f in dataclasses.fields(base)}
        cfg = SmolVLAMoEConfig(**{f.name: getattr(base, f.name) for f in dataclasses.fields(SmolVLAMoEConfig) if f.name in fields})
        policy = SmolVLAMoEPolicy.from_smolvla_pretrained(str(path), cfg)
    policy.config.device = device
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": device}})
    return policy.to(device).eval(), pre, path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root")
    ap.add_argument("--episodes", required=True, help="comma list")
    ap.add_argument("--sidecar", default="pseudolabels")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--rename", default='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}')
    ap.add_argument("--bootstrap", action="store_true", help="round 1: keep the full trace for every segment, unfiltered")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

    from surgicalvla.pseudolabels.cot import candidates, score
    from surgicalvla.pseudolabels.sidecar import SidecarStore

    store = SidecarStore(args.sidecar, args.repo_id)
    policy, pre, _ = load_policy(args.checkpoint, args.revision, args.device)
    tok = policy.model._tokenizer()
    rename = json.loads(args.rename)
    episodes = [int(e) for e in args.episodes.split(",")]
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    from lerobot.datasets.factory import resolve_delta_timestamps

    ds = StreamingLeRobotDataset(
        args.repo_id, root=args.root, episodes=episodes, streaming=args.root is None, shuffle=False,
        delta_timestamps=resolve_delta_timestamps(policy.config, meta),
    )
    wanted: dict[int, dict[int, int]] = {}
    for ep in episodes:
        d = store.read(ep)
        bounds = [0, *[int(f) for f in d["phase_frames"]], int(meta.episodes[ep]["length"])]
        wanted[ep] = {(a + b) // 2: i for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)) if b > a}
    results: dict[int, list] = {ep: [] for ep in episodes}
    for item in ds:
        ep, fi = int(item["episode_index"]), int(item["frame_index"])
        if fi not in wanted.get(ep, {}):
            continue
        raw = {rename.get(k, k): (v[None] if torch.is_tensor(v) else [v]) for k, v in item.items()}
        batch = pre(raw)
        d = store.read(ep)
        traces = candidates(d, fi)
        gains = score(policy, batch, traces, tok, policy.config.cot_max_tokens)
        best = 1 if args.bootstrap else int(np.argmax(gains))
        keep = args.bootstrap or gains[best] > 0
        results[ep].append({"segment": wanted[ep][fi], "frame": fi, "trace": traces[best] if keep else None,
                            "gain": float(gains[best]), "all_gains": dict(zip(traces, map(float, gains), strict=True))})
        print(json.dumps({"episode": ep, **results[ep][-1]}), flush=True)
    for ep, rows in results.items():
        (store.dir / f"cot_{ep:06d}.json").write_text(json.dumps(sorted(rows, key=lambda r: r["frame"]), indent=1))


if __name__ == "__main__":
    main()
