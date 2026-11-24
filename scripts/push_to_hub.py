"""Upload a saved checkpoint to the Hub under the name its config gives it.

    pixi run -e dev python scripts/push_to_hub.py results/libero_train/checkpoints/step_50000 \
        --config configs/benchmarks/libero/sft.yaml --stage sft [--model-name libero-sft-allsuites-s0] [--dry-run]

The repository is ``<hub.namespace>/surgical-vla-<model-name>`` (``src/surgicalvla/utils/hub.py``);
optimizer and resume state are not uploaded. Training pushes its final checkpoint by itself when
``hub.push: true``; this script is for earlier checkpoints, retries and runs made before that.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from surgicalvla.utils.hub import push_checkpoint, repo_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="directory holding model.safetensors, config.json and the processors")
    ap.add_argument("--config", required=True, help="the config the run was trained with")
    ap.add_argument("--stage", choices=["sft", "rl"], required=True)
    ap.add_argument("--model-name", help="override hub.model_name")
    ap.add_argument("--public", action="store_true", help="create the repository public (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="print the repository id only")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    hub = dict(cfg.get("hub") or {})
    hub["push"] = True
    if args.model_name:
        hub["model_name"] = args.model_name
    if args.public:
        hub["private"] = False
    cfg["hub"] = hub
    m = re.search(r"step_0*(\d+)", args.checkpoint)
    repo = repo_id(cfg, args.stage)
    print(repo)
    if not args.dry_run:
        push_checkpoint(args.checkpoint, cfg, args.stage, int(m.group(1)) if m else None)


if __name__ == "__main__":
    main()
