"""Run provenance: everything needed to tell two runs apart, written once at the start.

``collect(...)`` returns a JSON-serialisable dict:

* code: git commit, whether the tree is dirty and a hash of the diff, pixi lockfile hash;
* config: the fully resolved training config and its hash; seeds;
* hardware: GPU name and memory, driver, CPU, RAM, torch / CUDA / transformers versions;
* data: dataset id, episode count, feature-cache name and config hash;
* model: parameters per top-level module (total and trainable) and the measured forward
  FLOPs per sample (``torch.utils.flop_counter``) when a batch is given.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]


def _sh(*cmd: str) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_info() -> dict:
    diff = _sh("git", "diff", "HEAD") or ""
    lock = ROOT / "pixi.lock"
    return {"git_commit": _sh("git", "rev-parse", "HEAD"), "git_branch": _sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(diff), "git_diff_sha256": _sha(diff.encode()) if diff else None,
            "pixi_lock_sha256": _sha(lock.read_bytes()) if lock.exists() else None}


def hardware_info() -> dict:
    info = {"host": platform.node(), "cpu": platform.processor() or platform.machine(), "cpus": os.cpu_count(),
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 1024**3, 1),
                    driver=_sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"))
    return info


def parameter_counts(model: torch.nn.Module, depth: int = 2) -> dict:
    """Total and trainable parameters per module prefix (``depth`` name components)."""
    total, train = Counter(), Counter()
    for name, p in model.named_parameters():
        parts = name.split(".")
        key = ".".join(parts[: depth + (2 if "vlm_with_expert" in parts[:depth] else 0)])
        total[key] += p.numel()
        if p.requires_grad:
            train[key] += p.numel()
    return {"total": sum(total.values()), "trainable": sum(train.values()),
            "per_module": {k: {"total": total[k], "trainable": train[k]} for k in sorted(total, key=lambda k: -total[k])}}


def forward_flops(policy, batch: dict) -> int | None:
    """Measured FLOPs of one training forward pass on ``batch`` (None if the counter fails)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            policy.forward(batch)
        return int(counter.get_total_flops())
    except Exception:
        return None


def collect(cfg: dict, *, policy=None, batch: dict | None = None, seeds: dict | None = None, data: dict | None = None) -> dict:
    out = {"code": code_info(), "hardware": hardware_info(), "seeds": seeds or {}, "data": data or {},
           "config": cfg, "config_sha256": _sha(json.dumps(cfg, sort_keys=True, default=str).encode())}
    if policy is not None:
        out["parameters"] = parameter_counts(policy)
        if batch is not None:
            bsz = next((v.shape[0] for v in batch.values() if torch.is_tensor(v) and v.dim() > 0), 1)
            flops = forward_flops(policy, batch)
            out["flops"] = {"forward_per_batch": flops, "batch_size": int(bsz),
                            "forward_per_sample": None if flops is None else flops // max(int(bsz), 1),
                            "train_step_per_sample_estimate": None if flops is None else 3 * flops // max(int(bsz), 1)}
    return out


def write(info: dict, output_dir: str) -> Path:
    path = Path(output_dir) / "run_info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str))
    return path


__all__ = ["code_info", "collect", "forward_flops", "hardware_info", "parameter_counts", "write"]
