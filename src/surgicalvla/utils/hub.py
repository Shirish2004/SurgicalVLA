"""Hugging Face Hub names and uploads for trained models.

Every model is its own repository ``<namespace>/surgical-vla-<model-name>`` (the ``hub:`` section
of each config), so ``lerobot-eval --policy.path=<repo>`` and ``from_pretrained`` load it
directly. All of them are also gathered in one Hub collection named ``surgical-vla``.

``<model-name>`` is ``hub.model_name`` when set (``{benchmark}``, ``{stage}``, ``{seed}``,
``{variant}``, ``{mode}`` are filled in), otherwise it is derived from the run's settings, e.g.

    libero-sft-upcycled-s0                  SFT, all switches off
    libero-sft-matched-aux-lcot-spatial-s1  SFT on one suite with auxiliary heads + latent CoT
    libero-rl-joint-libero10-s0             RL (joint SFT+RL) on libero_10
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

PREFIX = "surgical-vla"
SKIP = ["trainer_state.pt", "rl_state.pt", "*.tmp"]  # optimizer / resume state stays local


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9.-]+", "-", text.lower())).strip("-.")


def _suite_tag(cfg: dict) -> str:
    eps = (cfg.get("dataset") or {}).get("episodes")
    if isinstance(eps, str):  # configs/data/libero_<suite>_episodes.json
        stem = Path(eps).stem.removesuffix("_episodes")
        return stem.removeprefix("libero_libero_").removeprefix("libero_")
    return ""


def model_name(cfg: dict, stage: str) -> str:
    """``<model-name>`` of a run (without the ``surgical-vla-`` prefix)."""
    hub = cfg.get("hub") or {}
    benchmark = cfg.get("benchmark") or hub.get("benchmark") or "model"
    seed = int((cfg.get("run") or {}).get("seed", 0))
    m = cfg.get("smolvla_moe") or {}
    variant = m.get("moe_variant", "upcycled")
    mode = (cfg.get("training") or {}).get("mode", "sequential")
    if hub.get("model_name"):
        return _slug(str(hub["model_name"]).format(benchmark=benchmark, stage=stage, seed=seed, variant=variant, mode=mode))
    if stage == "rl":
        suite = str((cfg.get("env") or {}).get("suite") or "")
        parts = [benchmark, "rl", mode, suite.replace("_", "") if suite and "," not in suite else "all"]
    else:
        policy = (cfg.get("model") or {}).get("policy_type", "smolvla_moe")
        parts = [benchmark, "sft", variant if policy == "smolvla_moe" else policy]
        aux = cfg.get("aux") or {}
        if m.get("aux_vision"):
            parts.append("aux")
        if m.get("latent_cot"):
            parts.append("lcot")
        if m.get("use_cot") and aux.get("cot_traces"):
            parts.append("cot")
        if int(m.get("cot_curriculum_steps") or 0) > 0:
            parts.append("cur")
        if _suite_tag(cfg):
            parts.append(_suite_tag(cfg))
    return _slug("-".join([*parts, f"s{seed}"]))


def repo_id(cfg: dict, stage: str) -> str | None:
    """``<namespace>/surgical-vla-<model-name>``, or None when ``hub.push`` is off."""
    hub = cfg.get("hub") or {}
    if not hub.get("push"):
        return None
    name = f"{PREFIX}-{model_name(cfg, stage)}"[:96]
    return f"{hub.get('namespace', 'Shirish24')}/{name}"


def model_card(repo: str, cfg: dict, stage: str, step: int | None) -> str:
    benchmark = cfg.get("benchmark") or "unknown"
    data = (cfg.get("demo") or {}).get("repo_id") if stage == "rl" else (cfg.get("dataset") or {}).get("repo_id")
    lines = [
        "---",
        "library_name: lerobot",
        "pipeline_tag: robotics",
        "base_model: lerobot/smolvla_base",
        f"tags: [robotics, vla, lerobot, smolvla, {PREFIX}, {benchmark}, {stage}]",
        *([f"datasets: [{data}]"] if data else []),
        "---",
        f"# {repo.split('/')[-1]}",
        "",
        f"SurgicalVLA ({stage.upper()}) for **{benchmark}**"
        + (f", step {step}" if step is not None else "") + ". Trained with the SurgicalVLA repository;",
        "the exact configuration is in `train_config.yaml`.",
        "",
        "Evaluate:",
        "```bash",
        f"python scripts/benchmark.py eval --benchmark {benchmark} --checkpoint {repo}",
        "```",
        "",
        "Results are reported in the repository's `docs/baselines.md`; this card carries no numbers.",
    ]
    return "\n".join(lines) + "\n"


def push_checkpoint(local_dir: str | Path, cfg: dict, stage: str, step: int | None = None) -> str | None:
    """Upload a saved checkpoint (weights, config, processors) as one commit. Returns the repo id.

    Failures are reported as warnings: training results are never lost to a network error;
    re-run ``scripts/push_to_hub.py`` later.
    """
    repo = repo_id(cfg, stage)
    if repo is None:
        return None
    hub = cfg.get("hub") or {}
    local_dir = Path(local_dir)
    try:
        import yaml
        from huggingface_hub import HfApi

        (local_dir / "train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (local_dir / "README.md").write_text(model_card(repo, cfg, stage, step))
        api = HfApi()
        api.create_repo(repo, repo_type="model", private=bool(hub.get("private", True)), exist_ok=True)
        api.upload_folder(folder_path=str(local_dir), repo_id=repo, ignore_patterns=SKIP,
                          commit_message=f"{stage} {local_dir.name}" + (f" (step {step})" if step is not None else ""))
        if hub.get("collection", PREFIX):
            try:
                ns = repo.split("/")[0]
                col = api.create_collection(title=hub.get("collection", PREFIX), namespace=ns,
                                            private=bool(hub.get("private", True)), exists_ok=True)
                api.add_collection_item(col.slug, item_id=repo, item_type="model", exists_ok=True)
            except Exception as exc:  # collection is a convenience; the model is already up
                warnings.warn(f"could not add {repo} to the collection: {exc}", stacklevel=2)
        print(f"pushed {local_dir} -> https://huggingface.co/{repo}", flush=True)
        return repo
    except Exception as exc:
        warnings.warn(f"push of {local_dir} to {repo} failed: {exc}. Retry: "
                      f"python scripts/push_to_hub.py {local_dir} --config <config> --stage {stage}", stacklevel=2)
        return None
