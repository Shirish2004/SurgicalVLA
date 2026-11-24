"""CLI entry point: python train.py --config train.yaml --repo_id <repo> [model.chunk_size=25 ...]"""

import argparse
import dataclasses
import json
import os
import random
import re
import warnings
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from surgicalvla.data.dataset import build_dataset, infer_features, resolve_camera_map
from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig
from surgicalvla.modules.modeling_smolvla_moe import SmolVLAMoEPolicy
from surgicalvla.modules.policy import SurgicalVLAConfig, SurgicalVLAPolicy
from surgicalvla.training.trainer import Trainer
from surgicalvla.utils.config import load_config, warn_unknown
from surgicalvla.utils.splits import block_episodes, load_split, save_split, split_episodes

# yaml key -> SurgicalVLAConfig field, for names that don't match 1:1
MOE_ALIASES = {
    "moe_router_z_loss_weight": "moe_z_coef",
    "moe_top_k": "moe_max_k",
}
NOT_MODEL_FIELDS = {"policy_type"}


def build_config(cfg: dict) -> SurgicalVLAConfig:
    valid_fields = {f.name for f in dataclasses.fields(SurgicalVLAConfig)}
    model_cfg = {k: v for k, v in cfg.get("model", {}).items() if k not in NOT_MODEL_FIELDS}
    warn_unknown([k for k in model_cfg if k not in valid_fields], valid_fields, "model")
    kwargs = {k: v for k, v in model_cfg.items() if k in valid_fields}
    for raw_key, v in cfg.get("moe", {}).items():
        k = MOE_ALIASES.get(raw_key, raw_key)
        if k in valid_fields:
            kwargs[k] = v
        else:
            warnings.warn(f"Unknown moe config key '{raw_key}', ignored.", stacklevel=2)
    if "gripper_dims" not in kwargs and "gripper_dims" in cfg.get("run", {}):
        kwargs["gripper_dims"] = cfg["run"]["gripper_dims"]
    return SurgicalVLAConfig(**kwargs)


def resolve_features(config: SurgicalVLAConfig, repo_id: str, camera_map: dict | None, root: str | None = None):
    """Fills config.input_features/output_features from the dataset if not already set (e.g. by a
    pretrained checkpoint), renaming dataset camera keys to the config's expected names first."""
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    key_map = resolve_camera_map(meta, camera_map)
    if config.input_features:
        return  # pretrained/resumed config already defines the expected feature names
    features = {key_map.get(k, k): v for k, v in infer_features(meta).items()}
    config.input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    config.output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}


def build_moe_config(cfg: dict) -> SmolVLAMoEConfig:
    valid = {f.name for f in dataclasses.fields(SmolVLAMoEConfig)}
    model_cfg = {k: v for k, v in cfg.get("model", {}).items() if k not in NOT_MODEL_FIELDS}
    warn_unknown([k for k in model_cfg if k not in valid], valid, "model")
    kwargs = {k: v for k, v in model_cfg.items() if k in valid}
    for key, value in cfg.get("smolvla_moe", {}).items():
        if key in valid:
            kwargs[key] = value
        else:
            warnings.warn(f"Unknown smolvla_moe config key '{key}', ignored.", stacklevel=2)
    if "gripper_dims" not in kwargs and "gripper_dims" in cfg.get("run", {}):
        kwargs["gripper_dims"] = cfg["run"]["gripper_dims"]
    return SmolVLAMoEConfig(**kwargs)


def checkpoint_type(path: str | None) -> str | None:
    if not path or not os.path.isdir(path):
        return None
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        return json.load(f).get("type")


def resolve_resume_dir(resume_arg: str | bool | None, output_dir: str) -> str | None:
    if not resume_arg:
        return None
    if isinstance(resume_arg, str) and os.path.isdir(resume_arg):
        return resume_arg
    
    ckpt_base = os.path.join(output_dir, "checkpoints") if os.path.exists(os.path.join(output_dir, "checkpoints")) else output_dir
    if os.path.isdir(ckpt_base):
        steps = []
        for name in os.listdir(ckpt_base):
            p = os.path.join(ckpt_base, name)
            if os.path.isdir(p):
                m = re.search(r"step_(\d+)$", name)
                if m:
                    steps.append((int(m.group(1)), p))
        if steps:
            steps.sort(key=lambda x: x[0])
            return steps[-1][1]
    return resume_arg if isinstance(resume_arg, str) else None


def cache_expectations(cfg: dict, config) -> dict:
    """Manifest fields the run requires: ``aux.expect`` from the config, plus what the model needs."""
    aux = cfg.get("aux") or {}
    expect = {"labeler": dict((aux.get("expect") or {}).get("labeler") or {})}
    expect.update({k: v for k, v in (aux.get("expect") or {}).items() if k != "labeler"})
    if getattr(config, "object_input", "none") != "none":
        expect["labeler"]["causal_inputs"] = True
        expect["labeler"].setdefault("object_inputs", {}).update(num_candidates=config.obj_candidates, num_grippers=config.obj_grippers)
    return expect


def build_aux_targets(cfg: dict, config, repo_id: str, tokenizer=None):
    """Cache reader when a switch needs cached data (auxiliary targets, traces or object inputs)."""
    aux = cfg.get("aux") or {}
    needs = getattr(config, "aux_vision", False) or aux.get("cot_traces") or getattr(config, "object_input", "none") != "none"
    if not needs:
        return None
    if not aux.get("sidecar_root"):
        raise ValueError("a switch needs the feature cache but aux.sidecar_root is not set")
    from surgicalvla.pseudolabels.reader import SidecarTargets

    reader = SidecarTargets(
        aux["sidecar_root"], repo_id, name=aux.get("cache_name"), expect=cache_expectations(cfg, config),
        feat_dim=config.aux_feat_dim, track_horizon=int(aux.get("track_horizon", 5)), num_phases=config.num_phases,
        tokenizer=tokenizer if aux.get("cot_traces") else None, cot_max_tokens=config.cot_max_tokens,
        object_input=getattr(config, "object_input", "none"), upload_repo=aux.get("upload_repo"),
    )
    if getattr(config, "object_input", "none") != "none":
        width = reader.manifest.get("columns", {}).get("in_obj_feat", [None, None])[-1]
        if width is not None and int(width) != int(config.obj_sam_feat_dim):
            raise ValueError(f"cached object features are {width}-D but the model expects {config.obj_sam_feat_dim}")
    return reader


def build_label_viz(cfg: dict, config, repo_id: str, camera_map: dict | None):
    """Sidecar reader for drawing pseudo-labels on logged frames, and the policy camera they belong to.

    Active whenever ``aux.sidecar_root`` is set, also with every auxiliary switch off: it only
    draws images and never adds training targets.
    """
    aux = cfg.get("aux") or {}
    if not aux.get("sidecar_root"):
        return None, None
    from surgicalvla.pseudolabels.reader import SidecarTargets

    try:
        reader = SidecarTargets(aux["sidecar_root"], repo_id, name=aux.get("cache_name"), feat_dim=getattr(config, "aux_feat_dim", 384),
                                track_horizon=int(aux.get("track_horizon", 5)), num_phases=getattr(config, "num_phases", 4))
    except FileNotFoundError:
        return None, None
    labelled = (reader.manifest.get("labeler") or {}).get("camera")
    keys = list(config.image_features)
    key = (camera_map or {}).get(labelled) if labelled else None
    return reader, key or (keys[0] if keys else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo_id", required=True, help="dataset identifier (any string for local-only data)")
    parser.add_argument("--root", default=None, help="local dataset root, e.g. a lerobot dataset directory on disk")
    parser.add_argument("--pretrained", default=None, help="pretrained SurgicalVLA checkpoint to finetune from")
    parser.add_argument("--resume", default=None, help="checkpoint dir (checkpoints/step_N) or 'latest' to continue training from")
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. model.chunk_size=25")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    seed = int(cfg.get("run", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds_cfg = cfg.get("dataset", {})
    run_cfg = cfg.get("run", {})
    proj_cfg = cfg.get("project", {})
    log_cfg = cfg.get("logging", {})
    root = ds_cfg.get("root", args.root)
    camera_map = ds_cfg.get("camera_map")

    policy_type = cfg.get("model", {}).get("policy_type", "surgical")
    config = build_moe_config(cfg) if policy_type == "smolvla_moe" else build_config(cfg)
    resolve_features(config, args.repo_id, camera_map, root=root)

    output_dir = run_cfg.get("ckpt_dir", "./checkpoints")
    resume_arg = args.resume or run_cfg.get("resume")
    pretrained = args.pretrained or run_cfg.get("pretrained")

    resume_dir = resolve_resume_dir(resume_arg, output_dir)
    ckpt = resume_dir or pretrained

    if policy_type == "smolvla_moe":
        if ckpt and checkpoint_type(ckpt) == "smolvla_moe":
            policy = SmolVLAMoEPolicy.from_pretrained(ckpt, config=config)
        elif ckpt:
            policy = SmolVLAMoEPolicy.from_smolvla_pretrained(ckpt, config)
        else:
            policy = SmolVLAMoEPolicy(config)
    else:
        policy = SurgicalVLAPolicy.from_pretrained(ckpt, config=config) if ckpt else SurgicalVLAPolicy(config)

    streaming = run_cfg.get("streaming", False)
    val_fraction = ds_cfg.get("val_fraction", 0.01)
    train_eps = val_eps = None
    if streaming and val_fraction > 0:
        warnings.warn("Streaming mode has no episode filtering; validation is disabled.", stacklevel=2)
    else:
        blocks = ds_cfg.get("episode_blocks")
        pool = None
        if ds_cfg.get("episodes"):  # explicit list, or a JSON file holding one (e.g. one LIBERO suite)
            eps = ds_cfg["episodes"]
            pool = json.loads(Path(eps).read_text()) if isinstance(eps, str) else [int(e) for e in eps]
        elif blocks:  # e.g. RoboTwin demo_clean: first 50 of every 550-episode task block
            meta = LeRobotDatasetMetadata(args.repo_id, root=root)
            pool = block_episodes(meta.total_episodes, int(blocks["block"]), int(blocks["take"]))
        split_path = os.path.join(output_dir, "split.json")
        if val_fraction > 0 and os.path.exists(split_path):
            train_eps, val_eps = load_split(split_path)
        elif val_fraction > 0:
            meta = LeRobotDatasetMetadata(args.repo_id, root=root)
            train_eps, val_eps = split_episodes(meta.total_episodes, val_fraction, ds_cfg.get("split_seed", 0), pool=pool)
            save_split(split_path, train_eps, val_eps)
        else:
            train_eps = pool

    dataset = build_dataset(
        args.repo_id,
        config,
        streaming=streaming,
        streaming_buffer_size=run_cfg.get("streaming_buffer_size", 1000),
        streaming_seed=run_cfg.get("streaming_seed", 42),
        root=root,
        episodes=train_eps,
        camera_map=camera_map,
    )
    val_dataset = (
        build_dataset(args.repo_id, config, root=root, episodes=val_eps, camera_map=camera_map) if val_eps else None
    )

    # Inspect checkpoint state for WandB run ID and dataset verification
    wandb_run_id = None
    if resume_dir and os.path.exists(resume_dir):
        txt_path = os.path.join(resume_dir, "wandb_run_id.txt")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                wandb_run_id = f.read().strip()

        state_path = os.path.join(resume_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            try:
                st = torch.load(state_path, map_location="cpu", weights_only=False)
                if not wandb_run_id and st.get("wandb_run_id"):
                    wandb_run_id = st.get("wandb_run_id")
                saved_repo_id = st.get("repo_id")
                if saved_repo_id and saved_repo_id != args.repo_id:
                    warnings.warn(
                        f"Resume checkpoint was trained on dataset '{saved_repo_id}', but current dataset is '{args.repo_id}'.",
                        stacklevel=2,
                    )
            except Exception as e:
                warnings.warn(f"Could not inspect checkpoint state: {e}", stacklevel=2)

    # checkpoints written before the run id was saved: pass it as project.wandb_id=<id>
    wandb_run_id = wandb_run_id or proj_cfg.get("wandb_id")
    wandb_mode = proj_cfg.get("wandb_mode", "disabled")
    logger_kwargs = dict(
        use_wandb=wandb_mode != "disabled",
        project=proj_cfg.get("wandb_project", "surgicalvla"),
    )
    if wandb_run_id:
        logger_kwargs["id"] = wandb_run_id
        logger_kwargs["resume"] = proj_cfg.get("wandb_resume", "allow")

    for k in ["entity", "run_name", "tags", "notes", "mode", "group", "job_type", "dir", "id", "resume"]:
        if k in proj_cfg and k not in logger_kwargs:
            logger_kwargs[k] = proj_cfg[k]
    for k in ["entity", "run_name", "tags", "notes", "mode"]:
        if k in log_cfg and k not in logger_kwargs:
            logger_kwargs[k] = log_cfg[k]

    aux_targets = build_aux_targets(cfg, config, args.repo_id,
                                    tokenizer=getattr(getattr(policy, "model", None), "_tokenizer", lambda: None)())
    label_viz, label_image_key = build_label_viz(cfg, config, args.repo_id, camera_map)
    from surgicalvla.utils.hub import push_checkpoint

    hub_cfg = {**cfg, "dataset": {**ds_cfg, "repo_id": args.repo_id}}
    push_fn = (lambda path, step: push_checkpoint(path, hub_cfg, "sft", step)) if (cfg.get("hub") or {}).get("push") else None
    trainer = Trainer(
        policy,
        dataset,
        output_dir=output_dir,
        batch_size=run_cfg.get("batch_size", 32),
        steps=run_cfg.get("steps", 100_000),
        log_every=proj_cfg.get("log_every", 50),
        eval_every=proj_cfg.get("val_every", 2_000),
        save_every=proj_cfg.get("save_every", 2_000),
        device=run_cfg.get("device", "cuda"),
        visual_every=log_cfg.get("log_every_n_steps"),
        visual_logging=log_cfg.get("enable_visual_logging", True),
        logger_kwargs=logger_kwargs,
        rebase_skills=bool(pretrained) and not resume_dir,
        use_amp=run_cfg.get("amp", True),
        val_dataset=val_dataset,
        val_samples=run_cfg.get("val_samples", 640),
        ema_decay=run_cfg.get("ema_decay", 0.999),
        resume_from=resume_dir,
        grad_accum=run_cfg.get("grad_accum", 4),
        repo_id=args.repo_id,
        seed=seed,
        label_viz=label_viz,
        push_fn=push_fn,
        push_every_save=bool((cfg.get("hub") or {}).get("push_every_save", False)),
        grad_diag_every=int(log_cfg.get("grad_diag_every", 0)),
        run_info={"config": cfg, "seeds": {"run.seed": seed, "split_seed": ds_cfg.get("split_seed", 0)},
                  "data": {"repo_id": args.repo_id, "train_episodes": len(train_eps) if train_eps is not None else "all",
                           "val_episodes": len(val_eps) if val_eps else 0,
                           "cache": None if aux_targets is None else {"dir": str(aux_targets.cache.dir),
                                                                    "config_hash": aux_targets.manifest.get("config_hash")}}},
        label_image_key=label_image_key,
        aux_targets=aux_targets,
    )
    trainer.fit()


if __name__ == "__main__":
    main()