"""Read-only instrumentation of a ``smolvla_moe`` policy during ``lerobot-eval`` rollouts.

Enabled by ``scripts/lerobot_eval.py`` when ``SVLA_EVAL_INSTRUMENT=1``. Nothing here changes
an action: every hook calls the original method and only observes inputs/outputs, so success
rates stay comparable with uninstrumented runs. The optional CoT probe runs an *extra*
``generate_cot`` pass on the same prefix and discards it (latent-only inference is unchanged).

Per rollout (one ``policy.reset()`` = one batch of parallel environments) it records, per
environment row: success, length, live-perception statistics of every chunk (valid target
candidates, SAM 3 scores, gripper found), probed CoT text; per rollout: perception / action-chunk /
CoT-probe latency and MoE expert usage. Written to ``<output_dir>/instrument/metrics.jsonl``.
Annotated videos (policy camera, SAM 3 boxes, target phrase, CoT text) of the first rollout of
the first ``SVLA_ANNOTATE_TASKS`` tasks go to ``<output_dir>/instrument/annotated/``.

Knobs (environment): ``SVLA_COT_PROBE_EVERY`` (probe every k-th chunk, 0 = off; default 4),
``SVLA_ANNOTATE_TASKS`` (default 2), ``SVLA_VIDEO_FPS`` (default 20).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _tasks(batch: dict, n: int) -> list[str]:
    t = batch.get("task") or [""] * n
    return [t] * n if isinstance(t, str) else [str(x) for x in t]


class EvalInstrument:
    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir) / "instrument"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "metrics.jsonl"
        self.cot_every = int(os.environ.get("SVLA_COT_PROBE_EVERY", "4"))
        self.annotate_tasks = int(os.environ.get("SVLA_ANNOTATE_TASKS", "2"))
        self.fps = int(os.environ.get("SVLA_VIDEO_FPS", "20"))
        self.lock = threading.Lock()
        self.rollout = -1
        self.annotated: set[str] = set()
        self._new_rollout()

    # ------------------------------------------------------------------ state
    def _new_rollout(self) -> None:
        self.chunks: list[dict] = []        # per chunk: per-row perception stats
        self.cot: list[list[str]] = []      # per probe: per-row CoT text
        self.t_perc: list[float] = []
        self.t_chunk: list[float] = []
        self.t_cot: list[float] = []
        self.usage: list[np.ndarray] = []
        self.chunk_ix = 0
        self.task_names: list[str] | None = None
        self.frames: list[np.ndarray] | None = None
        self.last_boxes: np.ndarray | None = None
        self.last_valid: np.ndarray | None = None
        self.last_cot = ""
        self.inference_cot_active = False

    # ------------------------------------------------------------------ attach
    def attach(self, policy) -> None:
        inst = self
        orig_reset, orig_live, orig_chunk = policy.reset, policy._live_inputs, policy._get_action_chunk
        orig_select, orig_sample = policy.select_action, policy.model.sample_actions
        k_targets = int(getattr(policy.config, "obj_candidates", 3))

        def reset(*a, **kw):
            with inst.lock:
                inst.rollout += 1
                inst._new_rollout()
            return orig_reset(*a, **kw)

        def live(batch):
            _sync(); t0 = time.perf_counter()
            out = orig_live(batch)
            _sync(); dt = time.perf_counter() - t0
            try:
                inst._record_perception(policy, batch, out, dt, k_targets)
            except Exception as e:  # observation only: never break the rollout
                print(f"[instrument] perception stats failed: {type(e).__name__}: {e}", flush=True)
            return out

        def sample(images, img_masks, lang, lang_mask, state, *a, objects=None, **kw):
            probe = inst.cot_every > 0 and inst.chunk_ix % inst.cot_every == 0 and objects is not None
            if probe:
                _sync(); t0 = time.perf_counter()
                texts = inst._probe_cot(policy, images, img_masks, lang, lang_mask, state, objects)
                _sync()
                with inst.lock:
                    inst.t_cot.append(time.perf_counter() - t0)
                    inst.cot.append(texts)
                    inst.last_cot = texts[0] if texts else ""
            out = orig_sample(images, img_masks, lang, lang_mask, state, *a, objects=objects, **kw)
            with inst.lock:
                inst.inference_cot_active |= policy.model._cot_cache is not None
            return out

        def chunk(batch, *a, **kw):
            _sync(); t0 = time.perf_counter()
            out = orig_chunk(batch, *a, **kw)
            _sync(); dt = time.perf_counter() - t0
            try:
                use = policy.expert_usage()
                use = torch.stack([u.flatten() for u in use.values()]).mean(0).numpy() if use else None
            except Exception:  # layers with different expert counts, or no MoE: skip usage
                use = None
            with inst.lock:
                inst.t_chunk.append(dt)
                inst.chunk_ix += 1
                if use is not None:
                    inst.usage.append(use)
            return out

        def select(batch, *a, **kw):
            out = orig_select(batch, *a, **kw)
            try:
                inst._grab_frame(policy, batch)
            except Exception as e:
                print(f"[instrument] frame grab failed: {type(e).__name__}: {e}", flush=True)
                inst.frames = None
            return out

        policy.reset, policy._live_inputs, policy._get_action_chunk = reset, live, chunk
        policy.select_action, policy.model.sample_actions = select, sample

    # ------------------------------------------------------------------ perception stats
    def _record_perception(self, policy, batch, out, dt: float, k_targets: int) -> None:
        valid = out["obj_valid_mask"].bool().cpu().numpy()
        pose = out["obj_pose_vec"].float().cpu().numpy()  # [x1, y1, x2, y2, cx, cy, depth, score]
        score = np.where(valid, pose[..., 7], np.nan)
        tv = valid[:, :k_targets]
        n = valid.shape[0]
        tasks = _tasks(batch, n)
        phrases = getattr(policy, "_phrases", {})
        with self.lock:
            self.t_perc.append(dt)
            self.task_names = tasks
            self.chunks.append({
                "n_targets": tv.sum(1).tolist(),
                "target_found": tv.any(1).tolist(),
                "gripper_found": valid[:, k_targets:].any(1).tolist() if valid.shape[1] > k_targets else [False] * n,
                "top_score": [float(np.nanmax(s)) if np.isfinite(s).any() else 0.0 for s in score[:, :k_targets]],
                "phrase": phrases.get(tasks[0], ""),
            })
            self.last_boxes, self.last_valid = pose[0, :, :4], valid[0]

    # ------------------------------------------------------------------ CoT probe
    @torch.no_grad()
    def _probe_cot(self, policy, images, img_masks, lang, lang_mask, state, objects) -> list[str]:
        model = policy.model
        if not model.config.use_cot or not model.cot_inference_enabled:
            return []
        try:
            gen = model.generate_cot(images, img_masks, lang, lang_mask, state, objects)
        except Exception as e:  # a probe must never break the evaluation
            return [f"<probe failed: {type(e).__name__}: {e}>"]
        tok = model._tokenizer()
        if gen is None or tok is None:
            return []
        ids = gen[0][:, len(model._cot_prompt_ids):]
        return [s.strip() for s in tok.batch_decode(ids, skip_special_tokens=True)]

    # ------------------------------------------------------------------ annotated video
    def _grab_frame(self, policy, batch) -> None:
        task = (_tasks(batch, 1) or [""])[0]
        if self.frames is None:
            if len(self.annotated) >= self.annotate_tasks or task in self.annotated:
                return
            self.annotated.add(task)
            self.frames = []
            self.video_task = task
        cam = getattr(policy, "_live_camera", None)
        if cam is None or cam not in batch:
            return
        img = batch[cam][0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        frame = (img * 255).round().astype(np.uint8)
        self.frames.append(self._draw(frame, task))

    def _draw(self, frame: np.ndarray, task: str) -> np.ndarray:
        from PIL import Image, ImageDraw

        scale = max(1, 384 // frame.shape[0]) if frame.shape[0] < 384 else 1
        im = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale))
        w, h = im.size
        pad = 64
        canvas = Image.new("RGB", (w, h + pad), "white")
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        if self.last_boxes is not None:
            for j, (box, ok) in enumerate(zip(self.last_boxes, self.last_valid, strict=True)):
                if not ok:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                color = (230, 60, 60) if j < len(self.last_valid) - 1 else (40, 120, 230)
                d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=color, width=2)
        phrase = self.chunks[-1]["phrase"] if self.chunks else ""
        d.text((4, h + 2), f"task: {task[:70]}", fill="black")
        d.text((4, h + 18), f"target: {phrase}   (red: SAM 3 targets, blue: gripper)", fill="black")
        d.text((4, h + 34), f"CoT: {self.last_cot[:80]}", fill=(90, 90, 90))
        return np.asarray(canvas)

    # ------------------------------------------------------------------ end of rollout
    def end_rollout(self, out: dict) -> None:
        done = out["done"].to(int)
        first = torch.argmax(done, dim=1)
        lengths = torch.where(done.any(dim=1), first + 1, torch.full_like(first, done.shape[1])).tolist()
        success = out["success"].any(dim=1).tolist()
        with self.lock:
            n = len(lengths)
            tasks = self.task_names or [""] * n
            eps = []
            for i in range(n):
                per = [c for c in self.chunks if i < len(c["n_targets"])]
                eps.append({
                    "row": i, "task": tasks[i] if i < len(tasks) else "", "success": bool(success[i]),
                    "length": int(lengths[i]), "n_chunks": len(per),
                    "detect_rate": float(np.mean([c["target_found"][i] for c in per])) if per else None,
                    "mean_targets": float(np.mean([c["n_targets"][i] for c in per])) if per else None,
                    "mean_top_score": float(np.mean([c["top_score"][i] for c in per])) if per else None,
                    "gripper_rate": float(np.mean([c["gripper_found"][i] for c in per])) if per else None,
                    "cot": [p[i] for p in self.cot if i < len(p)],
                })
            rec = {
                "type": "rollout", "rollout": self.rollout, "task": tasks[0] if tasks else "",
                "phrase": self.chunks[0]["phrase"] if self.chunks else "",
                "batch": n,
                "perception_s": self.t_perc, "chunk_s": self.t_chunk, "cot_probe_s": self.t_cot,
                "expert_usage": np.mean(self.usage, 0).tolist() if self.usage else None,
                "inference_cot_active": self.inference_cot_active,
                "episodes": eps,
            }
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if self.frames:
                self._write_video(int(lengths[0]), bool(success[0]))
            self.frames = None

    def _write_video(self, length: int, success: bool) -> None:
        from lerobot.utils.io_utils import write_video

        frames = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in self.frames[: max(1, length)]]  # libx264: even size
        slug = "".join(c if c.isalnum() else "_" for c in self.video_task)[:60].strip("_")
        path = self.dir / "annotated" / f"{slug}_r{self.rollout}_{'success' if success else 'fail'}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_video(path, frames, fps=self.fps)
        except Exception as e:  # a video must never break the evaluation
            print(f"[instrument] annotated video failed: {e}", flush=True)
