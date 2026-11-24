"""Publication-style logging: scalars, images, MoE routing and action trajectories."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter
    except Exception:
        SummaryWriter = None
try:
    from torchvision.utils import make_grid
except Exception:
    def make_grid(*args, **kwargs):
        raise RuntimeError("torchvision required for visual logging")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def truncate_csv(path: str, last_step: int) -> None:
    """Drop rows of a ``step,key,value`` log written after ``last_step``."""
    with open(path) as f:
        lines = f.readlines()
    keep = [ln for ln in lines[1:] if ln.strip() and int(ln.split(",", 1)[0]) <= last_step]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(lines[:1] + keep)
    os.replace(tmp, path)


class ExperimentLogger:
    """Writes scalars/images/figures to TensorBoard and saves vector-graphics copies to disk."""

    def __init__(
        self,
        output_dir: str,
        config=None,
        use_wandb: bool = False,
        project: str = "surgicalvla",
        resume_step: int | None = None,
        **kwargs,
    ):
        """``resume_step``: a run resumed from that step's checkpoint; logged values after it are
        dropped from the CSV and TensorBoard so the resumed run continues one clean curve."""
        self.fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)
        purge = {"purge_step": resume_step + 1} if resume_step is not None else {}
        self.writer = SummaryWriter(os.path.join(output_dir, "logs"), **purge) if SummaryWriter is not None else None
        self._history: dict[str, list] = {}
        # Append-only long-format scalar log. TensorBoard/W&B are the interactive
        # views; this file is what survives when neither is installed or a run
        # dies mid-way, and it is what `plot_metrics` reads.
        self.csv_path = os.path.join(output_dir, "metrics.csv")
        if resume_step is not None and os.path.exists(self.csv_path):
            truncate_csv(self.csv_path, resume_step)
        self._csv_header = not os.path.exists(self.csv_path)

        if config is not None:
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config.__dict__, f, default=str, indent=2)

        self.wandb = None
        self.run_id = None
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
                init_kwargs = dict(project=project, config=getattr(config, "__dict__", {}))
                # pass through common wandb init args if provided
                for k in ["entity", "group", "job_type", "tags", "notes", "dir", "id", "resume", "reinit", "mode"]:
                    if k in kwargs:
                        init_kwargs[k] = kwargs[k]
                if kwargs.get("run_name"):
                    init_kwargs["name"] = kwargs["run_name"]
                run = self.wandb.init(**init_kwargs)
                self.run_id = getattr(run, "id", None)  # saved with checkpoints to resume this run
            except Exception:
                self.wandb = None

    def log_scalars(self, metrics: dict, step: int):
        rows = []
        for k, v in metrics.items():
            v = float(v)
            if self.writer is not None:
                self.writer.add_scalar(k, v, step)
            self._history.setdefault(k, []).append((step, v))
            rows.append(f"{int(step)},{k},{v:.10g}")
        with open(self.csv_path, "a") as f:
            if self._csv_header:
                f.write("step,key,value\n")
                self._csv_header = False
            f.write("\n".join(rows) + "\n")
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def log_images(self, images: torch.Tensor, step: int, tag: str = "images", nrow: int = 4):
        if images.min() < 0:
            images = (images.clamp(-1, 1) + 1) / 2
        grid = make_grid(images, nrow=nrow)
        if self.writer is not None:
            self.writer.add_image(tag, grid, step)
        self._wandb_image(tag, (grid.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8), step)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(f"{tag} — step {step}")
        self._save(fig, f"{tag}_{step}")

    def log_expert_usage(self, usage: dict, step: int, tag: str = "moe/expert_usage", heatmap: bool = True):
        """Routing fractions as scalars every call; the heatmap figure only when ``heatmap``."""
        for layer_idx, counts in usage.items():
            frac = counts / counts.sum().clamp_min(1)
            for e, val in enumerate(frac.tolist()):
                if self.writer is not None:
                    self.writer.add_scalar(f"{tag}/layer{layer_idx}/expert{e}", val, step)
        if heatmap:
            self._plot_expert_heatmap(usage, step)

    def _plot_expert_heatmap(self, usage: dict, step: int):
        layers = sorted(usage.keys())
        mat = np.stack([(usage[idx] / usage[idx].sum().clamp_min(1)).numpy() for idx in layers])
        fig, ax = plt.subplots(figsize=(1 + mat.shape[1] * 0.6, 1 + mat.shape[0] * 0.4))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("expert"), ax.set_ylabel("expert layer")
        ax.set_xticks(range(mat.shape[1])), ax.set_yticks(range(mat.shape[0]), layers)
        fig.colorbar(im, ax=ax, label="routing fraction")
        ax.set_title(f"Expert utilization — step {step}")
        self._wandb_image("moe/expert_usage_heatmap", fig, step)
        if self.writer is not None:
            self.writer.add_figure("moe/expert_usage_heatmap", fig, step, close=False)
        self._save(fig, f"expert_usage_{step}")

    def log_action_trajectories(self, pred: torch.Tensor, target: torch.Tensor, step: int, dims: int | None = None, name: str = "actions"):
        pred, target = pred.numpy(), target.numpy()
        d = dims or pred.shape[-1]
        fig, axes = plt.subplots(d, 1, figsize=(6, 1.4 * d), sharex=True)
        axes = np.atleast_1d(axes)
        for i in range(d):
            axes[i].plot(target[0, :, i], label="ground truth", lw=1.5)
            axes[i].plot(pred[0, :, i], label="predicted", lw=1.5, ls="--")
            axes[i].set_ylabel(f"a[{i}]")
        axes[0].legend(frameon=False, ncol=2)
        axes[-1].set_xlabel("timestep")
        fig.suptitle(f"Action chunk ({name}) — step {step}")
        fig.tight_layout()
        self._wandb_image(f"rollout/{name}", fig, step)
        if self.writer is not None:
            self.writer.add_figure(f"rollout/{name}", fig, step, close=False)
        self._save(fig, f"{name}_{step}")

    def log_loss_curve(self, key: str = "loss"):
        if key not in self._history:
            return
        steps, values = zip(*self._history[key], strict=False)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(steps, values, lw=1.2)
        ax.set_xlabel("step"), ax.set_ylabel(key), ax.set_title(key)
        self._save(fig, f"curve_{key}")

    def _wandb_image(self, tag: str, image, step: int):
        """``image``: an (H, W, 3) uint8 array or a matplotlib figure."""
        if not self.wandb:
            return
        try:
            self.wandb.log({tag: self.wandb.Image(image)}, step=step)
        except Exception:
            pass

    def _save(self, fig, name: str):
        # fig.savefig(os.path.join(self.fig_dir, f"{name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(self.fig_dir, f"{name}.png"), bbox_inches="tight")
        plt.close(fig)

    def close(self):
        self.log_loss_curve("loss")
        if self.writer is not None:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()