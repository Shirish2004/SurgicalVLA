"""Publication figures and tables comparing the full model, its inference-time probes and SmolVLA.

    pixi run -e sim-eval python scripts/eval_ablation_report.py --root results/libero/spatial_full25k_eval \
        --baseline results/baselines/smolvla_libero --train-run outputs/libero_spatial/full_25k/seed0

``--root`` holds one directory per condition (``full``, ``obj_zero``, ...; see
``scripts/run_spatial_ablation_eval.sh``), each laid out as ``eval_libero_baseline.py run`` writes it.
Every condition runs the same tasks, init states and seeds, so episodes are paired by
(task, init state, seed). Statistics:

* success: mean over seeds; 95% CI from a cluster bootstrap over (task, init state);
* probe effect: paired difference to ``full`` with a paired cluster bootstrap CI and an exact
  McNemar test on the discordant episode pairs.

Writes under ``--root``: ``all_episodes.csv`` (every episode of every condition, with the
instrumented perception fields when present), ``comparison.json``, ``tables/*.tex``,
``figures/*.{pdf,png}`` and ``report.md``. Training-dynamics figures come from the run's
``metrics.csv`` when ``--train-run`` is given. Conditions that have not run yet are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL, DCOL = 3.4, 7.0  # single / double column width (in)
# Okabe-Ito (colour-blind safe)
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7f7f7f", "black": "#000000"}

# condition -> (label, component, colour); order = plotting order
CONDITIONS = {
    "smolvla": ("SmolVLA (released)", "baseline", OI["grey"]),
    "full": ("Ours (full)", "full", OI["blue"]),
    "obj_zero": ("− object tokens", "SAM 3 object tokens", OI["orange"]),
    "obj_shuffle": ("object slots shuffled", "SAM 3 object tokens", OI["yellow"]),
    "moe_shared": ("− routed experts", "phase MoE", OI["green"]),
    "moe_random": ("random routing", "phase MoE", OI["sky"]),
    "latent_drop": ("− latent CoT tokens", "latent CoT", OI["purple"]),
}
SUITE = "libero_spatial"


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- loading
def load_condition(cond_dir: Path) -> list[dict]:
    """Every episode of one condition: seed, task_id, task, episode_ix (= init state), success, length, perception."""
    eps: list[dict] = []
    for info_path in sorted((cond_dir / SUITE).glob("seed*/eval_info.json")):
        seed_dir = info_path.parent
        info = json.loads(info_path.read_text())
        desc_path = seed_dir / "task_descriptions.json"
        desc = json.loads(desc_path.read_text()) if desc_path.exists() else {}
        lengths_path = seed_dir / "episode_lengths.json"
        lengths = json.loads(lengths_path.read_text()) if lengths_path.exists() else None
        per_task = sorted(info["per_task"], key=lambda e: int(e["task_id"]))
        total = sum(len(e["metrics"]["successes"]) for e in info["per_task"])
        if lengths is not None and len(lengths) != total:
            lengths = None
        # episode_lengths.json follows LeRobot's run order (eval_info order), not the sorted order
        length_of: dict[tuple[int, int], int] = {}
        if lengths is not None:
            k = 0
            for e in info["per_task"]:
                for ix in range(len(e["metrics"]["successes"])):
                    length_of[(int(e["task_id"]), ix)] = int(lengths[k])
                    k += 1
        instr = instrument_episodes(seed_dir)
        for e in per_task:
            tid = int(e["task_id"])
            task = desc.get(f"{SUITE}_{tid}", f"task {tid}")
            for ix, s in enumerate(e["metrics"]["successes"]):
                rec = {"seed": int(seed_dir.name.removeprefix("seed")), "task_id": tid, "task": task, "episode_ix": ix,
                       "success": bool(s), "length": length_of.get((tid, ix))}
                rec.update(instr.get((task, ix), {}))
                eps.append(rec)
    return eps


def instrument_episodes(seed_dir: Path) -> dict[tuple[str, int], dict]:
    """(task, episode_ix) -> perception fields from instrument/metrics.jsonl (rollouts run in episode order)."""
    path = seed_dir / "instrument" / "metrics.jsonl"
    if not path.exists():
        return {}
    out, seen = {}, defaultdict(int)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        base = seen[r["task"]]
        for e in r["episodes"]:
            out[(r["task"], base + e["row"])] = {k: e.get(k) for k in ("detect_rate", "mean_targets", "mean_top_score", "gripper_rate")}
        seen[r["task"]] += r["batch"]
    return out


# ----------------------------------------------------------------------------- statistics
def grid(eps: list[dict], seeds: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """(clusters, seeds) success array over the (task, init state) clusters every seed has."""
    by = {(e["task_id"], e["episode_ix"], e["seed"]): e["success"] for e in eps}
    clusters = sorted({(t, i) for t, i, _ in by})
    clusters = [c for c in clusters if all((c[0], c[1], s) in by for s in seeds)]
    return np.array([[by[(c[0], c[1], s)] for s in seeds] for c in clusters], dtype=float), clusters


def bootstrap_ci(per_cluster: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_cluster), size=(n_boot, len(per_cluster)))
    means = per_cluster[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(eps: list[dict]) -> dict:
    seeds = sorted({e["seed"] for e in eps})
    g, clusters = grid(eps, seeds)
    per_seed = g.mean(axis=0)
    lo, hi = bootstrap_ci(g.mean(axis=1))
    per_task = defaultdict(list)
    for e in eps:
        per_task[e["task"]].append(e["success"])
    lens_s = [e["length"] for e in eps if e["success"] and e["length"] is not None]
    return {"seeds": seeds, "n_episodes": int(g.size), "success": float(per_seed.mean()),
            "per_seed": per_seed.tolist(), "std_over_seeds": float(per_seed.std(ddof=1)) if len(seeds) > 1 else 0.0,
            "ci95": [lo, hi], "per_task": {t: float(np.mean(v)) for t, v in per_task.items()},
            "mean_success_length": float(np.mean(lens_s)) if lens_s else None}


def paired(eps_a: list[dict], eps_b: list[dict]) -> dict | None:
    """Effect of B relative to A on the shared (task, init state, seed) episodes."""
    seeds = sorted({e["seed"] for e in eps_a} & {e["seed"] for e in eps_b})
    if not seeds:
        return None
    ga, ca = grid(eps_a, seeds)
    gb, cb = grid(eps_b, seeds)
    common = sorted(set(ca) & set(cb))
    ia, ib = {c: i for i, c in enumerate(ca)}, {c: i for i, c in enumerate(cb)}
    a = ga[[ia[c] for c in common]]
    b = gb[[ib[c] for c in common]]
    diff = (b - a).mean(axis=1)
    lo, hi = bootstrap_ci(diff)
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return {"seeds": seeds, "n_pairs": int(a.size), "delta": float(diff.mean()), "ci95": [lo, hi],
            "a_only_success": only_a, "b_only_success": only_b, "p_mcnemar": mcnemar_exact(only_a, only_b)}


# ----------------------------------------------------------------------------- figures
def short_names(tasks: list[str]) -> dict[str, str]:
    words = [t.split() for t in tasks]
    if len(words) < 2:
        return {t: t for t in tasks}
    pre = 0
    while all(len(w) > pre for w in words) and len({w[pre] for w in words}) == 1:
        pre += 1
    suf = 0
    while all(len(w) > pre + suf for w in words) and len({w[-1 - suf] for w in words}) == 1:
        suf += 1
    return {t: " ".join(w[pre:len(w) - suf]) or t for t, w in zip(tasks, words, strict=True)}


def fig_main(summ: dict, out: Path) -> None:
    names = [c for c in CONDITIONS if c in summ]
    fig, ax = plt.subplots(figsize=(COL, 0.24 * len(names) + 0.55))
    for i, c in enumerate(names):
        s = summ[c]
        lab, _, col = CONDITIONS[c]
        p, (lo, hi) = 100 * s["success"], s["ci95"]
        ax.barh(i, p, color=col, height=0.62, edgecolor="black" if c == "full" else "none", linewidth=0.6)
        ax.errorbar(p, i, xerr=[[p - 100 * lo], [100 * hi - p]], color="black", lw=0.7, capsize=1.6)
        for v in s["per_seed"]:
            ax.plot(100 * v, i, "o", ms=2.2, mfc="white", mec="black", mew=0.5)
        ax.text(min(100 * hi + 1.5, 101), i, f"{p:.1f}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(names)), [CONDITIONS[c][0] for c in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("LIBERO-Spatial success (%)")
    ax.xaxis.grid(True, lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    save(fig, out, "success_by_condition")


def fig_effects(effects: dict, out: Path) -> None:
    names = [c for c in CONDITIONS if c in effects and c not in ("full", "smolvla")]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(COL, 0.24 * len(names) + 0.6))
    for i, c in enumerate(names):
        e = effects[c]
        d, (lo, hi) = 100 * e["delta"], e["ci95"]
        ax.barh(i, d, color=CONDITIONS[c][2], height=0.6)
        ax.errorbar(d, i, xerr=[[d - 100 * lo], [100 * hi - d]], color="black", lw=0.7, capsize=1.6)
        star = "***" if e["p_mcnemar"] < 1e-3 else "**" if e["p_mcnemar"] < 1e-2 else "*" if e["p_mcnemar"] < 0.05 else "n.s."
        x = 100 * (lo if d < 0 else hi)
        ax.text(x + (-1 if d < 0 else 1), i, star, va="center", ha="right" if d < 0 else "left", fontsize=6.5)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(names)), [CONDITIONS[c][0] for c in names])
    ax.invert_yaxis()
    lim = max(10, max(abs(100 * v) for c in names for v in effects[c]["ci95"]) + 8)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Δ success vs. full model (pp), paired 95% CI")
    ax.xaxis.grid(True, lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    save(fig, out, "probe_effects")


def fig_per_task(summ: dict, out: Path) -> None:
    names = [c for c in CONDITIONS if c in summ]
    tasks = sorted(summ["full" if "full" in summ else names[0]]["per_task"])
    short = short_names(tasks)
    mat = np.array([[summ[c]["per_task"].get(t, np.nan) for c in names] for t in tasks])
    fig, ax = plt.subplots(figsize=(DCOL * 0.62, 0.26 * len(tasks) + 1.1))
    im = ax.imshow(100 * mat, cmap="cividis", vmin=0, vmax=100, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = 100 * mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white" if v < 55 else "black")
    ax.set_xticks(range(len(names)), [CONDITIONS[c][0] for c in names], rotation=35, ha="right")
    ax.set_yticks(range(len(tasks)), [short[t] for t in tasks])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="success (%)")
    save(fig, out, "per_task_success")


def fig_time_to_success(eps_by: dict, out: Path) -> None:
    names = [c for c in CONDITIONS if c in eps_by and any(e["length"] is not None for e in eps_by[c])]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    horizon = max(e["length"] for c in names for e in eps_by[c] if e["length"] is not None)
    t = np.arange(0, horizon + 1)
    for c in names:
        eps = [e for e in eps_by[c] if e["length"] is not None]
        done = np.array([e["length"] if e["success"] else np.inf for e in eps])
        frac = (done[None, :] <= t[:, None]).mean(axis=1)
        lab, _, col = CONDITIONS[c]
        ax.plot(t, 100 * frac, color=col, lw=1.4 if c == "full" else 0.9, label=lab,
                ls="-" if c in ("full", "smolvla") else "--")
    ax.set_xlabel("control step")
    ax.set_ylabel("episodes solved (%)")
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.grid(True, lw=0.3, alpha=0.5)
    save(fig, out, "time_to_success")


def fig_perception(eps: list[dict], out: Path) -> dict:
    eps = [e for e in eps if e.get("detect_rate") is not None]
    if not eps:
        return {}
    edges, labels = [0, 0.5, 0.8, 0.95, 1.0001], ["<50", "50–80", "80–95", "≥95"]
    res = {}
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    for i, (lo, hi, lab) in enumerate(zip(edges[:-1], edges[1:], labels, strict=True)):
        b = [e["success"] for e in eps if lo <= e["detect_rate"] < hi]
        if not b:
            continue
        p = float(np.mean(b))
        k, n = sum(b), len(b)
        z = 1.96
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        res[lab] = {"n": n, "success": p, "ci95": [c - h, c + h]}
        ax.bar(i, 100 * p, color=OI["blue"], width=0.6)
        ax.errorbar(i, 100 * p, yerr=[[100 * (p - c + h)], [100 * (c + h - p)]], color="black", lw=0.7, capsize=1.6)
        ax.text(i, 3, f"n={n}", ha="center", fontsize=6, color="white" if p > 0.12 else "black")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("chunks with the target found by live SAM 3 (%)")
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 100)
    save(fig, out, "perception_vs_success")
    return res


# ----------------------------------------------------------------------------- training dynamics
def load_metrics(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    d: dict[str, dict[int, float]] = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                d[r["key"]][int(r["step"])] = float(r["value"])
            except (ValueError, KeyError):
                continue
    return {k: (np.array(sorted(v)), np.array([v[s] for s in sorted(v)])) for k, v in d.items()}


def ema(y: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    out, m = np.empty_like(y), y[0]
    for i, v in enumerate(y):
        m = alpha * v + (1 - alpha) * m
        out[i] = m
    return out


def fig_training(m: dict, curriculum: int, out: Path) -> dict:
    def plot(ax, key, label, color, norm=False):
        if key not in m:
            return
        x, y = m[key]
        if norm:  # the first logged value can be ~0 (warm-up): scale by the early-training median
            y = y / max(float(np.median(y[: max(5, len(y) // 50)])), 1e-12)
        y = ema(y)
        ax.plot(x / 1e3, y, color=color, label=label)

    fig, axs = plt.subplots(2, 2, figsize=(DCOL, 3.6))
    a = axs[0, 0]
    plot(a, "fm_loss", "flow-matching (action)", OI["blue"])
    plot(a, "phase_loss", "phase", OI["orange"])
    a.set_yscale("log")
    a.set_ylabel("loss")
    a.set_title("(a) action and phase losses")
    a.legend(frameon=False)
    a = axs[0, 1]
    for key, lab, col in [("aux_track_loss", "point track", OI["green"]), ("aux_feat_loss", "DINOv2 feature", OI["purple"]),
                          ("aux_motion_loss", "motion", OI["sky"]), ("aux_rel_loss", "relative pose", OI["red"])]:
        plot(a, key, lab, col, norm=True)
    a.set_yscale("log")
    a.set_ylabel("loss / early-training loss")
    a.set_title("(b) auxiliary vision heads (pseudo-labels)")
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi * 30)  # headroom for the legend
    a.legend(frameon=False, ncol=2, loc="upper right")
    a = axs[1, 0]
    plot(a, "cot_token_acc", "explicit CoT token acc.", OI["blue"])
    if "latent_cot_loss" in m:
        b = a.twinx()
        x, y = m["latent_cot_loss"]
        b.plot(x / 1e3, ema(y), color=OI["purple"], label="latent CoT loss")
        b.set_yscale("log")
        b.set_ylabel("latent CoT loss", color=OI["purple"])
        b.spines["right"].set_visible(True)
    if curriculum > 0 and "fm_loss" in m:
        xs = np.linspace(0, m["fm_loss"][0][-1], 200)
        a.plot(xs / 1e3, np.clip(1 - xs / curriculum, 0, 1), color=OI["grey"], ls=":", label="p(explicit text)")
    a.set_ylim(0, 1.02)
    a.set_ylabel("accuracy / probability")
    a.set_title("(c) explicit → latent CoT curriculum")
    a.legend(frameon=False, loc="center right")
    a = axs[1, 1]
    plot(a, "moe/entropy_mean", "routing entropy (norm.)", OI["green"])
    layers = [k for k in m if k.startswith("moe/load_max_over_min/layer")]
    if layers:
        x = m[layers[0]][0]
        # an idle expert makes max/min infinite: clip, and summarize layers by median and IQR
        stack = np.clip(np.stack([np.interp(x, *m[k]) for k in layers]), 1.0, 20.0)
        b = a.twinx()
        b.plot(x / 1e3, ema(np.median(stack, 0)), color=OI["orange"], label="load max/min (median layer)")
        b.fill_between(x / 1e3, ema(np.percentile(stack, 25, 0)), ema(np.percentile(stack, 75, 0)),
                       color=OI["orange"], alpha=0.2, lw=0)
        b.set_yscale("log")
        b.set_ylim(1, 20)
        b.set_ylabel("expert load max/min (clipped at 20)", color=OI["orange"])
        b.spines["right"].set_visible(True)
    a.set_ylabel("entropy")
    a.set_title("(d) MoE routing")
    handles = a.get_legend_handles_labels()
    if layers:
        h2 = b.get_legend_handles_labels()
        handles = (handles[0] + h2[0], handles[1] + h2[1])
    a.legend(*handles, frameon=False, loc="center right")
    for a in axs.flat:
        a.set_xlabel("optimizer step (k)")
        a.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    save(fig, out, "training_dynamics")
    final = {k: float(np.mean(m[k][1][-10:])) for k in ("fm_loss", "phase_loss", "aux_track_loss", "aux_feat_loss",
                                                        "aux_motion_loss", "aux_rel_loss", "cot_token_acc",
                                                        "latent_cot_loss", "moe/entropy_mean") if k in m}
    return final


def fig_grad_cosine(m: dict, out: Path) -> dict:
    keys = [k for k in m if k.startswith("grad/cos/")]
    if not keys:
        return {}
    pairs = {tuple(k.removeprefix("grad/cos/").split("__")): float(np.mean(m[k][1])) for k in keys}
    names = ["fm_loss", "phase_loss", "aux_track_loss", "aux_feat_loss", "aux_motion_loss", "aux_rel_loss",
             "cot_loss", "latent_cot_loss", "moe_aux_loss"]
    names = [n for n in names if any(n in p for p in pairs)]
    lab = {"fm_loss": "action (FM)", "phase_loss": "phase", "aux_track_loss": "track", "aux_feat_loss": "feature",
           "aux_motion_loss": "motion", "aux_rel_loss": "rel. pose", "cot_loss": "explicit CoT",
           "latent_cot_loss": "latent CoT", "moe_aux_loss": "MoE balance"}
    mat = np.full((len(names), len(names)), np.nan)
    for (a, b), v in pairs.items():
        if a in names and b in names:
            i, j = names.index(a), names.index(b)
            mat[i, j] = mat[j, i] = v
    fig, ax = plt.subplots(figsize=(COL, COL * 0.9))
    lim = max(0.1, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap="RdBu_r", vmin=-lim, vmax=lim)  # diagonal left blank
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    ax.set_xticks(range(len(names)), [lab[n] for n in names], rotation=45, ha="right")
    ax.set_yticks(range(len(names)), [lab[n] for n in names])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="mean gradient cosine")
    save(fig, out, "gradient_cosine")
    norms = {k.removeprefix("grad/norm/"): float(np.mean(m[k][1])) for k in m if k.startswith("grad/norm/")}
    return {"cosine": {f"{a}__{b}": v for (a, b), v in pairs.items()}, "grad_norm_mean": norms}


def fig_expert_usage(root: Path, out: Path) -> dict:
    by = defaultdict(list)
    for p in sorted((root / "full" / SUITE).glob("seed*/instrument/metrics.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("expert_usage"):
                by[r["task"]].append(r["expert_usage"])
    if not by:
        return {}
    tasks = sorted(by)
    short = short_names(tasks)
    mat = np.array([np.mean(by[t], 0) for t in tasks])
    mat = mat / mat.sum(1, keepdims=True).clip(1e-9)
    fig, ax = plt.subplots(figsize=(COL, 0.24 * len(tasks) + 0.8))
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if mat[i, j] < mat.max() * 0.6 else "black")
    ax.set_xticks(range(mat.shape[1]), [f"E{i}" for i in range(mat.shape[1])])
    ax.set_yticks(range(len(tasks)), [short[t] for t in tasks])
    ax.set_xlabel("routed expert (mean over MoE layers)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="routing share")
    save(fig, out, "expert_usage_by_task")
    ent = [-(r * np.log(r.clip(1e-9))).sum() / math.log(len(r)) for r in mat]
    return {"per_task": {t: mat[i].tolist() for i, t in enumerate(tasks)},
            "normalized_entropy_mean": float(np.mean(ent)),
            "max_share_spread_across_tasks": float((mat.max(0) - mat.min(0)).max())}


# ----------------------------------------------------------------------------- tables / main
def tex_tables(summ: dict, effects: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in CONDITIONS:
        if c not in summ:
            continue
        s, e = summ[c], effects.get(c)
        lo, hi = s["ci95"]
        delta = "--" if e is None or c == "full" else (
            f"{100 * e['delta']:+.1f} [{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}]"
            + ("$^{*}$" if e["p_mcnemar"] < 0.05 else ""))
        label = CONDITIONS[c][0].replace("−", "$-$")
        if c == "full":
            label = f"\\textbf{{{label}}}"
        rows.append(f"{label} & {100 * s['success']:.1f} $\\pm$ {100 * s['std_over_seeds']:.1f} & "
                    f"[{100 * lo:.1f}, {100 * hi:.1f}] & {delta} \\\\")
    (out / "comparison.tex").write_text(
        "% success: mean $\\pm$ std over evaluation seeds; CI: cluster bootstrap over (task, init state);\n"
        "% $\\Delta$: paired difference to the full model, $^{*}$ exact McNemar p < 0.05\n"
        "\\begin{tabular}{lccc}\n\\toprule\nCondition & Success (\\%) & 95\\% CI & $\\Delta$ vs. full (pp) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    names = [c for c in CONDITIONS if c in summ]
    tasks = sorted(summ["full" if "full" in summ else names[0]]["per_task"])
    short = short_names(tasks)
    head = " & ".join(CONDITIONS[c][0].replace("−", "$-$") for c in names)
    body = "\n".join(f"{short[t]} & " + " & ".join(f"{100 * summ[c]['per_task'].get(t, float('nan')):.0f}" for c in names) + " \\\\"
                     for t in tasks)
    (out / "per_task.tex").write_text(
        f"\\begin{{tabular}}{{l{'c' * len(names)}}}\n\\toprule\nTask & {head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", help="eval_libero_baseline.py output of the reference policy (shown as SmolVLA)")
    ap.add_argument("--train-run", help="training output dir with metrics.csv / config.json")
    args = ap.parse_args()
    root = Path(args.root)
    figs = root / "figures"

    dirs = {c: root / c for c in CONDITIONS if c != "smolvla"}
    if args.baseline:
        dirs["smolvla"] = Path(args.baseline)
    eps_by = {c: load_condition(d) for c, d in dirs.items() if d.exists()}
    eps_by = {c: e for c, e in eps_by.items() if e}
    if not eps_by:
        raise SystemExit(f"no finished evaluations under {root}")
    # compare on the seeds the full model has (the baseline may have more)
    seeds = sorted({e["seed"] for e in eps_by.get("full", next(iter(eps_by.values())))})
    eps_by = {c: [e for e in v if e["seed"] in seeds] for c, v in eps_by.items()}
    eps_by = {c: v for c, v in eps_by.items() if v}

    with open(root / "all_episodes.csv", "w", newline="") as f:
        fields = ["condition", "seed", "task_id", "task", "episode_ix", "success", "length",
                  "detect_rate", "mean_targets", "mean_top_score", "gripper_rate"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c, eps in eps_by.items():
            for e in eps:
                w.writerow({"condition": c, **e})

    summ = {c: summarize(e) for c, e in eps_by.items()}
    effects = {c: paired(eps_by["full"], e) for c, e in eps_by.items() if "full" in eps_by and c != "full"}
    effects = {c: v for c, v in effects.items() if v is not None}

    fig_main(summ, figs)
    fig_effects(effects, figs)
    fig_per_task(summ, figs)
    fig_time_to_success(eps_by, figs)
    perception = fig_perception(eps_by.get("full", []), figs)
    experts = fig_expert_usage(root, figs)
    training, grads = {}, {}
    if args.train_run and (Path(args.train_run) / "metrics.csv").exists():
        m = load_metrics(Path(args.train_run) / "metrics.csv")
        cfg_path = Path(args.train_run) / "config.json"
        curriculum = int(json.loads(cfg_path.read_text()).get("cot_curriculum_steps", 0)) if cfg_path.exists() else 0
        training = fig_training(m, curriculum, figs)
        grads = fig_grad_cosine(m, figs)
    tex_tables(summ, effects, root / "tables")

    comp = {"suite": SUITE, "seeds": seeds, "conditions": {c: {"label": CONDITIONS[c][0], "component": CONDITIONS[c][1], **summ[c]}
                                                         for c in CONDITIONS if c in summ},
            "effects_vs_full": effects, "perception_bins_full": perception, "expert_usage_full": experts,
            "training_final": training, "gradients": grads}
    (root / "comparison.json").write_text(json.dumps(comp, indent=2))

    lines = [f"# LIBERO-Spatial: full model vs. inference-time probes ({root.name})\n",
             f"Seeds {seeds}; 10 tasks × 50 init states per seed; CI = cluster bootstrap over (task, init state).\n",
             "| condition | success % | 95% CI | Δ vs full (pp) | McNemar p |", "|---|---|---|---|---|"]
    for c in CONDITIONS:
        if c not in summ:
            continue
        s, e = summ[c], effects.get(c)
        d = "" if e is None else f"{100 * e['delta']:+.1f} [{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}]"
        p = "" if e is None else f"{e['p_mcnemar']:.2g}"
        lines.append(f"| {CONDITIONS[c][0]} | {100 * s['success']:.1f} ± {100 * s['std_over_seeds']:.1f} | "
                     f"[{100 * s['ci95'][0]:.1f}, {100 * s['ci95'][1]:.1f}] | {d} | {p} |")
    if experts:
        lines.append(f"\nMoE routing at test time: normalized entropy {experts['normalized_entropy_mean']:.3f}, "
                     f"largest per-expert share spread across tasks {experts['max_share_spread_across_tasks']:.3f}.")
    lines.append("\nFigures: " + ", ".join(sorted(p.name for p in figs.glob("*.pdf"))))
    lines.append("\nProbes alter the trained model at test time only: they measure reliance on a component, "
                 "not the gain from training with it (that needs the trained ablation arms). "
                 "The SmolVLA row is the released checkpoint, trained with its own recipe.")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
