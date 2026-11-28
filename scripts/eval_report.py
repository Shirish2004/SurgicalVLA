"""Publication figures, tables and selected videos for one instrumented LIBERO evaluation.

    pixi run -e dev python scripts/eval_report.py --root results/libero/spatial_full25k_s0

Reads what ``scripts/eval_libero_baseline.py`` (``summary.json``, ``<suite>/seed*/eval_info.json``)
and ``SVLA_EVAL_INSTRUMENT=1`` (``<suite>/seed*/instrument/metrics.jsonl``, annotated videos)
wrote under ``--root`` and produces, under the same root:

    figures/{success_per_task,episode_length,detection_vs_success,expert_usage,latency}.{pdf,png}
    tables/{main,perception}.tex     cot_examples.md     metrics_summary.json     report.md
    videos_selected/                 one success + one failure per task, plus the annotated clips
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 3.4  # single-column width (in)
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#7f7f7f"  # Okabe-Ito, colour-blind safe


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def short_names(tasks: list[str]) -> dict[str, str]:
    """Drop the words every task shares at the start and end (LIBERO-Spatial: 'pick up the black bowl ...')."""
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


# ----------------------------------------------------------------------------- loading
def load(root: Path, suite: str):
    seeds = sorted((root / suite).glob("seed*/eval_info.json"))
    if not seeds:
        raise SystemExit(f"no {suite}/seed*/eval_info.json under {root}")
    runs, rollouts = [], []
    for p in seeds:
        info = json.loads(p.read_text())
        per = {}
        for e in info.get("per_task") or []:
            m = e.get("metrics", e)
            per[int(e.get("task_id", len(per)))] = [bool(s) for s in m["successes"]]
        runs.append((p.parent, per))
        mpath = p.parent / "instrument" / "metrics.jsonl"
        if mpath.exists():
            for line in mpath.read_text().splitlines():
                r = json.loads(line)
                r["seed"] = p.parent.name
                rollouts.append(r)
    # task id -> language, from the instrument (rollouts run tasks in id order)
    lang: dict[int, str] = {}
    for _, per in runs[:1]:
        order = [r["task"] for r in rollouts if r["seed"] == runs[0][0].name]
        uniq = list(dict.fromkeys(order))
        for i, tid in enumerate(sorted(per)):
            lang[tid] = uniq[i] if i < len(uniq) else f"task {tid}"
    return runs, rollouts, lang


# ----------------------------------------------------------------------------- figures
def fig_success(runs, lang, short, out: Path) -> dict:
    tids = sorted(runs[0][1])
    rows = []
    for t in tids:
        succ = [s for _, per in runs for s in per.get(t, [])]
        k, n = sum(succ), len(succ)
        rows.append((t, k / n if n else math.nan, *wilson(k, n), n))
    mean = float(np.mean([r[1] for r in rows]))
    fig, ax = plt.subplots(figsize=(COL, 0.22 * len(rows) + 0.6))
    for i, (t, p, lo, hi, _) in enumerate(rows):
        ax.barh(i, 100 * p, color=BLUE, height=0.65)
        ax.errorbar(100 * p, i, xerr=[[100 * (p - lo)], [100 * (hi - p)]], color="black", lw=0.7, capsize=1.5)
    ax.axvline(100 * mean, color=ORANGE, lw=0.9, ls="--", label=f"mean {100 * mean:.1f}%")
    ax.set_yticks(range(len(rows)), [short.get(lang.get(t, ""), f"task {t}") for t, *_ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("success rate (%), 95% Wilson CI")
    ax.legend(loc="lower right", frameon=False)
    save(fig, out, "success_per_task")
    return {lang.get(t, str(t)): {"success": p, "ci95": [lo, hi], "n": n} for t, p, lo, hi, n in rows}


def episodes(rollouts) -> list[dict]:
    return [{**e, "seed": r["seed"]} for r in rollouts for e in r["episodes"]]


def fig_length(eps, out: Path) -> None:
    s = [e["length"] for e in eps if e["success"]]
    f = [e["length"] for e in eps if not e["success"]]
    if not eps:
        return
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    bins = np.linspace(0, max(e["length"] for e in eps) + 1, 30)
    ax.hist(s, bins=bins, color=BLUE, alpha=0.85, label=f"success (n={len(s)})")
    ax.hist(f, bins=bins, color=ORANGE, alpha=0.75, label=f"failure (n={len(f)})")
    ax.set_xlabel("episode length (control steps)")
    ax.set_ylabel("episodes")
    ax.legend(frameon=False)
    save(fig, out, "episode_length")


def fig_detection(eps, out: Path) -> dict:
    eps = [e for e in eps if e["detect_rate"] is not None]
    if not eps:
        return {}
    edges = [0, 0.5, 0.8, 0.95, 1.0001]
    labels = ["<50", "50-80", "80-95", "≥95"]
    res = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels, strict=True):
        b = [e for e in eps if lo <= e["detect_rate"] < hi]
        k = sum(e["success"] for e in b)
        res.append((lab, len(b), k / len(b) if b else math.nan, *wilson(k, len(b))))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL * 2, 1.9))
    x = np.arange(len(res))
    ps = np.array([r[2] for r in res])
    err = np.array([[r[2] - r[3], r[4] - r[2]] for r in res]).T
    a1.bar(x, 100 * ps, color=BLUE, width=0.6)
    a1.errorbar(x, 100 * ps, yerr=100 * err, fmt="none", color="black", lw=0.7, capsize=1.5)
    for i, r in enumerate(res):
        a1.text(i, 2, f"n={r[1]}", ha="center", fontsize=6, color="white" if r[2] > 0.1 else "black")
    a1.set_xticks(x, labels)
    a1.set_xlabel("target detected in % of chunks (SAM 3, live)")
    a1.set_ylabel("success rate (%)")
    a1.set_ylim(0, 100)
    sc = [e["mean_top_score"] for e in eps if e["success"]]
    fc = [e["mean_top_score"] for e in eps if not e["success"]]
    a2.boxplot([sc, fc], tick_labels=["success", "failure"], widths=0.5, showfliers=False,
               medianprops={"color": ORANGE}, boxprops={"color": BLUE}, whiskerprops={"color": BLUE}, capprops={"color": BLUE})
    a2.set_ylabel("mean top SAM 3 score")
    save(fig, out, "detection_vs_success")
    return {lab: {"n": n, "success": p, "ci95": [lo, hi]} for lab, n, p, lo, hi in res}


def fig_experts(rollouts, short, out: Path) -> dict:
    by = defaultdict(list)
    for r in rollouts:
        if r.get("expert_usage"):
            by[r["task"]].append(r["expert_usage"])
    if not by:
        return {}
    tasks = list(by)
    mat = np.array([np.mean(by[t], 0) for t in tasks])
    mat = mat / mat.sum(1, keepdims=True).clip(1e-9)
    fig, ax = plt.subplots(figsize=(COL, 0.22 * len(tasks) + 0.8))
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
    ax.set_xticks(range(mat.shape[1]), [f"E{i}" for i in range(mat.shape[1])])
    ax.set_yticks(range(len(tasks)), [short.get(t, t) for t in tasks])
    ax.set_xlabel("routed expert (mean over MoE layers)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="routing share")
    save(fig, out, "expert_usage")
    return {t: mat[i].tolist() for i, t in enumerate(tasks)}


def fig_latency(rollouts, out: Path) -> dict:
    perc = [x / r["batch"] for r in rollouts for x in r["perception_s"]]
    chunk = [x / r["batch"] for r in rollouts for x in r["chunk_s"]]
    cot = [x / r["batch"] for r in rollouts for x in r["cot_probe_s"]]
    if not chunk:
        return {}
    pm, cm = 1e3 * np.mean(perc), 1e3 * np.mean(chunk)
    policy = max(cm - pm, 0.0)  # the chunk call includes perception
    stats = {"perception_ms_per_env": pm, "policy_ms_per_env": policy, "chunk_ms_per_env": cm,
             "cot_probe_ms_per_env": 1e3 * np.mean(cot) if cot else None,
             "chunk_ms_p95_per_env": 1e3 * float(np.percentile(chunk, 95))}
    fig, ax = plt.subplots(figsize=(COL, 1.3))
    ax.barh(0, pm, color=ORANGE, label="live perception (SAM 3 + DINOv2 + depth)")
    ax.barh(0, policy, left=pm, color=BLUE, label="VLM + MoE action expert")
    if cot:
        ax.barh(1, stats["cot_probe_ms_per_env"], color=GREY, hatch="///", label="explicit CoT decode (probe only)")
    ax.set_yticks([0, 1] if cot else [0], ["per action chunk", "CoT probe"] if cot else ["per action chunk"])
    ax.set_xlabel("ms per environment (batched inference)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=1)
    save(fig, out, "latency")
    return stats


# ----------------------------------------------------------------------------- tables / text
def tables(summary: dict, suite: str, per_task: dict, det: dict, eps, lat: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    s = summary["suites"][suite]
    lo, hi = s["ci95_cluster_bootstrap"]
    (out / "main.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\nSuite & Success (\\%) & 95\\% CI & Episodes \\\\\n\\midrule\n"
        f"LIBERO-Spatial & {100 * s['success_mean']:.1f} $\\pm$ {100 * s['success_std_over_seeds']:.1f} & "
        f"[{100 * lo:.1f}, {100 * hi:.1f}] & {s['n_tasks']}$\\times${s['episodes_per_task_per_seed']}$\\times${len(s['seeds'])} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n")
    if eps:
        def m(key, ok):
            v = [e[key] for e in eps if e[key] is not None and e["success"] == ok]
            return f"{100 * np.mean(v):.1f}" if v else "--"
        rows = [("Target detected (\\% chunks)", "detect_rate"), ("Gripper detected (\\% chunks)", "gripper_rate")]
        body = "".join(f"{n} & {m(k, True)} & {m(k, False)} \\\\\n" for n, k in rows)
        sc = lambda ok: f"{np.mean([e['mean_top_score'] for e in eps if e['success'] == ok and e['mean_top_score'] is not None]):.2f}"  # noqa: E731
        body += f"Mean top SAM 3 score & {sc(True)} & {sc(False)} \\\\\n"
        if lat:
            body += f"\\midrule\nPerception / chunk (ms) & \\multicolumn{{2}}{{c}}{{{lat['perception_ms_per_env']:.1f}}} \\\\\n"
            body += f"Policy / chunk (ms) & \\multicolumn{{2}}{{c}}{{{lat['policy_ms_per_env']:.1f}}} \\\\\n"
        (out / "perception.tex").write_text(
            "\\begin{tabular}{lcc}\n\\toprule\n & Success & Failure \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n")


def cot_examples(eps, short, path: Path) -> int:
    lines = ["# Probed explicit CoT (extra decode, not used for control)\n"]
    n = 0
    by = defaultdict(dict)
    for e in eps:
        if e["cot"] and ("success" if e["success"] else "failure") not in by[e["task"]]:
            by[e["task"]]["success" if e["success"] else "failure"] = e
    for task, d in by.items():
        lines.append(f"\n## {task}\n")
        for kind, e in d.items():
            lines.append(f"\n**{kind}** ({e['seed']}, row {e['row']}, {e['length']} steps)\n")
            lines += [f"{i + 1}. {t}" for i, t in enumerate(e["cot"][:4])]
            n += 1
    path.write_text("\n".join(lines) + "\n")
    return n


def select_videos(runs, root: Path, suite: str, lang, short, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    picked = []
    seed_dir, per = runs[0]
    for tid, succ in sorted(per.items()):
        vids = seed_dir / "videos" / f"{suite}_{tid}"
        for want in (True, False):
            k = next((i for i, s in enumerate(succ[:10]) if s == want and (vids / f"eval_episode_{i}.mp4").exists()), None)
            if k is None:
                continue
            name = f"task{tid}_{'success' if want else 'failure'}_{seed_dir.name}_ep{k}.mp4"
            shutil.copy2(vids / f"eval_episode_{k}.mp4", out / name)
            picked.append(name)
    for run_dir, _ in runs:
        for v in sorted((run_dir / "instrument" / "annotated").glob("*.mp4")):
            name = f"annotated_{run_dir.name}_{v.name}"
            shutil.copy2(v, out / name)
            picked.append(name)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    args = ap.parse_args()
    root = Path(args.root)
    summary = json.loads((root / "summary.json").read_text())
    runs, rollouts, lang = load(root, args.suite)
    short = short_names(list(lang.values()))
    figs = root / "figures"
    eps = episodes(rollouts)
    per_task = fig_success(runs, lang, short, figs)
    fig_length(eps, figs)
    det = fig_detection(eps, figs)
    experts = fig_experts(rollouts, short, figs)
    lat = fig_latency(rollouts, figs)
    tables(summary, args.suite, per_task, det, eps, lat, root / "tables")
    n_cot = cot_examples(eps, short, root / "cot_examples.md")
    vids = select_videos(runs, root, args.suite, lang, short, root / "videos_selected")
    s = summary["suites"][args.suite]
    ms = {"suite": args.suite, "success": s, "per_task": per_task, "detection_bins": det, "expert_usage": experts,
          "latency": lat, "instrumented_episodes": len(eps),
          "inference_cot_active": any(r.get("inference_cot_active") for r in rollouts),
          "cot_examples": n_cot, "videos": vids}
    (root / "metrics_summary.json").write_text(json.dumps(ms, indent=2))
    lo, hi = s["ci95_cluster_bootstrap"]
    report = [f"# {args.suite} evaluation: {root.name}\n",
              f"- success {100 * s['success_mean']:.1f}% ± {100 * s['success_std_over_seeds']:.1f} over seeds, "
              f"95% CI [{100 * lo:.1f}, {100 * hi:.1f}] ({s['n_tasks']} tasks × {s['episodes_per_task_per_seed']} × {len(s['seeds'])} seeds)",
              f"- instrumented episodes: {len(eps)}; explicit CoT at inference: {ms['inference_cot_active']} (probe examples: {n_cot})"]
    if lat:
        report.append(f"- per env: perception {lat['perception_ms_per_env']:.1f} ms, policy {lat['policy_ms_per_env']:.1f} ms per chunk")
    report += ["- figures: " + ", ".join(sorted(p.name for p in figs.glob("*.pdf"))), f"- videos: {len(vids)} in videos_selected/"]
    (root / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
