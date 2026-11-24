"""Publication figures (vector PDF) and LaTeX tables from the logged results, in one command.

    pixi run -e dev figures        # = collect_results.py, then this script, then build_baselines_doc.py

Inputs: ``results/results.json`` (``scripts/collect_results.py``) and the per-episode records
under ``results/ablations`` (for paired differences). Outputs:

* ``results/figures/success_<suite>.pdf``: success with 95 % CI per arm (ours), the official
  checkpoint in our harness when present, and published numbers as reference ticks;
* ``results/figures/ablation_<suite>.pdf``: paired difference of each arm to the arm it
  is compared with (``configs/ablations/arms.yaml``), with cluster-bootstrap CI;
* ``results/tables/main.tex``: arms x suites (success, CI);
* ``results/tables/ablation_<suite>.tex``: paired differences, CIs, McNemar p;
* ``results/tables/published.tex``: other VLAs as reported (reference only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SHORT = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 150})


def pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


def success_figure(entries: list[dict], suite: str, out: Path) -> Path | None:
    ours = [e for e in entries if e["source"] == "ours" and e["suite"] == suite]
    official = [e for e in entries if e["source"] == "official_checkpoint" and e["suite"] == suite]
    pub = [e for e in entries if e["source"] == "published" and e["suite"] == suite]
    rows = ours + official
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(3.4, 0.28 * len(rows) + 0.9))
    for i, e in enumerate(rows):
        lo, hi = e["ci95"] or (e["success"], e["success"])
        color = "#1f5aa6" if e["source"] == "ours" else "#7f7f7f"
        ax.errorbar(100 * e["success"], i, xerr=[[100 * (e["success"] - lo)], [100 * (hi - e["success"])]],
                    fmt="o", color=color, ms=3.5, capsize=2, lw=1)
    for e in pub:
        ax.axvline(100 * e["success"], color="#bbbbbb", lw=0.6, ls=":")
        ax.text(100 * e["success"], len(rows) - 0.4, e["arm"].split(" ")[0], rotation=90, fontsize=5.5, color="#888888",
                va="bottom", ha="right")
    ax.set_yticks(range(len(rows)), [e["arm"] for e in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f"success rate, LIBERO-{SHORT.get(suite, suite)} (%)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    path = out / f"success_{suite}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def ablation_rows(suite: str, arms: dict, root: Path = ROOT) -> list[dict]:
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.compare_arms import compare, load

    base = root / "results/ablations" / suite
    rows = []
    for arm, spec in arms.items():
        ref = spec.get("compare_to")
        if not ref or not (base / arm).exists() or not (base / ref).exists():
            continue
        try:
            r = compare(load(base / ref), load(base / arm)).get(suite)
        except FileNotFoundError:
            continue
        if r:
            rows.append({"arm": arm, "vs": ref, **r})
    return rows


def ablation_figure(rows: list[dict], suite: str, out: Path) -> Path | None:
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(3.4, 0.28 * len(rows) + 0.9))
    for i, r in enumerate(rows):
        d = 100 * r["diff_b_minus_a"]
        lo, hi = (100 * x for x in r["ci95_cluster_bootstrap"])
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color="#1f5aa6" if lo > 0 else ("#b22222" if hi < 0 else "#555555"),
                    ms=3.5, capsize=2, lw=1)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(rows)), [f"{r['arm']} − {r['vs']}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f"paired difference in success, LIBERO-{SHORT.get(suite, suite)} (points)")
    fig.tight_layout()
    path = out / f"ablation_{suite}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main_table(entries: list[dict]) -> str:
    ours = [e for e in entries if e["source"] in ("ours", "official_checkpoint")]
    arms = list(dict.fromkeys(e["arm"] for e in ours))
    lines = [r"\begin{tabular}{l" + "c" * len(SUITES) + "}", r"\toprule",
             "Model & " + " & ".join(SHORT[s] for s in SUITES) + r" \\", r"\midrule"]
    for arm in arms:
        cells = []
        for s in SUITES:
            e = next((x for x in ours if x["arm"] == arm and x["suite"] == s), None)
            if e is None:
                cells.append("--")
            elif e["ci95"]:
                cells.append(f"{pct(e['success'])} \\scriptsize{{[{pct(e['ci95'][0])}, {pct(e['ci95'][1])}]}}")
            else:
                cells.append(pct(e["success"]))
        lines.append(arm.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def ablation_table(rows: list[dict]) -> str:
    lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Arm & vs. & $\Delta$ success & 95\% CI & McNemar $p$ \\", r"\midrule"]
    for r in rows:
        lo, hi = r["ci95_cluster_bootstrap"]
        lines.append(f"{r['arm'].replace('_', chr(92) + '_')} & {r['vs'].replace('_', chr(92) + '_')} & "
                     f"{100 * r['diff_b_minus_a']:+.1f} & [{100 * lo:+.1f}, {100 * hi:+.1f}] & {r['mcnemar_p']:.3g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def published_table(entries: list[dict]) -> str:
    pub = [e for e in entries if e["source"] == "published"]
    models = list(dict.fromkeys(e["arm"] for e in pub))
    lines = [r"\begin{tabular}{l" + "c" * len(SUITES) + "l}", r"\toprule",
             "Model & " + " & ".join(SHORT[s] for s in SUITES) + r" & Source \\", r"\midrule"]
    for m in models:
        row = {e["suite"]: e for e in pub if e["arm"] == m}
        cite = next(iter(row.values()))["citation"]
        lines.append(m + " & " + " & ".join(pct(row[s]["success"]) if s in row else "--" for s in SUITES) + f" & {cite} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    entries = json.loads((ROOT / args.results).read_text())
    arms = yaml.safe_load((ROOT / "configs/ablations/arms.yaml").read_text())["arms"]
    figs, tabs = ROOT / args.out / "figures", ROOT / args.out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    written = []
    for suite in SUITES:
        if p := success_figure(entries, suite, figs):
            written.append(p)
        rows = ablation_rows(suite, arms)
        if p := ablation_figure(rows, suite, figs):
            written.append(p)
        if rows:
            (tabs / f"ablation_{suite}.tex").write_text(ablation_table(rows))
            written.append(tabs / f"ablation_{suite}.tex")
    (tabs / "main.tex").write_text(main_table(entries))
    (tabs / "published.tex").write_text(published_table(entries))
    written += [tabs / "main.tex", tabs / "published.tex"]
    for p in written:
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
