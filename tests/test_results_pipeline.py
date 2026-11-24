"""Results pipeline: per-episode records -> results.json -> figures, LaTeX and the baselines doc."""

from __future__ import annotations

import json

import numpy as np
import yaml

from scripts import build_baselines_doc, collect_results, make_figures


def _episodes(path, rate, rng):
    path.mkdir(parents=True)
    rows = [{"suite": "libero_spatial", "seed": "seed1000", "task": f"t{t}", "episode_ix": i, "init_state": i,
             "success": bool(rng.random() < rate), "length": 100, "outcome": "x"} for t in range(10) for i in range(50)]
    (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_results_flow_from_episodes_to_tables(tmp_path):
    rng = np.random.default_rng(0)
    spec = yaml.safe_load(open("configs/ablations/arms.yaml"))
    for arm, rate in (("smolvla", 0.5), ("aux_only", 0.8)):
        for seed in (0, 1):
            _episodes(tmp_path / f"results/ablations/libero_spatial/{arm}/seed{seed}", rate, rng)
    entries = collect_results.ours(tmp_path, spec)
    for e in entries:
        collect_results.validate(e)
    by = {e["arm"]: e for e in entries}
    assert by["aux_only"]["success"] > by["smolvla"]["success"] and len(by["smolvla"]["per_seed"]) == 2
    lo, hi = by["smolvla"]["ci95"]
    assert lo < by["smolvla"]["success"] < hi
    rows = make_figures.ablation_rows("libero_spatial", spec["arms"], root=tmp_path)
    assert rows and rows[0]["arm"] == "aux_only" and rows[0]["ci95_cluster_bootstrap"][0] > 0
    assert "aux\\_only" in make_figures.ablation_table(rows) and "Spatial" in make_figures.main_table(entries)
    assert make_figures.success_figure(entries, "libero_spatial", tmp_path).exists()
    pub = collect_results.published(collect_results.ROOT)
    md = build_baselines_doc.render(entries + pub)
    assert "| aux_only | ours | libero_spatial" in md and "OpenVLA" in md
