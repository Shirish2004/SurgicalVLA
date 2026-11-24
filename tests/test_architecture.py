"""Architecture components: MoE variants, auxiliary vision heads, latent CoT, sidecar targets, CoT traces."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from surgicalvla.modules.aux_heads import AuxVisionHeads
from surgicalvla.modules.moe_layers import PhaseConditionedMoEFFN, SwiGLUExpert, expert_rows
from surgicalvla.pseudolabels.cot import candidates, motion_word, phase_name
from surgicalvla.pseudolabels.reader import SidecarTargets, split_future_frames
from tests.cache_fixtures import make_cache
from tests.test_smolvla_moe import H, MLP, make_batch, make_config, make_model, to_model_inputs


# ----------------------------------------------------------------------------- MoE variants
def _active_ffn_width(moe: PhaseConditionedMoEFFN) -> int:
    shared = sum(e.gate_proj.out_features for e in moe.shared_experts)
    return shared + moe.top_k * moe.routed_experts[0].gate_proj.out_features


def test_matched_moe_costs_one_dense_ffn_and_upcycled_three():
    dense = MLP(H, 6 * H)
    width = dense.gate_proj.out_features
    matched = PhaseConditionedMoEFFN(dense, H, H, H, num_experts=4, num_shared_experts=1, top_k=2, variant="matched")
    upcycled = PhaseConditionedMoEFFN(dense, H, H, H, num_experts=4, num_shared_experts=1, top_k=2)
    assert _active_ffn_width(matched) == width
    assert _active_ffn_width(upcycled) == 3 * width
    shared, routed = expert_rows(width, "matched", 1, 4, 2, seed=0)
    assert len({tuple(r.tolist()) for r in routed}) == 4, "matched experts are distinct slices"


def test_widened_expert_equals_dense_at_init_and_reloads():
    dense = MLP(H, 2 * H).eval()
    rows = torch.cat([torch.arange(2 * H), torch.full((4 * H,), -1)])
    wide = SwiGLUExpert.from_dense(dense, rows=rows).eval()
    x = torch.randn(3, 5, H)
    with torch.no_grad():
        torch.testing.assert_close(wide(x), dense(x))
        new = MLP(H, 2 * H).eval()
        wide.load_dense(new.gate_proj.weight, new.up_proj.weight, new.down_proj.weight)
        torch.testing.assert_close(wide(x), new(x))


@pytest.mark.parametrize(("variant", "n_moe"), [("upcycled", 4), ("matched", 4), ("dense", 0), ("dense_param_matched", 0)])
def test_model_builds_every_variant(variant, n_moe):
    cfg = make_config(moe_variant=variant)
    model = make_model(cfg)
    assert len(model._moe_layers) == n_moe
    out = model(**to_model_inputs(model, make_batch(cfg)))
    assert torch.isfinite(out["loss"])


# ----------------------------------------------------------------------------- auxiliary heads
def test_aux_losses_are_confidence_weighted():
    heads = AuxVisionHeads(H, feat_dim=8)
    pooled = torch.randn(4, H)
    aux = {
        "aux.obj_feat": torch.randn(4, 8), "aux.obj_feat_conf": torch.tensor([1.0, 0, 0, 0]),
        "aux.obj_motion": torch.randn(4, 2), "aux.obj_motion_conf": torch.zeros(4),
    }
    losses = heads.losses(pooled, aux)
    assert set(losses) == {"aux_feat_loss", "aux_motion_loss"}
    assert float(losses["aux_motion_loss"]) == 0.0  # all-zero confidence contributes nothing
    perturbed = {**aux, "aux.obj_feat": torch.cat([aux["aux.obj_feat"][:1], torch.randn(3, 8)])}
    torch.testing.assert_close(heads.losses(pooled, perturbed)["aux_feat_loss"], losses["aux_feat_loss"])


def test_model_trains_aux_heads():
    cfg = make_config(aux_vision=True, aux_feat_dim=8)
    model = make_model(cfg).train()
    batch = make_batch(cfg)
    aux = {"aux.obj_feat": torch.randn(2, 8), "aux.obj_feat_conf": torch.ones(2),
           "aux.track_disp": torch.randn(2, 2) * 0.01, "aux.track_disp_conf": torch.ones(2)}
    out = model(**to_model_inputs(model, batch), aux=aux)
    assert "aux_feat_loss" in out and "aux_track_loss" in out
    out["loss"].backward()
    assert model.aux_heads.heads["feat"].weight.grad is not None


# ----------------------------------------------------------------------------- latent CoT
def test_latent_tokens_come_last_and_train():
    cfg = make_config(latent_cot=True, latent_cot_tokens=3)
    model = make_model(cfg).train()
    batch = make_batch(cfg)
    inputs = to_model_inputs(model, batch)
    target = torch.randn(2, H)
    out = model(**inputs, future_latent=target)
    assert "latent_cot_loss" in out
    start, end = model._last_latent_slice
    assert end - start == 3
    out["loss"].backward()
    assert model.latent.queries.grad is not None and model.latent.queries.grad.abs().sum() > 0
    assert cfg.observation_delta_indices == [0, cfg.latent_cot_horizon]
    assert make_config().observation_delta_indices == [0]


def test_curriculum_drops_explicit_cot_over_time():
    steps, rng = 100, np.random.default_rng(0)
    p = [max(0.0, 1.0 - t / steps) for t in (0, 50, 100)]
    assert p == [1.0, 0.5, 0.0]
    assert rng is not None


# ----------------------------------------------------------------------------- sidecar targets / traces
def _store(tmp_path):
    return make_cache(tmp_path)


def test_sidecar_targets_align_frames_and_confidences(tmp_path):
    _store(tmp_path)
    reader = SidecarTargets(tmp_path, "org/ds", feat_dim=8, track_horizon=2, num_phases=4)
    t = reader.frame_targets(3, 7)  # nearest labelled frame at or before 7 is 6
    assert float(t["aux.obj_feat_conf"]) == pytest.approx(0.9, abs=1e-3)
    np.testing.assert_allclose(t["aux.track_disp"], [0.02, 0.02], atol=2e-3)
    assert int(t["subtask_phase_id"]) == 1  # after the boundary at 6
    missing = reader.frame_targets(99, 0)
    assert float(missing["aux.obj_feat_conf"]) == 0.0
    assert reader.trace(3, 10) == "<cot> phase: grasp </cot>" and reader.trace(3, 2) == ""
    batch = reader.for_batch(torch.tensor([3, 99]), torch.tensor([7, 0]))
    assert batch["aux.obj_feat"].shape == (2, 8) and batch["subtask_phase_id"].tolist() == [1, 0]


def test_split_future_frames():
    batch = {"observation.images.camera1": torch.randn(2, 2, 3, 4, 4), "observation.state": torch.randn(2, 2, 5), "action": torch.randn(2, 6, 7)}
    out = split_future_frames(dict(batch), ["observation.images.camera1"])
    assert out["observation.images.camera1"].shape == (2, 3, 4, 4)
    torch.testing.assert_close(out["future.observation.images.camera1"], batch["observation.images.camera1"][:, 1])
    assert out["observation.state"].shape == (2, 5) and out["action"].shape == (2, 6, 7)


def test_cot_candidates_and_phase_names(tmp_path):
    from surgicalvla.pseudolabels.reader import episode_view

    d = episode_view(_store(tmp_path).read(3))
    assert phase_name(2, d["phase_frames"], d["phase_kinds"]) == "approach"
    assert phase_name(8, d["phase_frames"], d["phase_kinds"]) == "grasp"
    assert phase_name(30, d["phase_frames"], d["phase_kinds"]) == "release"
    assert motion_word(np.array([0.01, 0.0])) == "right" and motion_word(np.zeros(2)) == "still"
    c = candidates(d, 8)
    assert c[1] == "<cot> phase: grasp | object: black bowl | motion: right </cot>"
