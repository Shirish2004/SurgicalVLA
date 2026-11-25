"""Feature cache: round trip, provenance checks, past-only inputs, bounded disk."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surgicalvla.perception.object_inputs import ObjectInputConfig, describe, grid_masks, select_objects
from surgicalvla.pseudolabels.cache import CacheMismatch, FeatureCache, check_manifest, rle_decode, rle_encode
from surgicalvla.pseudolabels.reader import SidecarTargets
from tests.cache_fixtures import MANIFEST, make_cache


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for m in (rng.random((7, 9)) > 0.5, np.ones((3, 4), bool), np.zeros((3, 4), bool)):
        assert np.array_equal(rle_decode(rle_encode(m), m.shape), m)


@pytest.mark.parametrize("runs", [np.zeros(1, np.int32), np.zeros(0, np.int32)])
def test_frames_without_a_mask_decode_to_an_empty_mask(runs):
    mask = rle_decode(runs, (4, 5))  # [0] is what make_pseudolabels writes for a frame without a mask
    assert mask.shape == (4, 5) and not mask.any()


def test_cache_roundtrip_resume_and_checksum(tmp_path):
    cache = make_cache(tmp_path)
    d = cache.read(3)
    assert d["frame"].tolist() == list(range(20)) and d["has_target"].sum() == 10
    assert d["in_obj_feat"].shape == (20, 4, 8) and d["ep.phrase"] == "black bowl"
    reopened = FeatureCache(tmp_path, "org/ds", "test")
    assert reopened.done(3) and not reopened.done(4)
    shard = next(reopened.shard_dir.glob("frames-*.parquet"))
    shard.write_bytes(shard.read_bytes()[:-8] + b"corrupt!")
    with pytest.raises(CacheMismatch, match="checksum"):
        FeatureCache(tmp_path, "org/ds", "test").read(3)


def test_empty_episode_strings_read_back_as_strings(tmp_path):
    assert make_cache(tmp_path, phrase="").read(3)["ep.phrase"] == ""


def test_reader_refuses_a_mismatched_cache_and_names_the_field(tmp_path):
    make_cache(tmp_path)
    with pytest.raises(CacheMismatch) as err:
        SidecarTargets(tmp_path, "org/ds", name="test", expect={"labeler": {"stride": 4, "causal_inputs": True}})
    assert "labeler.stride" in str(err.value) and "expects 4" in str(err.value) and "causal_inputs" not in str(err.value)
    SidecarTargets(tmp_path, "org/ds", name="test", expect={"labeler": {"stride": 2}, "config_hash": "test-hash"})
    with pytest.raises(CacheMismatch, match="config_hash"):
        check_manifest(MANIFEST, {"config_hash": "other"})


def test_inputs_never_come_from_a_later_frame(tmp_path):
    make_cache(tmp_path)  # cached input feature of frame f equals f
    reader = SidecarTargets(tmp_path, "org/ds", name="test", feat_dim=8, object_input="objects")
    for f in range(20):
        got = reader.frame_inputs(3, f)["obj_sam_feat"]
        assert got.max() <= f  # never a later frame
        assert got.max() == f  # and the latest available one
    batch = reader.for_batch(torch.tensor([3, 99]), torch.tensor([5, 0]))
    assert batch["obj_sam_feat"].shape == (2, 4, 8) and float(batch["obj_sam_feat"][0].max()) == 5
    assert not batch["obj_valid_mask"][1].any()  # uncached episode: no objects


def test_object_selection_drops_the_gripper_and_duplicates():
    cfg = ObjectInputConfig(num_candidates=2)
    m = lambda y0, y1: np.pad(np.ones((y1 - y0, 4), bool), ((y0, 16 - y1), (0, 12)))  # noqa: E731
    inst = {"bowl": [(0.8, m(0, 4)), (0.7, m(0, 4)), (0.6, m(8, 12))], "black bowl": [(0.5, m(12, 16))],
            "robot end effector": [(0.9, m(8, 12))], "robot hand": []}
    targets, grippers = select_objects(inst, ["black bowl", "bowl"], cfg)
    assert [g[0] for g in grippers] == [0.9]
    assert [t[0] for t in targets] == [0.8, 0.5]  # duplicate 0.7 merged, 0.6 is the gripper
    pose = describe(m(0, 4), 0.8, np.full((16, 16), 0.3))
    assert pose.shape == (8,) and pose[6] == pytest.approx(0.3) and pose[7] == pytest.approx(0.8)
    assert sum(g.sum() for g in grid_masks(16, 16)) == 256


def test_eviction_keeps_disk_bounded(tmp_path):
    cache = FeatureCache(tmp_path, "org/ds", "evict", manifest=MANIFEST, shard_episodes=1)
    row = lambda: {"frame": np.arange(2, dtype=np.int32), "x": np.ones((2, 4), np.float16)}  # noqa: E731
    cache.add({"episode": 0, "length": 2}, row())
    cache.max_bytes = int(cache.disk_bytes() * 1.5)  # room for one shard
    with pytest.raises(RuntimeError, match="max_gb"):  # nothing uploaded, so nothing can be evicted
        cache.add({"episode": 1, "length": 2}, row())
    for s in cache.manifest["shards"]:
        s["uploaded"] = True  # as if pushed to the Hub
    cache.add({"episode": 2, "length": 2}, row())
    assert not (cache.shard_dir / "frames-00000.parquet").exists() and cache.disk_bytes() <= cache.max_bytes


def test_two_grippers_for_bimanual_robots():
    cfg = ObjectInputConfig(num_candidates=1, num_grippers=2)
    m = lambda x0: np.pad(np.ones((4, 4), bool), ((0, 12), (x0, 12 - x0)))  # noqa: E731
    inst = {"robot end effector": [(0.9, m(0)), (0.8, m(8))], "robot hand": [(0.85, m(0))], "cup": [(0.7, m(4))]}
    targets, grippers = select_objects(inst, ["cup"], cfg)
    assert [g[0] for g in grippers] == [0.9, 0.8]  # duplicate 0.85 (same mask as 0.9) merged
    assert [t[0] for t in targets] == [0.7]


def test_per_object_tokens_and_usage_probe():
    from surgicalvla.modules.modeling_smolvla_moe import ablate_object_inputs
    from surgicalvla.modules.object_encoder import ObjectCentric6DProjector

    proj = ObjectCentric6DProjector(hidden_size_vlm=16, hidden_size_expert=8, sam_feat_dim=6, pose_dim=8, num_tokens=4,
                                    layout="per_object").eval()
    feat, pose = torch.randn(2, 4, 6), torch.rand(2, 4, 8)
    valid = torch.tensor([[True, True, False, True], [False] * 4])
    out = proj(feat, pose, valid)
    assert out.tokens.shape == (2, 4, 16) and out.token_mask.tolist() == valid.tolist()
    assert out.tokens[0, 2].abs().sum() == 0 and out.context[1].abs().sum() == 0
    batch = {"obj_sam_feat": feat, "obj_pose_vec": pose, "obj_valid_mask": valid}
    assert not ablate_object_inputs(batch, "zero")["obj_valid_mask"].any()
    shuf = ablate_object_inputs(batch, "shuffle")
    assert sorted(shuf["obj_sam_feat"][0].sum(-1).tolist()) == pytest.approx(sorted(feat[0].sum(-1).tolist()))


def test_object_input_config_fixes_the_token_shapes():
    from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig

    cfg = SmolVLAMoEConfig(object_input="objects", obj_candidates=3)
    assert (cfg.num_objects, cfg.obj_pose_dim, cfg.obj_sam_feat_dim, cfg.obj_token_layout) == (4, 8, cfg.aux_feat_dim, "per_object")
    assert SmolVLAMoEConfig(object_input="grid").num_objects == 4
    with pytest.raises(ValueError):
        SmolVLAMoEConfig(object_input="bad")



def test_training_data_has_no_geometric_augmentation(monkeypatch):
    """Cached boxes, masks and tracks are in the dataset frame's coordinates; no augmentation may move pixels."""
    import types

    import surgicalvla.data.dataset as ds

    seen = {}

    class Fake:
        def __init__(self, repo_id, **kw):
            seen.update(kw)

    monkeypatch.setattr(ds, "LeRobotDataset", Fake)
    monkeypatch.setattr(ds, "LeRobotDatasetMetadata", lambda *a, **k: types.SimpleNamespace(camera_keys=[], fps=10, features={}))
    monkeypatch.setattr(ds, "_delta_timestamps", lambda c, m: None)
    monkeypatch.setattr(ds, "resolve_camera_map", lambda m, c: None)
    ds.build_dataset("org/ds", config=None)
    assert seen.get("image_transforms") is None


def test_episode_reader_keeps_only_the_labelled_camera_as_uint8(monkeypatch):
    import types

    import lerobot.datasets.streaming_dataset as sd

    from scripts.make_pseudolabels import iter_episodes

    def item(ep, f):
        return {"episode_index": torch.tensor(ep), "frame_index": torch.tensor(f), "task": "t",
                "cam_a": torch.rand(3, 8, 8), "cam_b": torch.ones(3, 8, 8),
                "observation.state": torch.zeros(2), "action": torch.zeros(3)}

    class Fake:
        def __init__(self, *a, **k):
            self.meta = types.SimpleNamespace(camera_keys=["cam_a", "cam_b"])
            self.eps = k.get("episodes")

        def __iter__(self):  # frames out of order within the episode
            return iter([item(e, f) for e in self.eps for f in (1, 0)])

    monkeypatch.setattr(sd, "StreamingLeRobotDataset", Fake)
    args = types.SimpleNamespace(repo_id="x", root="r", revision=None, camera="cam_b")  # the named camera, not the first
    out = list(iter_episodes(args, [0, 1], {0: 2, 1: 2}))
    assert [o[0] for o in out] == [0, 1]
    frames = out[0][1]
    assert frames.dtype == np.uint8 and frames.shape == (2, 8, 8, 3) and (frames == 255).all()
    with pytest.raises(SystemExit, match="not a camera"):
        list(iter_episodes(types.SimpleNamespace(repo_id="x", root="r", revision=None, camera="front"), [0], {0: 2}))
