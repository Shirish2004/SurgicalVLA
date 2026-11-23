"""Joint SFT + RL: lambda schedule, gradient bookkeeping, resumable demo stream."""

from __future__ import annotations

import math

import pytest
import torch

from surgicalvla.rl.joint import DemoStream, LambdaSchedule, combine_gradients, take_grads
from surgicalvla.utils.checkpoints import latest_pretrained
from surgicalvla.utils.splits import block_episodes


@pytest.mark.parametrize("kind", ["constant", "linear", "cosine", "exp"])
def test_lambda_schedule_endpoints(kind):
    s = LambdaSchedule(start=1.0, end=0.1, kind=kind, steps=100)
    assert s(0) == pytest.approx(1.0)
    assert s(100) == pytest.approx(1.0 if kind == "constant" else 0.1)
    assert s(1000) == s(100)
    if kind != "constant":
        assert s(25) > s(75)


def test_combine_gradients_writes_weighted_sum_and_stats():
    p1, p2 = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))
    g_rl = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0])]
    g_fm = [torch.tensor([-1.0, 0.0, 0.0]), None]
    stats = combine_gradients([p1, p2], g_rl, g_fm, lam=0.5)
    torch.testing.assert_close(p1.grad, torch.tensor([0.5, 0.0, 0.0]))
    torch.testing.assert_close(p2.grad, torch.zeros(2))
    assert stats["grad_cosine_rl_fm"] == pytest.approx(-1.0)  # opposing objectives
    assert stats["grad_norm_rl"] == pytest.approx(1.0) and stats["grad_norm_fm_weighted"] == pytest.approx(0.5)


def test_separate_gradients_equal_joint_backward():
    """Measuring the two terms separately must not change the update."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 1)
    x = torch.randn(8, 4)
    rl = lambda: lin(x).pow(2).mean()  # noqa: E731
    fm = lambda: (lin(x) - 1).abs().mean()  # noqa: E731
    lam = 0.3
    (rl() + lam * fm()).backward()
    ref = [p.grad.clone() for p in lin.parameters()]
    params = list(lin.parameters())
    for p in params:
        p.grad = None
    rl().backward()
    a = take_grads(params)
    fm().backward()
    b = take_grads(params)
    combine_gradients(params, a, b, lam)
    for p, r in zip(params, ref, strict=True):
        torch.testing.assert_close(p.grad, r)


class FakeStream:
    """Deterministic stand-in for StreamingLeRobotDataset with its state API."""

    def __init__(self, n=10):
        self.n, self.offset = n, 0

    def __iter__(self):
        start, self.offset = self.offset, self.offset
        for i in range(start, self.n):
            self.offset = i + 1
            yield {"x": torch.tensor([float(i)]), "observation.images.image": torch.zeros(1)}

    def state_dict(self):
        return {"offset": self.offset}

    def load_state_dict(self, state):
        self.offset = state["offset"]


def test_demo_stream_resumes_exactly_and_renames():
    stream = DemoStream(FakeStream(), batch_size=3, rename={"observation.images.image": "observation.images.camera1"})
    first = stream.next_batch()
    assert "observation.images.camera1" in first and first["x"].flatten().tolist() == [0, 1, 2]
    state = stream.state_dict()
    expected = stream.next_batch()["x"].flatten().tolist()

    resumed = DemoStream(FakeStream(), batch_size=3)
    resumed.load_state_dict(state)
    assert resumed.next_batch()["x"].flatten().tolist() == expected == [3, 4, 5]


def test_demo_stream_wraps_epochs():
    stream = DemoStream(FakeStream(n=4), batch_size=3)
    stream.next_batch()
    stream.dataset.offset = 4  # exhausted
    stream._it = iter([])
    stream.dataset.offset = 0
    assert stream.next_batch()["x"].numel() == 3 and stream.epoch == 1


def test_block_episodes_selects_robotwin_demo_clean():
    eps = block_episodes(27500, 550, 50)
    assert len(eps) == 2500 and eps[:2] == [0, 1] and eps[50] == 550 and eps[-1] == 49 * 550 + 49


def test_latest_pretrained_prefers_final_then_newest_step(tmp_path):
    for d in ("step_000025", "step_000100", "step_000050"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "config.json").write_text("{}")
    assert latest_pretrained(tmp_path).name == "step_000100"
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "config.json").write_text("{}")
    assert latest_pretrained(tmp_path).name == "final"
    assert not math.isnan(0.0)
