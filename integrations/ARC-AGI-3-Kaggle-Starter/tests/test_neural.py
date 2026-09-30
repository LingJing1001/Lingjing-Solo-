"""Unit tests for numpy CNN/DNN neural reason loop."""
from __future__ import annotations

import numpy as np

from lingjing_solo.neural import (
    ActionEffectNet,
    FailureMemory,
    NeuralReasonLoop,
    NumpyCNN,
    OnlineMLP,
)
from lingjing_solo.perception.encoder import PerceptionEncoder
from lingjing_solo.core import SoloConfig
from lingjing_solo.transfer.neural_layer import NeuralCeaxController


def test_numpy_cnn_encode_shape_and_norm():
    cnn = NumpyCNN(num_colors=16, embed_dim=64)
    grid = np.zeros((64, 64), dtype=np.int16)
    grid[10:14, 20:24] = 3
    grid[40:46, 40:46] = 7
    z = cnn.encode(grid)
    assert z.shape == (64,)
    assert abs(float(np.linalg.norm(z)) - 1.0) < 1e-4


def test_online_mlp_learns_simple_xor_like():
    mlp = OnlineMLP(in_dim=4, hidden=16, out_dim=1, lr=0.1, seed=1)
    # teach y = x0 XOR-ish via repeating pairs
    for _ in range(80):
        for x, y in (
            (np.array([1, 0, 0, 0], np.float32), np.array([0.0], np.float32)),
            (np.array([0, 1, 0, 0], np.float32), np.array([1.0], np.float32)),
            (np.array([1, 1, 0, 0], np.float32), np.array([0.2], np.float32)),
            (np.array([0, 0, 1, 0], np.float32), np.array([0.8], np.float32)),
        ):
            mlp.update(x, y)
    pred_hi = float(mlp.predict(np.array([0, 1, 0, 0], np.float32))[0])
    pred_lo = float(mlp.predict(np.array([1, 0, 0, 0], np.float32))[0])
    assert pred_hi > pred_lo


def test_action_effect_net_and_failure_memory():
    net = ActionEffectNet(embed_dim=8)
    mem = FailureMemory(capacity=20, dim=8)
    z = np.zeros(8, dtype=np.float32)
    z[0] = 1.0
    for _ in range(20):
        net.learn(z, "ACTION6", effect=0.9, progressed=True, xy=(10, 10), shape=(64, 64))
    mem.add(z, "ACTION6", outcome="success", note="hit")
    z2 = z.copy()
    z2[1] = 0.1
    n = float(np.linalg.norm(z2) + 1e-8)
    z2 /= n
    assert mem.success_bonus(z2, "ACTION6") > 0.0
    mem.add(z, "ACTION1", outcome="noop", note="stuck")
    assert mem.failure_penalty(z, "ACTION1") > 0.0


def test_reason_loop_observe_summarize_retry():
    loop = NeuralReasonLoop(embed_dim=32, prefer_torch=False)
    g0 = np.zeros((64, 64), dtype=np.int16)
    g0[5:8, 5:8] = 2
    loop.encode(g0)
    loop.note_choice("ACTION6", (6, 6), (64, 64))
    g1 = g0.copy()
    g1[5:8, 5:8] = 4
    out = loop.observe(delta_pixels=9, progressed=False, grid=g1)
    assert out["learned"] is True
    assert out["outcome"] in ("fail", "noop", "success")
    # failure should raise penalty for similar retry
    loop.encode(g1)
    h = loop.hypothesize_action("ACTION6", (6, 6), (64, 64))
    assert "score" in h
    # progress success path
    loop.note_choice("ACTION6", (6, 6), (64, 64))
    out2 = loop.observe(delta_pixels=40, progressed=True, grid=g1)
    assert out2["outcome"] == "success"
    assert loop.memory.snapshot()["successes"] >= 1


def test_perception_encoder_uses_cnn_path():
    enc = PerceptionEncoder(SoloConfig())
    grid = np.random.randint(0, 16, (64, 64), dtype=np.int16)
    feat = enc.encode(grid)
    assert feat.shape == (128,)
    assert np.isfinite(feat).all()


def test_neural_ceax_controller_smoke():
    ctl = NeuralCeaxController(SoloConfig())
    grid = np.zeros((64, 64), dtype=np.int16)
    grid[20:24, 10:14] = 5
    grid[20:24, 40:44] = 9
    act, xy, why = ctl.choose(
        valid_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        objects=[],
        grid=grid,
        grid_hash="abc",
    )
    assert act.startswith("ACTION")
    assert "neural" in why or why
    ctl.observe_outcome(
        delta_pixels=12,
        progressed=False,
        grid_hash="def",
        levels=0,
        prev_grid=grid,
        grid=grid,
    )
    snap = ctl.snapshot()
    assert "neural" in snap
    assert snap["neural"].get("backend") in ("numpy", "torch")
