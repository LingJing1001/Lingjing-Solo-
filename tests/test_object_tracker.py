"""object_tracker：跨帧物体身份（平移不变 hash）。

覆盖四件事：分割产物契约、位移检出、状态机语义、异常不外抛。
用合成帧而不是录制帧，这样每个断言都能指向一个具体构造。
"""
from __future__ import annotations

import numpy as np

from lingjing_solo.perception.object_tracker import (
    ObjectTracker,
    compare_frames,
    moving_nodes,
    objects_of,
)


def _blank(h: int = 16, w: int = 16) -> np.ndarray:
    return np.zeros((h, w), dtype=int)


def _stamp(grid, r0, c0, h, w, color):
    grid[r0:r0 + h, c0:c0 + w] = color
    return grid


# --------------------------------------------------------------------------- #
# 1. 分割产物契约
# --------------------------------------------------------------------------- #

def test_objects_of_reports_hash_bbox_and_centre():
    g = _blank()
    _stamp(g, 2, 3, 3, 3, 5)
    objs = objects_of(g)
    assert len(objs) == 2, "背景 + 方块"
    fg = [o for o in objs if o["npix"] == 9]
    assert len(fg) == 1
    o = fg[0]
    assert o["bbox"] == (2, 3, 4, 5)
    assert o["centre"] == (3, 4)
    assert isinstance(o["hash"], str) and len(o["hash"]) == 16


def test_same_shape_different_position_same_hash():
    """平移不变性：形状一样、位置不同 -> hash 相同。这是跨帧匹配的全部基础。

    只比前景方块：背景连通域被挖的洞在不同位置，归一化后形状本来就不同，
    它的 hash 该变 —— 那不是反例。
    """
    a = _blank()
    _stamp(a, 1, 1, 3, 3, 5)
    b = _blank()
    _stamp(b, 9, 9, 3, 3, 5)
    ha = [o["hash"] for o in objects_of(a) if o["npix"] == 9]
    hb = [o["hash"] for o in objects_of(b) if o["npix"] == 9]
    assert len(ha) == 1 and len(hb) == 1
    assert ha == hb


def test_different_shape_different_hash():
    a = _blank()
    _stamp(a, 1, 1, 3, 3, 5)
    b = _blank()
    _stamp(b, 1, 1, 3, 4, 5)
    assert set(o["hash"] for o in objects_of(a)) != set(o["hash"] for o in objects_of(b))


# --------------------------------------------------------------------------- #
# 2. 位移检出
# --------------------------------------------------------------------------- #

def test_compare_frames_detects_pure_translation():
    a = _blank()
    _stamp(a, 2, 2, 3, 3, 7)
    b = _blank()
    _stamp(b, 7, 2, 3, 3, 7)  # 向下 5

    rep = compare_frames(a, b)
    assert rep["ok"] is True
    moved = [m for m in rep["moved"] if m["npix"] == 9]
    assert len(moved) == 1
    assert moved[0]["delta"] == (5, 0)
    assert moved[0]["shift"] == 5


def test_compare_frames_no_move_when_identical():
    g = _blank()
    _stamp(g, 2, 2, 3, 3, 7)
    rep = compare_frames(g, g.copy())
    assert rep["ok"] is True
    # 只有背景与方块，两者都没动
    assert [m for m in rep["moved"] if m["npix"] == 9] == []


def test_min_shift_filters_small_jitter():
    """min_shift 是严格大于：shift=1 的抖动在 min_shift=1 时就被挡住。"""
    a = _blank()
    _stamp(a, 2, 2, 3, 3, 7)
    b = _blank()
    _stamp(b, 3, 2, 3, 3, 7)  # 只动 1 格

    assert [m for m in compare_frames(a, b, min_shift=0)["moved"] if m["npix"] == 9]
    assert [m for m in compare_frames(a, b, min_shift=1)["moved"] if m["npix"] == 9] == []


def test_min_npix_drops_tiny_components():
    g = _blank()
    _stamp(g, 1, 1, 1, 1, 3)  # 单像素
    _stamp(g, 8, 8, 3, 3, 4)
    assert any(o["npix"] == 1 for o in objects_of(g, min_npix=0))
    assert all(o["npix"] > 1 for o in objects_of(g, min_npix=2))


def test_dominant_is_weighted_average_within_grid():
    """dominant 必须是平均位移，不能跑到网格外（早期版本用了加权和）。"""
    a = _blank(32, 32)
    _stamp(a, 1, 1, 4, 4, 5)
    _stamp(a, 20, 20, 4, 4, 6)
    b = _blank(32, 32)
    _stamp(b, 5, 1, 4, 4, 5)    # +4 行
    _stamp(b, 24, 20, 4, 4, 6)  # +4 行
    rep = compare_frames(a, b)
    dr, dc = rep["dominant"]
    assert (dr, dc) == (4, 0)


# --------------------------------------------------------------------------- #
# 3. moving_nodes
# --------------------------------------------------------------------------- #

def test_moving_nodes_returns_current_position():
    a = _blank()
    _stamp(a, 2, 2, 3, 3, 7)
    b = _blank()
    _stamp(b, 7, 2, 3, 3, 7)
    pts = moving_nodes(a, b, limit=4, min_npix=4)
    # 方块从 (2..4, 2..4) 移到 (7..9, 2..4)，中心 (3,3) -> (8,3)
    assert (8, 3) in pts


# --------------------------------------------------------------------------- #
# 4. 状态机
# --------------------------------------------------------------------------- #

def test_tracker_cold_start_then_detects_move():
    g0 = _blank()
    _stamp(g0, 2, 2, 3, 3, 7)
    g1 = _blank()
    _stamp(g1, 6, 2, 3, 3, 7)

    tr = ObjectTracker(min_shift=2, min_npix=4)
    r0 = tr.observe(g0)
    assert r0["ok"] is True and r0["moved"] == [], "冷启动不该报位移"

    r1 = tr.observe(g1)
    assert tr.steps == 2
    assert any(m["npix"] == 9 for m in r1["moved"])
    assert tr.moved_events == 1


def test_tracker_reset_clears_state():
    g = _blank()
    _stamp(g, 2, 2, 3, 3, 7)
    tr = ObjectTracker()
    tr.observe(g)
    tr.observe(g)
    assert tr.steps == 2
    tr.reset()
    assert tr.steps == 0 and tr.moved_events == 0
    assert tr.observe(g)["moved"] == [], "reset 后第一帧是冷启动"


def test_tracker_summary_is_json_safe():
    g = _blank()
    _stamp(g, 2, 2, 3, 3, 7)
    tr = ObjectTracker()
    tr.observe(g)
    s = tr.summary()
    assert set(["steps", "n_cur", "matched", "moved", "dominant",
                "moved_events", "errors"]).issubset(s)
    import json
    json.dumps(s)  # 不含 ndarray / 浮点，可直接进日志


# --------------------------------------------------------------------------- #
# 5. 异常不外抛
# --------------------------------------------------------------------------- #

def test_bad_inputs_return_empty_never_raise():
    # 注意 [[1, 2], [3, 4]] 是**合法**的 2x2 网格（会正常分割出 4 个单像素物体），
    # 不是坏输入；真的畸形是一维、标量、字符串这些。
    for bad in (None, [], "xyz", 42, [1, 2, 3]):
        assert objects_of(bad) == []
        assert compare_frames(bad, bad)["ok"] is False
        assert moving_nodes(bad, bad) == []


def test_tiny_but_valid_grid_is_handled():
    objs = objects_of([[1, 2], [3, 4]])
    assert len(objs) == 4
    assert all(o["npix"] == 1 for o in objs)


def test_tracker_survives_bad_frame_and_counts_error():
    tr = ObjectTracker()
    tr.observe(None)
    assert tr.errors == 1
    assert tr.steps == 0, "畸形帧不该推进步数"


def test_deterministic_same_input_same_output():
    g0 = _blank()
    _stamp(g0, 2, 2, 3, 3, 7)
    g1 = _blank()
    _stamp(g1, 4, 2, 3, 3, 7)
    a = compare_frames(g0, g1)
    b = compare_frames(g0, g1)
    assert a == b
