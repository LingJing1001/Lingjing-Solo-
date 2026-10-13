"""统一帧解码入口：一份实现，三处委托，FrameData 接缝消除。

覆盖三件事：
1. ``core.frame_decode.grid_from_frame`` 本身对各类输入的语义；
2. **三处委托方给出完全一致的结果**（arc_env / object_graph / core.utils）；
3. 接缝消失的端到端证据：**裸 FrameData 直接喂** ``build_graph`` / ``Harness.decide``
   不再需要调用方先解码。
"""

from __future__ import annotations

import numpy as np
import pytest

from lingjing_solo.core import frame_decode
from lingjing_solo.core.frame_decode import grid_from_frame, is_grid
from lingjing_solo.core.utils import extract_grid


class _FD:
    """鸭子类型的 FrameData：只有 ``.frame``（真实结构：一帧一层的列表）。"""

    def __init__(self, layers, state=None, levels=0, avail=None):
        self.frame = layers
        self.state = state
        self.levels_completed = levels
        self.available_actions = avail or []


def _grid(n=8, cells=()):
    g = [[0] * n for _ in range(n)]
    for r, c, v in cells:
        g[r][c] = v
    return g


# --------------------------------------------------------------------------- #
# 1. 解码器本身
# --------------------------------------------------------------------------- #


def test_nested_list_and_tuple():
    assert grid_from_frame([[0, 1], [2, 3]]) == [[0, 1], [2, 3]]
    assert grid_from_frame(((0, 1), (2, 3))) == [[0, 1], [2, 3]]


def test_numpy_2d_and_float_cast():
    assert grid_from_frame(np.array([[4, 5], [6, 7]])) == [[4, 5], [6, 7]]
    assert grid_from_frame(np.array([[1.0, 2.0]])) == [[1, 2]]


def test_frame_stack_takes_last_layer():
    """FrameData.frame = [layer0, layer1] → 取最后一层。"""
    fd = _FD([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])])
    assert grid_from_frame(fd) == [[5, 6], [7, 8]]


def test_real_frame_data():
    pytest.importorskip("arcengine")
    from arcengine import FrameData

    fd = FrameData(levels_completed=0, frame=[np.zeros((4, 4), dtype=int)])
    g = grid_from_frame(fd)
    assert g is not None and len(g) == 4 and len(g[0]) == 4
    assert all(v == 0 for row in g for v in row)


def test_dict_containers():
    assert grid_from_frame({"frame": [[1, 2], [3, 4]]}) == [[1, 2], [3, 4]]
    assert grid_from_frame({"grid": np.array([[9, 9]])}) == [[9, 9]]
    assert grid_from_frame({"observation": [[7, 7]]}) == [[7, 7]]


def test_generic_container_attrs():
    class WithGrid:
        grid = [[1, 1], [2, 2]]

    class WithObservation:
        observation = np.array([[3, 3]])

    assert grid_from_frame(WithGrid()) == [[1, 1], [2, 2]]
    assert grid_from_frame(WithObservation()) == [[3, 3]]


@pytest.mark.parametrize("bad", [None, [], (), 7, "nope"])
def test_unusable_inputs_return_none(bad):
    assert grid_from_frame(bad) is None


def test_is_grid_semantics():
    assert is_grid([[1, 2], [3, 4]])
    assert not is_grid([])              # 空不算网格
    assert not is_grid([1, 2, 3])       # 一维不是网格
    assert not is_grid([[[1]], [[2]]])  # 三维不是网格（是帧栈）
    assert not is_grid("abc")


def test_self_reference_does_not_hang():
    class Self:
        def __init__(self):
            self.frame = self

    assert grid_from_frame(Self()) is None

    a: list = []
    a.append(a)  # 自引用列表
    assert grid_from_frame(a) is None


def test_depth_guard_bounded_nesting():
    """极深嵌套不递归爆栈（受 _MAX_DEPTH 约束，返回 None 而非异常）。"""
    deep: list = [[0]]
    for _ in range(40):
        deep = [deep]
    assert grid_from_frame(deep) is None


# --------------------------------------------------------------------------- #
# 2. 三处委托方结果一致
# --------------------------------------------------------------------------- #


def _as_lists(value):
    """把各入口的返回值（可能是 ndarray）折算成 list[list[int]] 便于比较。"""
    if value is None:
        return None
    return [[int(v) for v in row] for row in np.asarray(value)]


def test_three_entries_agree_on_every_kind_of_frame():
    from lingjing_solo.harness import arc_env
    from lingjing_solo.harness.object_graph import _normalize_frame

    top = _grid(4, [(1, 1, 5)])
    bottom = _grid(4, [(2, 2, 3)])
    samples = {
        "nested_list": _grid(4, [(1, 1, 5)]),
        "numpy": np.array(_grid(4, [(3, 3, 9)]), dtype=np.int16),
        "frame_stack": [top, bottom],
        "multi_layer": [[[0, 1], [2, 3]], [[9, 8], [7, 6]]],
        "fd": _FD([np.array(top)]),
        "dict": {"frame": bottom},
        "empty": [],
        "none": None,
    }
    for name, frame in samples.items():
        a = _as_lists(grid_from_frame(frame))
        b = _as_lists(arc_env.frame_to_grid(frame))
        c = _as_lists(_normalize_frame(frame))
        d = _as_lists(extract_grid(frame))
        # 三个"取不到返回 None"的入口必须一致
        assert a == b == d, f"{name}: 解码不一致 {a!r} / {b!r} / {d!r}"
        # _normalize_frame 是唯一把 None 折成 [] 的那个（build_graph 需要列表）
        assert c == (a if a is not None else []), f"{name}: _normalize_frame {c!r}"


def test_extract_grid_fast_path_preserves_ndarray_identity():
    """已是 ndarray → 原样返回（dtype 不能抖，hash_grid 按字节哈希）。"""
    arr = np.zeros((4, 4), dtype=np.int16)
    assert extract_grid(arr) is arr


def test_extract_grid_returns_numpy_for_frame_data():
    fd = _FD([np.array(_grid(4, [(1, 1, 5)]))])
    out = extract_grid(fd)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 4)
    assert out[1][1] == 5


def test_extract_grid_now_accepts_bare_list():
    """旧实现对裸嵌套 list 返回 None；统一后应能解出（新增能力，行为只增不减）。"""
    assert _as_lists(extract_grid([[0, 1], [2, 3]])) == [[0, 1], [2, 3]]


# --------------------------------------------------------------------------- #
# 3. 接缝消失：裸帧直喂 harness
# --------------------------------------------------------------------------- #


def test_build_graph_accepts_raw_frame_data():
    from lingjing_solo.harness.object_graph import build_graph

    fd = _FD([np.array(_grid(8, [(1, 1, 5), (6, 6, 3)]))])
    g = build_graph(fd)  # 不再需要先 frame_to_grid
    assert g.shape == (8, 8)
    assert {"5", "3"} <= {o.color for o in g.objects}


def test_decide_accepts_raw_frame_data():
    from lingjing_solo.harness.loop import FakeLLM, Harness

    fd = _FD([np.array(_grid(8, [(1, 1, 5), (6, 6, 3)]))])
    dec = Harness(FakeLLM()).decide(fd)  # 裸帧直接进
    assert dec.ok and dec.action is not None
    assert dec.graph.n_objects > 0


def test_run_episode_still_works():
    """回归：env.reset 返回裸 list 的老路径不受影响。"""
    from lingjing_solo.harness.loop import FakeEnv, FakeLLM, Harness

    res = Harness(FakeLLM()).run_episode(FakeEnv(), max_turns=60)
    assert res.solved
