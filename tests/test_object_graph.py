"""对象图：平移不变身份 + 确定性渲染。"""

from __future__ import annotations

import pytest

from lingjing_solo.harness.object_graph import (
    Graph,
    Obj,
    build_graph,
    graph_signature,
    render,
)


def _grid(n=5, cells=()):
    g = [[0] * n for _ in range(n)]
    for r, c, v in cells:
        g[r][c] = v
    return g


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_build_graph_basic_shape_and_order():
    g = build_graph(_grid(6, [(1, 1, 5), (4, 4, 3)]))
    assert g.shape == (6, 6)
    # 背景 0 是最大连通域，包住另外两个 -> 有 kids
    bg = g.objects[0]
    assert bg.color == "0"
    assert bg.n == 36 - 2
    assert len(bg.children) == 2
    # id 按左上角阅读序升序
    assert [o.id for o in g.objects] == sorted(o.id for o in g.objects)


def test_bbox_and_center():
    g = build_graph(_grid(6, [(2, 3, 7)]))
    obj = [o for o in g.objects if o.color == "7"][0]
    assert obj.bbox == (2, 3, 2, 3)
    assert (obj.r, obj.c) == (2, 3)


def test_hash_is_translation_invariant():
    a = build_graph(_grid(8, [(1, 1, 7), (1, 2, 7)]))  # 横躺 1x2
    b = build_graph(_grid(8, [(4, 3, 7), (4, 4, 7)]))  # 同形状，挪到别处
    oa = [o for o in a.objects if o.color == "7"][0]
    ob = [o for o in b.objects if o.color == "7"][0]
    assert oa.hash == ob.hash, "同一形状平移后 hash 必须相同"
    assert (oa.r, oa.c) != (ob.r, ob.c), "位置应当不同"


def test_hash_is_color_sensitive():
    a = build_graph(_grid(6, [(1, 1, 7)]))
    b = build_graph(_grid(6, [(1, 1, 9)]))
    assert [o for o in a.objects if o.color == "7"][0].hash != \
           [o for o in b.objects if o.color == "9"][0].hash


def test_hash_shape_sensitive():
    square = build_graph(_grid(6, [(1, 1, 7), (1, 2, 7), (2, 1, 7), (2, 2, 7)]))
    line = build_graph(_grid(6, [(1, 1, 7), (1, 2, 7), (1, 3, 7), (1, 4, 7)]))
    hs = [o for o in square.objects if o.color == "7"][0].hash
    hl = [o for o in line.objects if o.color == "7"][0].hash
    assert hs != hl


def test_adjacency_present_for_touching_objects():
    g = build_graph(_grid(5, [(1, 1, 7), (1, 2, 8)]))
    assert g.adjacency, "相邻对象应当出现在邻接表里"


def test_empty_frame():
    g = build_graph([])
    assert g.n_objects == 0
    assert render(g).startswith("FRAME 0x0")


def test_multilayer_frame_takes_render_layer():
    top = _grid(4, [(1, 1, 5)])
    bottom = _grid(4, [(2, 2, 3)])
    g = build_graph([top, bottom])
    assert g.shape == (4, 4)
    assert {o.color for o in g.objects} == {"0", "3"}


def test_numpy_input_supported():
    np = pytest.importorskip("numpy")
    arr = np.zeros((6, 6), dtype=np.int16)
    arr[1, 1] = 5
    g = build_graph(arr)
    assert g.shape == (6, 6)
    assert "5" in {o.color for o in g.objects}


def test_as_api_is_json_safe():
    import json

    g = build_graph(_grid(5, [(1, 1, 5)]))
    json.dumps(g.as_api())  # 不抛即通过


# --------------------------------------------------------------------------- #
# 渲染确定性
# --------------------------------------------------------------------------- #


def test_render_is_byte_stable():
    g1 = build_graph(_grid(7, [(1, 1, 5), (5, 5, 3)]))
    g2 = build_graph(_grid(7, [(1, 1, 5), (5, 5, 3)]))
    assert render(g1) == render(g2)
    assert graph_signature(g1) == graph_signature(g2)


def test_render_invariants_when_scene_order_shuffled():
    """同样的场景，构造顺序不同 -> 渲染必须一致。"""
    cells = [(1, 1, 5), (5, 5, 3), (3, 4, 7)]
    a = build_graph(_grid(8, cells))
    b = build_graph(_grid(8, list(reversed(cells))))
    assert render(a) == render(b)


def test_render_changes_when_object_moves():
    a = build_graph(_grid(6, [(1, 1, 5)]))
    b = build_graph(_grid(6, [(2, 2, 5)]))
    assert render(a) != render(b)


def test_render_truncation_is_deterministic():
    cells = [(2 * i, 1, 7) for i in range(1, 6)]  # 行 2/4/6/8/10，彼此不相邻 -> 5 个独立对象
    g = build_graph(_grid(12, cells))
    assert g.n_objects == 6  # 5 个点 + 背景
    out = render(g, max_objects=4)
    assert out == render(g, max_objects=4)
    assert f"(+{g.n_objects - 4} more objects)" in out


def test_render_max_adjacency_limits_line():
    g = build_graph(_grid(6, [(1, 1, 7), (1, 2, 8), (1, 3, 9)]))

    def _adj_pairs(text):
        for line in text.splitlines():
            if line.startswith("adj "):
                return line[4:].split()
        return []

    assert len(_adj_pairs(render(g, max_adjacency=0))) > len(_adj_pairs(render(g, max_adjacency=1)))
    assert len(_adj_pairs(render(g, max_adjacency=1))) == 1


# --------------------------------------------------------------------------- #
# Obj / Graph 的小接口
# --------------------------------------------------------------------------- #


def test_obj_as_dict_roundtrip():
    o = Obj(id=3, color="5", hash="abc", n=4, bbox=(1, 2, 3, 4), children=(7,))
    d = o.as_dict()
    assert d["r"] == 2 and d["c"] == 3
    assert d["children"] == [7]


def test_graph_by_hash_finds_all_matches():
    g = build_graph(_grid(9, [(1, 1, 7), (5, 5, 7)]))
    h = [o for o in g.objects if o.color == "7"][0].hash
    assert len(g.by_hash(h)) == 2


def test_graph_empty_constructor():
    g = Graph(shape=(0, 0), objects=())
    assert g.n_objects == 0
    assert g.as_api() == []
