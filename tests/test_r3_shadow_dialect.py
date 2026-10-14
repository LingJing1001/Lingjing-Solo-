"""R3 专用求解器「方言 → 规范格式」边界测试（2026-10-13 修复的回归网）。

背景
----
`shadow_adapter.solve` 的契约是返回 ``int`` 或 ``(6, x, y)``；但
``agent/my_agent.py`` 里内联的转换循环只认 ``len(a) == 2`` 的二元组，
归一后的 ``(6, x, y)`` 三元组落进 ``int(a)`` 分支抛 TypeError，
整段 R3 **连同其后的 IDDFS 兜底**被外层 ``except`` 吞掉。

实测日志（修复前，25 局离线跑分）：``r3-raised gid=dc22/ka59/lf52(×2)/cn04
TypeError: int() argument must be a string, ... not 'tuple'``。
探针 (`_probe_r3_dialect.py`) 进一步证明：dc22 的 20 步、ka59 的 11 步、
lf52 的 8 步**完整解法**当时已被算出来，纯粹死在转换层。

第二处：``arc_shadow_sp80.solve_level`` 以 ``list(path) + [("p", 5)]``
收尾，"p"（放水宏）方言缺分支 → 整条 4 步解法被判为"方言不认识"作废。

本文件是**静态 + 纯函数**测试，不 import 也不运行引擎
（``my_agent`` 的函数通过 AST 抽取后执行），因此在没有 ``arcengine``
的普通 CI 里也能跑。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STARTER = REPO / "integrations" / "ARC-AGI-3-Kaggle-Starter"
AGENT_SRC = STARTER / "agent" / "my_agent.py"
ADAPTER_SRC = STARTER / "agent" / "shadow_adapter.py"

#: 修复前的转换层实现（逐字保留），用来证明"旧实现确实会炸"这条前提。
_LEGACY_CONVERTER = """
def _legacy_r3_converter(path):
    found = []
    for a in path:
        if isinstance(a, dict):
            found.append((a.get("a", 6), a.get("x", 0), a.get("y", 0)))
        elif isinstance(a, tuple) and len(a) == 2:
            found.append((6, a[0], a[1]))
        else:
            found.append(int(a))
    return found
"""


# ── 载入被测对象 ───────────────────────────────────────────────


def _load_adapter():
    """按文件路径加载 shadow_adapter（纯 stdlib，可安全 import）。"""
    name = "shadow_adapter_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ADAPTER_SRC)
    assert spec and spec.loader, f"cannot load {ADAPTER_SRC}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _agent_source() -> str:
    return AGENT_SRC.read_text(encoding="utf-8")


def _extract_agent_function(func_name: str):
    """从 my_agent.py 抽出某个模块级函数并单独执行（不 import 引擎）。"""
    tree = ast.parse(_agent_source())
    node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name),
        None,
    )
    assert node is not None, f"{func_name} not found at module level in my_agent.py"
    ns: dict = {}
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(AGENT_SRC), "exec"), ns)
    return ns[func_name]


@pytest.fixture(scope="module")
def adapter():
    return _load_adapter()


@pytest.fixture(scope="module")
def legacy_converter():
    ns: dict = {}
    exec(_LEGACY_CONVERTER, ns)
    return ns["_legacy_r3_converter"]


@pytest.fixture(scope="module")
def normalize_r3_path():
    return _extract_agent_function("_normalize_r3_path")


# ── 前提：旧实现确实会炸（防止"测试其实没覆盖到真问题"）────────


def test_legacy_converter_really_raises_on_three_tuple(legacy_converter):
    """把 bug 本身固化下来：旧实现在 (6, x, y) 上必抛 TypeError。"""
    with pytest.raises(TypeError):
        legacy_converter([1, (6, 48, 36), 1])


# ── shadow_adapter 方言表 ─────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        # 纯数字 / 朴素养
        (3, 3),
        ([1, 2], (6, 1, 2)),
        ((13, 30), (6, 13, 30)),
        ((6, 13, 30), (6, 13, 30)),
        # dict 方言
        ({"a": 6, "x": 1, "y": 2}, (6, 1, 2)),
        ({"a": 4, "x": 0, "y": 0}, 4),
        ({"action": 6, "x": 7, "y": 8}, (6, 7, 8)),
        # 方言 A：("m"/"k"/"key"/"move", n)
        (("m", 3), 3),
        (("M", 1), 1),
        (("key", 4), 4),
        # 方言 B：("c"/"click", x, y)
        (("c", 13, 30), (6, 13, 30)),
        (("click", 0, 0), (6, 0, 0)),
        # 方言 C：("i", n)
        (("i", 5), 5),
        # 方言 D：("r",)
        (("r",), 5),
        # 方言 E：("p", n) —— sp80 放水宏（本次新增）
        (("p", 5), 5),
        (("p", 1), 1),
        (("pour", 5), 5),
        (("p",), 5),
        # 不可识别 → None
        (("q", 1), None),
        (("p", 9), None),
        (("p", "x"), None),
        (("m", 0), None),
        (True, None),
        ("3", None),
        ((1, 2, 3, 4), None),
    ],
)
def test_normalize_action_dialects(adapter, raw, expected):
    assert adapter.normalize_action(raw) == expected


def test_normalize_path_sp80_pour_path_is_accepted(adapter):
    """修复点：sp80 的 4 步解法（以 ("p", 5) 收尾）必须整条通过。"""
    raw = [("m", 4), ("m", 4), ("m", 4), ("p", 5)]
    assert adapter.normalize_path(raw) == [4, 4, 4, 5]


def test_normalize_path_rejects_unknown_dialect_wholesale(adapter):
    """未知方言仍然整条作废（绝不允许半条解法被执行）。"""
    assert adapter.normalize_path([("m", 4), ("zzz", 1)]) is None


def test_normalize_path_empty_is_valid(adapter):
    """``[]`` = 求解器说"这关已经过了"，是合法值不是失败。"""
    assert adapter.normalize_path([]) == []


# ── my_agent._normalize_r3_path ──────────────────────────────


def test_three_tuple_passes_through(normalize_r3_path):
    """核心回归：归一后的 (6, x, y) 不能再被 int() 吃掉。"""
    assert normalize_r3_path([1, (6, 48, 36), 1]) == [1, (6, 48, 36), 1]


def test_two_tuple_becomes_click(normalize_r3_path):
    assert normalize_r3_path([(13, 30)]) == [(6, 13, 30)]


def test_dict_becomes_click(normalize_r3_path):
    assert normalize_r3_path([{"a": 6, "x": 19, "y": 20}]) == [(6, 19, 20)]


def test_plain_int_stays_int(normalize_r3_path):
    assert normalize_r3_path([1, 2, 3]) == [1, 2, 3]


def test_list_form_three_tuple(normalize_r3_path):
    assert normalize_r3_path([[6, 19, 20]]) == [(6, 19, 20)]


def test_unknown_arity_raises_clearly(normalize_r3_path):
    with pytest.raises(ValueError):
        normalize_r3_path([(1, 2, 3, 4)])


@pytest.mark.parametrize("gid_key", ["dc22", "ka59", "lf52", "cn04"])
def test_end_to_end_chain_for_the_four_victims(adapter, normalize_r3_path, gid_key):
    """端到端：solver 内层方言 → normalize_path → _normalize_r3_path 全程不炸。

    方言**种类**来自 2026-10-13 探针从真实求解器抓下的输出：dc22 是
    ``("m", n)`` / ``("c", x, y)``，ka59 是纯 int 混点击，lf52 是 dict 点击，
    cn04（当时唯一走通的一局）是 ``("i", n)`` / ``("m", n)``。这里把它们串成
    序列，钉住"经归一后必是 int 或 (6,x,y)、绝不再抛 TypeError"这条不变式。
    """
    inner_dialects = {
        "dc22": [("m", 1), ("c", 48, 36), ("m", 1), ("m", 1)],
        "ka59": [4, 4, 4, 3, 3, 3, {"a": 6, "x": 5, "y": 5}],
        "lf52": [{"a": 6, "x": 19, "y": 20}, {"a": 6, "x": 31, "y": 20}],
        "cn04": [("i", 5), ("i", 5), ("i", 5), ("m", 4)],
    }
    raw = inner_dialects[gid_key]
    normalized = adapter.normalize_path(raw)
    assert normalized is not None, f"{gid_key}: 方言表漏了一种格式 {raw!r}"
    final = normalize_r3_path(normalized)
    assert len(final) == len(raw)
    for item in final:
        assert isinstance(item, int) or (
            isinstance(item, tuple) and len(item) == 3 and item[0] == 6
        ), f"{gid_key}: 出口格式不合法 {item!r}"


# ── 静态守卫：防止修复被回退 ─────────────────────────────────


def test_agent_calls_shared_normalizer():
    src = _agent_source()
    assert "_normalize_r3_path(path)" in src, (
        "my_agent.py 的 R3 出口不再调用 _normalize_r3_path —— 修复被回退了"
    )


def test_shadow_solver_block_has_own_try():
    """专用求解器必须有自己的 try，异常不得连累 IDDFS 兜底。"""
    src = _agent_source()
    assert "r3-shadow-raised" in src, (
        "R3 的 shadow_solver 段缺少独立 except —— 一次方言异常会再次吞掉 IDDFS 兜底"
    )


def test_adapter_handles_pour_dialect_statically():
    src = ADAPTER_SRC.read_text(encoding="utf-8")
    assert '"p"' in src and "pour" in src, (
        "shadow_adapter 的 p/pour 方言分支被删掉了（sp80 会再次整条作废）"
    )
