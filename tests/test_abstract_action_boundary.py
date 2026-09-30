"""planner 出口只给 abstract action name；名字→引擎枚举归边界 adapter（团队规范 §3.2 兼容要求）。

用 AST 查而不是 grep：`agent.py` 的注释里就要写"arcengine"来解释为什么不能在那儿转换，
纯文本匹配会误伤，而真正的违规形态是一条 import 语句。
"""
from __future__ import annotations

import ast
import pathlib
import sys
import types

import pytest

from lingjing_solo import LingjingSoloAgent
from lingjing_solo.core import SoloConfig
from lingjing_solo.harness import kaggle_adapter

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "lingjing_solo"

#: 允许认识引擎枚举的目录：边界 adapter 层。规划/感知/搜索都不许。
BOUNDARY_DIRS = {"harness"}


def _arcengine_imports(path: pathlib.Path) -> list[int]:
    # utf-8-sig：包里 transfer/__init__.py 带 BOM。Python 的 import 能吃 BOM，
    # ast.parse 不能——扫描器要容错，不然报的是工具自己的 SyntaxError。
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    hits: list[int] = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom):
            mod = node.module
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        if mod and "arcengine" in mod:
            hits.append(node.lineno)
    return hits


def test_only_boundary_adapters_import_the_engine():
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if BOUNDARY_DIRS & set(path.relative_to(PACKAGE_ROOT).parts[:-1]):
            continue
        offenders += [f"{path.relative_to(REPO_ROOT)}:{line}" for line in _arcengine_imports(path)]
    assert not offenders, "规划层不得知道引擎枚举，转换请放边界 adapter:\n" + "\n".join(offenders)


def test_planner_imports_do_not_touch_engine():
    assert not _arcengine_imports(PACKAGE_ROOT / "agent.py")


@pytest.fixture
def stub_arcengine(monkeypatch):
    """假引擎枚举：让 adapter 的转换分支能在无 arcengine 的环境里被验到。"""
    class FakeGameAction:
        ACTION1 = "enum:ACTION1"
        ACTION4 = "enum:ACTION4"

        @classmethod
        def from_name(cls, name):
            return getattr(cls, name)

    module = types.ModuleType("arcengine")
    module.GameAction = FakeGameAction
    monkeypatch.setitem(sys.modules, "arcengine", module)
    return FakeGameAction


def test_adapter_converts_names_to_enum(stub_arcengine):
    assert kaggle_adapter.to_game_action("ACTION4") == "enum:ACTION4"
    assert kaggle_adapter.to_game_action("action4") == "enum:ACTION4"   # 入口先 canonicalize
    assert kaggle_adapter.to_game_action("ACTION9") == "ACTION9"        # 转不动就原样给回


def test_adapter_degrades_to_name_without_engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "arcengine", None)   # import 时抛错的最简等价
    assert kaggle_adapter.to_game_action("ACTION4") == "ACTION4"


def _agent_with_returned(monkeypatch, returned: str, **cfg_kwargs):
    agent = kaggle_adapter.MyAgent(cfg=SoloConfig(llm_calls_per_game=0, enable_undo=False,
                                                  **cfg_kwargs))
    monkeypatch.setattr(LingjingSoloAgent, "choose_action",
                        lambda self, frames, latest_frame, valid_actions=None: returned)
    return agent


def test_kaggle_entry_returns_enum_at_the_boundary(monkeypatch, stub_arcengine):
    agent = _agent_with_returned(monkeypatch, "ACTION1")
    assert agent.choose_action([], None) == "enum:ACTION1"


def test_kaggle_entry_can_stay_abstract_when_asked(monkeypatch, stub_arcengine):
    agent = _agent_with_returned(monkeypatch, "ACTION1", return_game_action=False)
    assert agent.choose_action([], None) == "ACTION1"


def test_planner_emit_returns_a_name_even_when_config_asks_for_an_enum(monkeypatch, stub_arcengine):
    """配置里那个 return_game_action=True 不再能改变 planner 的出口类型。"""
    agent = LingjingSoloAgent(cfg=SoloConfig(llm_calls_per_game=0, enable_undo=False,
                                             return_game_action=True))
    assert agent._emit("action4") == "ACTION4"
    assert isinstance(agent._emit("action4"), str)
