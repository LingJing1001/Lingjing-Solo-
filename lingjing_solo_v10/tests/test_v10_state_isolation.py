"""
test_v10_state_isolation.py — v1.0 根因回归测试（不可删除）

背景：v1.0 中期发现一个隐蔽的浅拷贝 bug ——
   生成的 step() 通常只做外层浅拷贝 `s['objects'] = dict(state['objects'])`，
   内部的每个 obj dict 仍是同一引用；一旦 step 里对 obj 做 o['x']=... 原地修改，
   就会污染调用方的 state，导致 BFS 缓存/搜索树状态互相污染，
   表现为「方向依赖的隐蔽错位」（right/down 看似正确，left/up 失效）。

修复：codegen._deepcopy_state 在 DynamicsProgram.simulate 入口处保证
      传给 step_fn 的 state 与调用方完全独立。

这些测试锁定该不变量，任何"LLM 生成原地修改代码"或"重构 simulate"的改动
一旦破坏隔离，这里会立刻报警。
"""
import os
import sys
import copy
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lingjing_solo.world_model.program import WorldModelProgram, WMPEvidence
from lingjing_solo.world_model.codegen import FakeLLM, _deepcopy_state
from lingjing_solo.world_model.symbols import SymbolTable


def _make_base():
    s = SymbolTable()
    s.avatar_id = "a"
    s.add_object(1, 1, color=1, role="avatar", obj_id="a")
    s.add_object(2, 1, color=2, role="box", obj_id="b")
    return s


def _to_dict(sym):
    return {
        "objects": {oid: {"x": o.x, "y": o.y, "role": o.role}
                    for oid, o in sym.objects.items()},
        "avatar_id": sym.avatar_id,
        "extras": {},
    }


def _feed(wmp, before):
    """喂 4 条「仅 avatar 移动」的转移证据。"""
    for action, (dx, dy) in {"right": (1, 0), "left": (-1, 0),
                              "down": (0, 1), "up": (0, -1)}.items():
        after = copy.deepcopy(before)
        after.objects["a"].x += dx
        after.objects["a"].y += dy
        wmp.learn(WMPEvidence(action=action, before=before, after=after))


class TestDeepCopyUtility(unittest.TestCase):
    """_deepcopy_state 自身正确性。"""

    def test_objects_independent(self):
        a = {"objects": {"a": {"x": 1, "y": 1, "role": "avatar"}}, "extras": {"g": [(4, 1)]}}
        b = _deepcopy_state(a)
        b["objects"]["a"]["x"] = 999
        self.assertEqual(a["objects"]["a"]["x"], 1, "obj 不应被污染")

    def test_extras_independent(self):
        a = {"objects": {}, "extras": {"goals": [(4, 1)]}}
        b = _deepcopy_state(a)
        b["extras"] = {"goals": [(9, 9)]}
        self.assertEqual(a["extras"]["goals"], [(4, 1)], "extras 不应被污染")

    def test_non_dict_passthrough(self):
        self.assertEqual(_deepcopy_state(42), 42)
        self.assertIsNone(_deepcopy_state(None))


class TestSimulateInputIsolation(unittest.TestCase):
    """
    核心不变量：simulate 不得修改调用方传入的 state。

    这是对 v1.0 浅拷贝根因的回归防护 —— 一旦失效，
    Planner 的 BFS 缓存就会因状态污染而算错路径。
    """

    def test_base_unchanged_after_each_direction(self):
        wmp = WorldModelProgram(llm_client=FakeLLM(),
                                min_support=1, confidence_threshold=0.0)
        before = _make_base()
        _feed(wmp, before)
        wmp.compile(llm=FakeLLM())

        base = _to_dict(before)
        expected = {"right": (2, 1), "left": (0, 1), "down": (1, 2), "up": (1, 0)}

        for action, exp in expected.items():
            pred = wmp.simulate(base, action)
            self.assertEqual((pred["objects"]["a"]["x"], pred["objects"]["a"]["y"]), exp,
                             f"{action} 方向预测错误")
            # 关键断言：每次调用后 base 必须仍是初始值
            self.assertEqual(base["objects"]["a"]["x"], 1)
            self.assertEqual(base["objects"]["a"]["y"], 1)

    def test_repeated_calls_independent(self):
        """同一份 state 连续调 right→left，left 必须基于原始 (1,1) 而非被污染的 (2,1)。"""
        wmp = WorldModelProgram(llm_client=FakeLLM(),
                                min_support=1, confidence_threshold=0.0)
        before = _make_base()
        _feed(wmp, before)
        wmp.compile(llm=FakeLLM())

        base = _to_dict(before)
        # 先调 right（若污染，a.x 会变成 2）
        wmp.simulate(base, "right")
        self.assertEqual(base["objects"]["a"]["x"], 1, "right 不应污染 base")
        # 再调 left：必须基于 (1,1) → (0,1)，而非 (2,1) → (1,1)
        pred = wmp.simulate(base, "left")
        self.assertEqual(pred["objects"]["a"]["x"], 0,
                         "left 基于被污染的 state 计算 —— 隔离失效！")


class TestPushSemanticsPreserved(unittest.TestCase):
    """修复不得破坏推箱语义。"""

    def test_box_moves_when_pushed(self):
        wmp = WorldModelProgram(llm_client=FakeLLM(),
                                min_support=1, confidence_threshold=0.0)
        before = _make_base()  # avatar=(1,1), box=(2,1) 相邻
        after = copy.deepcopy(before)
        after.objects["a"].x = 2
        after.objects["b"].x = 3
        wmp.learn(WMPEvidence(action="right", before=before, after=after))
        wmp.compile(llm=FakeLLM())

        pred = wmp.simulate(_to_dict(before), "right")
        self.assertEqual(pred["objects"]["b"]["x"], 3, "box 应被推动")
        self.assertEqual(pred["objects"]["a"]["x"], 2, "avatar 应前进")


if __name__ == "__main__":
    unittest.main(verbosity=2)
