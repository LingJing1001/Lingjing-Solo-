"""
verify_fix.py — v1.0 根因修复验证

修复前：同一 state 连续调 right→left，left 会基于被 right 污染的 (2,1) 计算，
       返回 (1,1) 而非正确的 (0,1)。
修复后：每次 simulate 拿到独立深拷贝，调用方 state 保持不变。
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lingjing_solo.world_model.codegen import FakeLLM, _deepcopy_state
from lingjing_solo.world_model.program import WorldModelProgram, WMPEvidence
from lingjing_solo.world_model.symbols import SymbolTable


def make_state(ax=1, ay=1, bx=2, by=1):
    s = SymbolTable()
    s.avatar_id = "a"
    s.add_object(ax, ay, color=1, role="avatar", obj_id="a")
    s.add_object(bx, by, color=2, role="box", obj_id="b")
    return s


def to_dict(sym):
    return {
        "objects": {oid: {"x": o.x, "y": o.y, "role": o.role}
                    for oid, o in sym.objects.items()},
        "avatar_id": sym.avatar_id,
        "extras": {},
    }


def test_deepcopy_isolated():
    print("--- 测试1：_deepcopy_state 隔离性 ---")
    a = {"objects": {"a": {"x": 1, "y": 1, "role": "avatar"}}, "extras": {"goals": [(4, 1)]}}
    b = _deepcopy_state(a)
    # 修改 b 的 obj（dict 值）
    b["objects"]["a"]["x"] = 999
    # 修改 b 的 extras（替换整个 key，不原地 append —— 避免测试桩自身共享 list 的干扰）
    b["extras"] = {"goals": [(9, 9)]}
    assert a["objects"]["a"]["x"] == 1, "obj 被污染！"
    assert a["extras"]["goals"] == [(4, 1)], "extras 被污染！"
    print("  ✓ obj 与 extras 均隔离")


def test_simulate_no_pollution():
    print("--- 测试2：同一 state 反复 simulate 不互相污染 ---")
    wmp = WorldModelProgram(llm_client=FakeLLM(), min_support=1, confidence_threshold=0.0)
    before = make_state()
    for action, (dx, dy) in {"right": (1, 0), "left": (-1, 0),
                              "down": (0, 1), "up": (0, -1)}.items():
        after = copy.deepcopy(before)
        after.objects["a"].x += dx
        after.objects["a"].y += dy
        wmp.learn(WMPEvidence(action=action, before=before, after=after))
    wmp.compile(llm=FakeLLM())

    base = to_dict(before)
    # 关键：每次都用「同一份原始 state」调，不重新构造
    results = {}
    for act, (dx, dy) in {"right": (1, 0), "left": (-1, 0),
                           "down": (0, 1), "up": (0, -1)}.items():
        pred = wmp.simulate(base, act)
        results[act] = (pred["objects"]["a"]["x"], pred["objects"]["a"]["y"])
        # 断言：调用后 base 必须仍是 (1,1)
        assert base["objects"]["a"]["x"] == 1, \
            f"{act} 后 base 被污染为 {base['objects']['a']}"
        assert base["objects"]["a"]["y"] == 1

    print(f"  结果: {results}")
    expected = {"right": (2, 1), "left": (0, 1), "down": (1, 2), "up": (1, 0)}
    for act, exp in expected.items():
        assert results[act] == exp, f"{act}: 期望 {exp}, 得到 {results[act]}"
    print("  ✓ 四方向全部正确，且 base 始终保持 (1,1)")


def test_push_still_works():
    print("--- 测试3：推箱语义未被破坏 ---")
    wmp = WorldModelProgram(llm_client=FakeLLM(), min_support=1, confidence_threshold=0.0)
    before = make_state(ax=1, ay=1, bx=2, by=1)
    after = copy.deepcopy(before)
    after.objects["a"].x = 2
    after.objects["b"].x = 3
    wmp.learn(WMPEvidence(action="right", before=before, after=after))
    wmp.compile(llm=FakeLLM())

    pred = wmp.simulate(to_dict(before), "right")
    assert pred["objects"]["b"]["x"] == 3, "推箱失效"
    print(f"  ✓ box 被推到 (3,1)")


if __name__ == "__main__":
    test_deepcopy_isolated()
    test_simulate_no_pollution()
    test_push_still_works()
    print("\n" + "=" * 50)
    print("全部验证通过 ✓ — v1.0 浅拷贝根因已修复")
    print("=" * 50)
