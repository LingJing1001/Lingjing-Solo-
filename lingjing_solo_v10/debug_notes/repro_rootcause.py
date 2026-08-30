"""
repro_rootcause.py — 精确复现 v1.0 中断时锁定的根因

中断前的判断（待验证）：
    "FakeLLM 生成的 step() 在 right/down 正方向移动时失效——
     av['x']=tx 这行未执行。"

验证方法：构造真实转移证据 → learn → compile(llm=FakeLLM()) → simulate，
然后逐个方向检查 avatar 坐标是否按预期变化。

约定：方向映射 right=(+1,0), down=(0,+1), left=(-1,0), up=(0,-1)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lingjing_solo.world_model.program import WorldModelProgram, WMPEvidence
from lingjing_solo.world_model.codegen import FakeLLM
from lingjing_solo.world_model.symbols import SymbolTable


def make_state(aid="a", bid="b", ax=1, ay=1, bx=2, by=1):
    """构造一个最简单的 avatar+box 状态（与 test_v10_performance 一致）。"""
    before = SymbolTable()
    before.avatar_id = aid
    before.add_object(ax, ay, color=1, role="avatar", obj_id=aid)
    before.add_object(bx, by, color=2, role="box", obj_id=bid)
    return before


def shifted(sym, oid, dx, dy):
    """拷贝一份 SymbolTable 并把 oid 平移 (dx,dy)。"""
    import copy
    out = copy.deepcopy(sym)
    obj = out.objects[oid]
    obj.x = obj.x + dx
    obj.y = obj.y + dy
    return out


def state_dict_from_symbol(sym, aid="a"):
    """把 SymbolTable 转成 simulate() 用的 dict 契约。"""
    objs = {}
    for oid, o in sym.objects.items():
        objs[oid] = {"x": o.x, "y": o.y, "role": o.role}
    return {"objects": objs, "avatar_id": aid, "extras": {}}


def main():
    print("=" * 60)
    print("复现：learn → compile(FakeLLM) → simulate 各方向")
    print("=" * 60)

    wmp = WorldModelProgram(llm_client=FakeLLM(),
                            min_support=1, confidence_threshold=0.0)

    aid, bid = "a", "b"
    before = make_state(aid=aid, bid=bid, ax=1, ay=1, bx=2, by=1)

    # 喂 4 条转移：每条 = avatar 单方向移动 1 步，box 不动
    deltas = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}
    for action, (dx, dy) in deltas.items():
        after = shifted(before, aid, dx, dy)   # 只平移 avatar
        wmp.learn(WMPEvidence(action=action, before=before, after=after))

    ok = wmp.compile(llm=FakeLLM())
    print(f"\ncompile() -> {ok}  provenance={wmp.provenance}")

    base = make_state(aid=aid, bid=bid, ax=1, ay=1, bx=2, by=1)
    base_dict = state_dict_from_symbol(base)

    print("\n--- simulate 各方向（初始 avatar=(1,1), box=(2,1)）---")
    failures = []
    for action, (dx, dy) in deltas.items():
        pred = wmp.simulate(base_dict, action)
        if pred is None:
            print(f"  {action:6s}: simulate 返回 None  ❌")
            failures.append(action)
            continue
        av = pred["objects"][aid]
        expected = (1 + dx, 1 + dy)
        ok_dir = (av["x"], av["y"]) == expected
        print(f"  {action:6s}: avatar=({av['x']},{av['y']})  "
              f"expected={expected}  {'✓' if ok_dir else '❌'}")
        if not ok_dir:
            failures.append(action)

    # ---- 关键测试：推箱 ----
    print("\n--- simulate 'right' 推箱（avatar=(1,1), box=(2,1) 相邻）---")
    pred = wmp.simulate(base_dict, "right")
    if pred is None:
        print("  返回 None ❌")
        failures.append("push")
    else:
        av = pred["objects"][aid]
        box = pred["objects"][bid]
        print(f"  avatar=({av['x']},{av['y']})  box=({box['x']},{box['y']})")
        if box["x"] == 3:
            print("  ✓ box 被正确推动到 (3,1)")
        else:
            print("  ❌ box 未被推动（push 语义缺失）")
            failures.append("push")

    print("\n" + "=" * 60)
    if failures:
        print(f"复现成功：失效方向/语义 = {failures}")
    else:
        print("未复现：所有方向均正确")
    print("=" * 60)


if __name__ == "__main__":
    main()
