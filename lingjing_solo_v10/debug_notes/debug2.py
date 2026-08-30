"""debug2.py — 跟踪首层每个动作的 value 计算"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingjing_solo.planning.planner import (
    Planner, make_box_goal_evaluator, state_key, _shallow_copy_state,
)
from lingjing_solo.world_model.program import WorldModelProgram, WMPEvidence
from lingjing_solo.world_model.codegen import FakeLLM
from lingjing_solo.world_model.symbols import SymbolTable


def _state(avatar_xy, box_xy, goals):
    return {
        "objects": {
            "a": {"x": avatar_xy[0], "y": avatar_xy[1], "role": "avatar"},
            "b": {"x": box_xy[0], "y": box_xy[1], "role": "box"},
        },
        "avatar_id": "a",
        "extras": {"goals": list(goals), "grid_w": 8, "grid_h": 6},
    }


wmp = WorldModelProgram(llm_client=FakeLLM(), min_support=1, confidence_threshold=0.0)
before = SymbolTable()
aid = before.add_object(1, 1, color=1, role="avatar"); before.avatar_id = aid
bid = before.add_object(2, 1, color=2, role="box")
after = SymbolTable()
after.add_object(2, 1, color=1, role="avatar", obj_id=aid)
after.add_object(3, 1, color=2, role="box", obj_id=bid); after.avatar_id = aid
wmp.learn(WMPEvidence(action="right", before=before, after=after))
wmp.compile(llm=FakeLLM())
print("provenance:", wmp.provenance)
print("rule_summary:", wmp.rule_summary())

state = _state((1, 1), (2, 1), goals=[(4, 1)])
ev = make_box_goal_evaluator([(4, 1)])

print("\n首层各动作 simulate 结果 + goal_evaluator:")
for act in ["up", "down", "left", "right"]:
    nxt = wmp.simulate(state, act)
    if nxt is None:
        print(f"  {act}: None"); continue
    box = nxt["objects"]["b"]
    av = nxt["objects"]["a"]
    v = ev(nxt)
    print(f"  {act}: avatar=({av['x']},{av['y']}) box=({box['x']},{box['y']}) value={v}")

# 直接用 _apply_move 对比（不带 WMP）
p = Planner(max_depth=2)
print("\n_apply_move (无 WMP) 各动作:")
for act in ["up", "down", "left", "right"]:
    nxt = p._apply_move(state, act)
    box = nxt["objects"]["b"]; av = nxt["objects"]["a"]
    print(f"  {act}: avatar=({av['x']},{av['y']}) box=({box['x']},{box['y']})")
