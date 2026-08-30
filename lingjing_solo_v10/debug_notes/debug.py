"""debug.py — 诊断 chosen 为何是 up 而非 right"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingjing_solo.planning.planner import (
    Planner, make_box_goal_evaluator, state_key, _shallow_copy_state,
)
from lingjing_solo.world_model.program import WorldModelProgram, WMPEvidence
from lingjing_solo.world_model.codegen import FakeLLM
from lingjing_solo.world_model.symbols import SymbolTable


def _state(avatar_xy, box_xy, goals, grid_w=8, grid_h=6, aid="a", bid="b"):
    return {
        "objects": {
            aid: {"x": avatar_xy[0], "y": avatar_xy[1], "role": "avatar"},
            bid: {"x": box_xy[0], "y": box_xy[1], "role": "box"},
        },
        "avatar_id": aid,
        "extras": {"goals": list(goals), "grid_w": grid_w, "grid_h": grid_h},
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

state = _state((1, 1), (2, 1), goals=[(4, 1)])
ev = make_box_goal_evaluator([(4, 1)])

print("goal_evaluator values per action (after WMP step):")
for act in ["up", "down", "left", "right"]:
    nxt = wmp.simulate(state, act)
    if nxt is None:
        print(f"  {act}: None")
        continue
    box = next(o for o in nxt["objects"].values() if o["role"] == "box")
    v = ev(nxt)
    print(f"  {act}: box=({box['x']},{box['y']}) value={v:.1f}")

# 关键：WMP 的 fallback 规则只处理单对象移动，box 会不会动？
print("\nWMP simulate right:", wmp.simulate(state, "right"))
