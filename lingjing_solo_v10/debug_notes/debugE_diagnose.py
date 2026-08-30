"""
debugE_diagnose.py — 诊断首层短路为何没生效

在 _bfs 里临时打印 first_layer_best 与各层展开情况。
直接在 Planner 上加 monkey-patch 观察，不修改源码。
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from tests.test_v10_performance import _make_wmp, _state
from lingjing_solo.planning.planner import Planner, make_box_goal_evaluator
from lingjing_solo.world_model.codegen import FakeLLM


# 包裹 WMP，记录 sim 调用次数 + 打印每次调用参数
class TraceWMP:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.first_states = []
        self.is_compiled = True

    def simulate(self, state, action):
        self.calls += 1
        if self.calls <= 6:
            self.first_states.append((action, state["objects"]["a"]["x"],
                                     state["objects"]["a"]["y"]))
        return self.inner.simulate(state, action)


def main():
    wmp = _make_wmp(llm=FakeLLM())
    twmp = TraceWMP(wmp)
    planner = Planner(wmp=twmp,
                      goal_evaluator=make_box_goal_evaluator([(4, 1)]),
                      max_depth=10, adaptive_depth=True)
    state = _state((1, 1), (2, 1), goals=[(4, 1)])
    acts = ["up", "down", "left", "right"]

    print("调用前: a=", state["objects"]["a"], " b=", state["objects"]["b"])
    chosen = planner.search(state, acts)
    print(f"\n选择: {chosen}")
    print(f"simulate 总调用: {twmp.calls}")
    print(f"首 6 次调用 (action, ax, ay):")
    for t in twmp.first_states:
        print(f"   {t}")
    print(f"调用后: a=", state["objects"]["a"], " b=", state["objects"]["b"])

    # 关键：检查 _bfs 是否在第一层就 return
    # 若首层短路生效，sim_calls 应 ≈ 4（仅首层 4 个动作）
    print(f"\n判断: sim_calls={twmp.calls}  {'✓ 短路生效' if twmp.calls <= 8 else '❌ 短路未生效（完整BFS展开）'}")


if __name__ == "__main__":
    main()
