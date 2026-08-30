"""
debugF_bfs.py — 最小复现：直接调 _bfs，打印每层 frontier 大小与返回值
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from tests.test_v10_performance import _make_wmp, _state
from lingjing_solo.planning.planner import Planner, make_box_goal_evaluator
from lingjing_solo.world_model.codegen import FakeLLM


# Monkey-patch _bfs 加探针
from lingjing_solo.planning import planner as P

orig_bfs = P.Planner._bfs

def probed_bfs(self, state, actions, source):
    print(f"\n[_bfs 进入] source={source}  depth_limit={self._effective_depth()}")
    # 临时替换 range 循环体无法做到，改为在 _cached_successor 探针
    return orig_bfs(self, state, actions, source)

P.Planner._bfs = probed_bfs


# 探针 _cached_successor
orig_cs = P.Planner._cached_successor
call_count = [0]
def probed_cs(self, state, action, source):
    call_count[0] += 1
    if call_count[0] <= 3 or (source == "transition"):
        print(f"  _cs#{call_count[0]} source={source} act={action} "
              f"a=({state['objects']['a']['x']},{state['objects']['a']['y']})")
    return orig_cs(self, state, action, source)
P.Planner._cached_successor = probed_cs


def main():
    wmp = _make_wmp(llm=FakeLLM())
    planner = Planner(wmp=wmp,
                      goal_evaluator=make_box_goal_evaluator([(4, 1)]),
                      max_depth=10, adaptive_depth=True)
    state = _state((1, 1), (2, 1), goals=[(4, 1)])
    acts = ["up", "down", "left", "right"]

    chosen = planner.search(state, acts)
    print(f"\n返回: {chosen}")
    print(f"_cached_successor 总调用: {call_count[0]}")
    print(f"cache_hits={planner.cache_hits} cache_misses={planner.cache_misses}")


if __name__ == "__main__":
    main()
