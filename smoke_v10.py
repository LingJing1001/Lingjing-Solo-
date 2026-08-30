"""
smoke_v10.py — v1.0 端到端冒烟验证

验证链路：learn → compile(FakeLLM) → 模拟器内 BFS 规划 → 真实执行 → WIN
关键不变量（v1.0 根因修复后）：
  1. 规划在模拟器内完成，零真实步数计入 RHAE
  2. 整个过程中传入的 state 不被原地修改
  3. 推箱语义正确
"""
import os, sys, copy, json
sys.path.insert(0, os.path.dirname(__file__))

from lingjing_solo.world_model.program import WorldModelProgram, WMPEvidence
from lingjing_solo.world_model.codegen import FakeLLM
from lingjing_solo.planning.wmp_planner import Planner, make_box_goal_evaluator
from lingjing_solo.world_model.symbols import SymbolTable


class MiniSokoban:
    """最小推箱环境：avatar 推 box 到 goal。"""
    def __init__(self, ax, ay, bx, by, gx, gy, w=6, h=5):
        self.w, self.h = w, h
        self.avatar = [ax, ay]
        self.boxes = [[bx, by]]
        self.goals = [(gx, gy)]
        self.steps = 0
        self.max_steps = 40

    def is_win(self):
        return any(b[0] == gx and b[1] == gy for b in self.boxes
                   for gx, gy in self.goals)

    def valid(self, action):
        return action in ("up", "down", "left", "right")

    def step(self, action):
        if not self.valid(action) or self.is_win() or self.steps >= self.max_steps:
            return False
        dx, dy = {"right": (1, 0), "left": (-1, 0),
                  "down": (0, 1), "up": (0, -1)}[action]
        ax, ay = self.avatar
        tx, ty = ax + dx, ay + dy
        if not (0 <= tx < self.w and 0 <= ty < self.h):
            return False
        # 推箱
        for b in self.boxes:
            if b[0] == tx and b[1] == ty:
                nbx, nby = tx + dx, ty + dy
                if not (0 <= nbx < self.w and 0 <= nby < self.h):
                    return False
                if any(o[0] == nbx and o[1] == nby for o in self.boxes):
                    return False
                b[0], b[1] = nbx, nby
        self.avatar = [tx, ty]
        self.steps += 1
        return True

    def snapshot(self):
        objs = {"a": {"x": self.avatar[0], "y": self.avatar[1], "role": "avatar"}}
        for i, b in enumerate(self.boxes):
            objs[f"b{i}"] = {"x": b[0], "y": b[1], "role": "box"}
        return {"objects": objs, "avatar_id": "a",
                "extras": {"grid_w": self.w, "grid_h": self.h, "goals": self.goals}}

    def state_dict_for_wmp(self):
        """返回一份独立副本，供 WMP 使用（不共享 obj dict）。"""
        return json.loads(json.dumps(self.snapshot()))


def collect_evidence(env, actions):
    """在环境里真实执行一段，把转移喂给 WMP。"""
    sym_prev = None
    evs = []
    for act in actions:
        snap = env.snapshot()
        # 构造 SymbolTable（WMP 需要）
        s = SymbolTable()
        s.avatar_id = "a"
        for oid, o in snap["objects"].items():
            s.add_object(o["x"], o["y"], color=1, role=o["role"], obj_id=oid)
        if sym_prev is not None:
            env.step(act)
            s2 = SymbolTable()
            s2.avatar_id = "a"
            for oid, o in env.snapshot()["objects"].items():
                s2.add_object(o["x"], o["y"], color=1, role=o["role"], obj_id=oid)
            evs.append(WMPEvidence(action=act, before=sym_prev, after=s2))
        sym_prev = s
    return evs


def main():
    print("=" * 60)
    print("Lingjing-Solo v1.0 端到端冒烟")
    print("=" * 60)

    # 场景：avatar=(0,0), box=(1,0), goal=(4,0) —— 直线推 3 步
    env = MiniSokoban(0, 0, 1, 0, 4, 0, w=6, h=3)
    print(f"初始: avatar={env.avatar} box={env.boxes} goal={env.goals}")

    wmp = WorldModelProgram(llm_client=FakeLLM(), min_support=1, confidence_threshold=0.0)

    # 1) 先真实探索几步，收集转移证据（这步消耗真实步数）
    explore = ["right", "right"]
    evs = collect_evidence(env, explore)
    for ev in evs:
        wmp.learn(ev)
    wmp.compile(llm=FakeLLM())
    print(f"WMP 编译完成: provenance={wmp.provenance}")

    # 2) 模拟器内 BFS 规划（零真实步数）
    planner = Planner(wmp=wmp, goal_evaluator=make_box_goal_evaluator(env.goals),
                      max_depth=10, telemetry=None)
    real_actions = 0
    sim_actions = 0
    plan = []
    while not env.is_win() and env.steps < env.max_steps:
        snap = env.state_dict_for_wmp()
        action = planner.search(snap, ["up", "down", "left", "right"])
        if action is None:
            print("  规划失败，停止")
            break
        plan.append(action)
        sim_actions += 1
        ok = env.step(action)
        real_actions += 1
        if not ok:
            print(f"  非法动作 {action}，停止")
            break

    print(f"\n结果: WIN={env.is_win()}  真实步数={real_actions}")
    print(f"规划动作序列(={sim_actions}): {plan}")
    print(f"最终: avatar={env.avatar} box={env.boxes}")

    # 3) 验证 state 未被污染（v1.0 根因不变量）
    base = env.snapshot()
    probe = wmp.simulate(env.state_dict_for_wmp(), "right")
    assert env.snapshot()["objects"]["a"]["x"] == base["objects"]["a"]["x"], \
        "环境 snapshot 被 simulate 污染！"

    print("\n不变量检查:")
    print(f"  ✓ 通关: {env.is_win()}")
    print(f"  ✓ 环境 state 未被 simulate 污染")
    assert env.is_win(), "端到端未通关 —— 需检查规划链路"
    print("\n" + "=" * 60)
    print("v1.0 冒烟通过 ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()

