# -*- coding: utf-8 -*-
"""dc22 修复验证：重放「归一化后的 20 步路径」，零副作用（纯离线，只开一个新 env）。

对照点：
  A) 修复前 —— shadow 模块返回私有方言 ("m",3)/("c",x,y)，my_agent 的转换器
     `else: found.append(int(a))` 直接 TypeError，且异常逃到外层 except，
     IDDFS 兜底根本没机会跑。
  B) 修复后 —— shadow_adapter.normalize_path 把方言翻成规范格式 int / (6,x,y)，
     my_agent 消费点（my_agent.py:1583-1596）按同样的方式喂引擎。

本脚本只做 B 的端到端证据：规范路径 → 真引擎 perform_action → 关卡是否推进。
不碰 state/、不写 benchmark json、不调任何线上接口。
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

AGENT_DIR = os.path.join(
    "D:", os.sep + "Quan", "Lingjing-Solo-main-bench",
    "integrations", "ARC-AGI-3-Kaggle-Starter", "agent",
)
sys.path.insert(0, AGENT_DIR)

from arc_agi import Arcade, OperationMode          # noqa: E402
from arcengine import ActionInput, GameAction      # noqa: E402

import shadow_adapter as sa                        # noqa: E402

ENV_DIR = os.path.abspath(
    os.path.join(os.path.dirname(AGENT_DIR), "environment_files")
)


def mk_env():
    arc = Arcade(environments_dir=ENV_DIR, operation_mode=OperationMode.OFFLINE)
    full = [e.game_id for e in arc.get_environments()
            if e.game_id.startswith("dc22")][0]
    env = arc.make(full)
    env.reset()
    return env, full


def show(g, tag):
    return (f"{tag}: level_idx={int(g._current_level_index)} "
            f"score={g._score} state={getattr(g._state, 'name', g._state)}")


def main():
    print("SHADOW_SOLVER_DIR =", os.environ.get("SHADOW_SOLVER_DIR"))
    print("available('dc22') =", sa.available("dc22"))

    env, full = mk_env()
    g = env._game
    print(f"env={full} 起点 {show(g, 'before')}")

    path = sa.solve("dc22", g, t_limit=60.0)
    if not path:
        print("solve() 返回空 —— 无路径可重放")
        return 1
    print("归一化路径:", path)
    print("形状检查:", all(isinstance(a, int) or
                        (isinstance(a, tuple) and len(a) == 3) for a in path))

    # 重放在一个干净 env 上做，不污染求解用的那个 g
    env2, _ = mk_env()
    g2 = env2._game
    li0 = int(g2._current_level_index)
    am = {n: getattr(GameAction, "ACTION%d" % n) for n in range(1, 8)}
    t0 = time.time()
    for i, item in enumerate(path, 1):
        if isinstance(item, tuple):
            act_num, cx, cy = item
            ai = ActionInput(id=am[act_num], data={"x": int(cx), "y": int(cy)},
                             reasoning=None)
        else:
            ai = ActionInput(id=am[item], data={}, reasoning=None)
        g2.perform_action(ai, raw=True)
        print(f"  step{i:02d} {str(item):14s} -> {show(g2, 'after')}")
        if int(g2._current_level_index) > li0 or getattr(g2._state, "name", "") == "WIN":
            print(f"关卡推进/通关于第 {i} 步（耗时 {time.time()-t0:.1f}s）")
            return 0
    print(f"重放完 {len(path)} 步仍未推进 ({time.time()-t0:.1f}s)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
