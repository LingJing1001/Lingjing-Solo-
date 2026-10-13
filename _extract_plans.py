"""提取 11 个失败游戏的预存计划 — 用 arc_agi env 跑 shadow solver BFS。"""
import os, sys, json, time, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))
VENDOR = KAGGLE / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState

FAILING = ["tu93", "sp80", "sk48", "s5i5", "re86", "m0r0", "lf52", "ka59", "g50t", "dc22", "cn04"]


def get_n_levels(gid):
    import glob
    metas = glob.glob(str(ROOT / "environment_files" / gid / "*" / "metadata.json"))
    if not metas:
        return 0
    with open(metas[0], encoding="utf-8") as f:
        return len(json.load(f).get("baseline_actions", []))


def normalize_action(item):
    """归一化动作 → (act_id, x, y) 或 (act_id, 0, 0)。"""
    if isinstance(item, (list, tuple)):
        if len(item) == 1:
            return (int(item[0]), 0, 0)
        if len(item) == 2:
            return (6, int(item[0]), int(item[1]))
        if len(item) == 3:
            return (int(item[0]), int(item[1]), int(item[2]))
    elif isinstance(item, dict):
        a = int(item.get("a", item.get("action", 6)))
        x = int(item.get("x", 0))
        y = int(item.get("y", 0))
        return (a, x, y)
    elif isinstance(item, int):
        return (item, 0, 0)
    return None


def verify_plan(arc, gid, plan, level):
    """用 arc_agi 引擎回放验证。"""
    env = arc.make(gid)
    if env is None:
        return False
    frame = env.reset()
    for act_id, x, y in plan:
        if act_id == 6:
            frame = env.step(GameAction.ACTION6, data={"x": x, "y": y})
        else:
            frame = env.step(GameAction.from_id(act_id), data={})
        if frame.levels_completed > level or frame.state.name == "WIN":
            return True
    return frame.levels_completed > level or frame.state.name == "WIN"


def solve_game(arc, gid):
    """用 shadow solver 解一个游戏的所有关。"""
    n = get_n_levels(gid)
    if n == 0:
        return {}

    mod_name = "arc_shadow_" + gid
    try:
        mod = __import__("lingjing_solo.planning." + mod_name, fromlist=[mod_name])
    except Exception as e:
        print(f"  {gid}: import fail: {e}")
        return {}

    solve_fn = getattr(mod, "solve_level_enhanced", None) or getattr(mod, "solve_level", None)
    if not solve_fn:
        print(f"  {gid}: no solve function")
        return {}

    plans = {}
    for lvl in range(n):
        t0 = time.time()
        try:
            # 创建 env 并设置到 shadow module
            env = arc.make(gid)
            if env is None:
                continue
            frame = env.reset()
            # 跳到目标关
            game = env._game
            for _ in range(lvl):
                if hasattr(game, "next_level"):
                    game.next_level()
                else:
                    break
            # 设置 shadow module 的 _game 和 _env
            if hasattr(mod, "_game"):
                mod._game = game
            if hasattr(mod, "_env"):
                mod._env = env
            # 求解
            raw = solve_fn(t_limit=60)
            if raw:
                plan = []
                for item in raw:
                    norm = normalize_action(item)
                    if norm:
                        plan.append(norm)
                if plan:
                    # 验证
                    ok = verify_plan(arc, gid, plan, lvl)
                    if ok:
                        plans[lvl] = plan
                        print(f"  {gid} L{lvl}: {len(plan)} actions OK  {time.time()-t0:.1f}s")
                    else:
                        print(f"  {gid} L{lvl}: {len(plan)} actions VERIFY FAIL  {time.time()-t0:.1f}s")
                else:
                    print(f"  {gid} L{lvl}: raw={len(raw)} but normalize empty  {time.time()-t0:.1f}s")
            else:
                print(f"  {gid} L{lvl}: no solution  {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  {gid} L{lvl}: error: {type(e).__name__}: {e}  {time.time()-t0:.1f}s")
    return plans


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_plans = {}
    for gid in FAILING:
        n = get_n_levels(gid)
        print(f"\n=== {gid} ({n} levels) ===")
        plans = solve_game(arc, gid)
        if plans:
            all_plans[gid] = plans

    out = ROOT / "_extracted_plans.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_plans, f, ensure_ascii=False, indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"保存到 {out}")
    for gid, plans in all_plans.items():
        total = sum(len(v) for v in plans.values())
        print(f"  {gid}: {len(plans)}/{get_n_levels(gid)} levels, {total} actions")
    solved = sum(len(v) for v in all_plans.values())
    print(f"总计: {solved} levels solved")


if __name__ == "__main__":
    main()
