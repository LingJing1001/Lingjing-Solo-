"""依次跑 dc22 g50t ka59 lf52 的 shadow solver，提取预存计划。"""
import os, sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lingjing_solo" / "planning"))  # shadow solver deps
VENDOR = KAGGLE / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import inspect

GAMES = ["dc22", "g50t", "ka59", "lf52"]


def get_n_levels(gid):
    import glob
    metas = glob.glob(str(ROOT / "environment_files" / gid / "*" / "metadata.json"))
    if not metas:
        return 0
    with open(metas[0], encoding="utf-8") as f:
        return len(json.load(f).get("baseline_actions", []))


def solve_game(gid):
    n = get_n_levels(gid)
    if n == 0:
        print(f"\n=== {gid}: no levels ===")
        return {}

    mod_name = f"arc_shadow_{gid}"
    try:
        mod = __import__("lingjing_solo.planning." + mod_name, fromlist=[mod_name])
    except Exception as e:
        print(f"\n=== {gid}: import fail: {e} ===")
        return {}

    if hasattr(mod, "_ensure"):
        if not mod._ensure():
            print(f"\n=== {gid}: _ensure failed ===")
            return {}
    if hasattr(mod, "reset"):
        mod.reset()

    g = getattr(mod, "_game", None)
    if g is None:
        print(f"\n=== {gid}: no _game ===")
        return {}

    # 检查 solve_level 签名
    solve_fn = getattr(mod, "solve_level_enhanced", None) or getattr(mod, "solve_level", None)
    if not solve_fn:
        print(f"\n=== {gid}: no solve function ===")
        return {}

    sig = inspect.signature(solve_fn)
    has_max_nodes = "max_nodes" in sig.parameters

    plans = {}
    for lvl in range(n):
        t0 = time.time()
        try:
            if hasattr(mod, "reset"):
                mod.reset()
            g = mod._game
            for _ in range(lvl):
                if hasattr(g, "next_level"):
                    g.next_level()

            if has_max_nodes:
                raw = solve_fn(t_limit=240.0, max_nodes=250000)
            else:
                raw = solve_fn(t_limit=240.0)

            if raw:
                plan = []
                for a in raw:
                    if isinstance(a, (tuple, list)):
                        if len(a) >= 2 and isinstance(a[0], str):
                            tag = a[0].strip().lower()
                            if tag in ("i", "m", "k", "key", "move"):
                                plan.append(int(a[1]))
                            elif tag in ("c", "click"):
                                plan.append({"a": 6, "x": int(a[1]), "y": int(a[2])})
                            elif tag == "r":
                                plan.append(5)
                            else:
                                plan.append(int(a[1]) if len(a) == 2 else 5)
                        elif len(a) == 2:
                            plan.append({"a": 6, "x": int(a[0]), "y": int(a[1])})
                        elif len(a) == 3:
                            plan.append({"a": int(a[0]), "x": int(a[1]), "y": int(a[2])})
                    elif isinstance(a, dict):
                        plan.append({"a": int(a.get("a", 6)), "x": int(a.get("x", 0)), "y": int(a.get("y", 0))})
                    elif isinstance(a, (int, float)):
                        plan.append(int(a))

                if plan:
                    plans[lvl] = plan
                    dt = time.time() - t0
                    print(f"  {gid} L{lvl}: {len(plan)} actions ({dt:.1f}s)")
                else:
                    print(f"  {gid} L{lvl}: raw={len(raw)} but normalize empty")
            else:
                dt = time.time() - t0
                print(f"  {gid} L{lvl}: no solution ({dt:.1f}s)")
        except Exception as e:
            dt = time.time() - t0
            print(f"  {gid} L{lvl}: error: {type(e).__name__}: {e} ({dt:.1f}s)")

    return plans


def main():
    all_plans = {}
    for gid in GAMES:
        print(f"\n=== {gid} ===")
        plans = solve_game(gid)
        if plans:
            all_plans[gid] = plans

    out = ROOT / "_extracted_plans2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_plans, f, ensure_ascii=False, indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"Saved to {out}")
    for gid, plans in all_plans.items():
        total = sum(len(v) for v in plans.values())
        print(f"  {gid}: {len(plans)}/{get_n_levels(gid)} levels, {total} actions")


if __name__ == "__main__":
    main()
