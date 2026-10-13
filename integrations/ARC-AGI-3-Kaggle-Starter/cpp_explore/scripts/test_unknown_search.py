"""本地未知环境搜索压测：直接打 MyAgent._bfs_search / _iddfs_search。

用法（在 Kaggle-Starter 根目录）:
  python cpp_explore/scripts/test_unknown_search.py
  python cpp_explore/scripts/test_unknown_search.py --games cn04,su15 --tl 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path[:0] = [
    str(REPO),
    str(ROOT),
    str(ROOT / "vendor" / "ARC-AGI-3-Agents"),
    str(ROOT / "cpp_explore" / "python"),
]
os.chdir(ROOT)
os.environ.setdefault("LINGJING_SRC", str(REPO / "lingjing_solo"))
os.environ.setdefault("ENGINE_SOLVER", "off")  # 避免 engine 抢入口，专注测搜索核

from arc_agi import Arcade, OperationMode  # noqa: E402
from agent.my_agent import MyAgent, BUILD_TAG, _try_import_lingjing_explore  # noqa: E402


def make_env(gid: str):
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    env = arc.make(gid)
    env.reset()
    return env


def run_one(gid: str, method: str, t_limit: float) -> dict:
    env = make_env(gid)
    agent = object.__new__(MyAgent)  # 绕过完整 Agent 初始化，只测搜索
    fn = getattr(agent, method)
    t0 = time.time()
    found = fn(env, t_limit=t_limit)
    dt = time.time() - t0
    return {
        "game": gid,
        "method": method,
        "found": found is not None,
        "path_len": len(found) if found else 0,
        "path_preview": (found[:12] if found else None),
        "sec": round(dt, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--game",
        default="cn04,su15,lf52",
        help="逗号分隔的未知/半未知局（默认 cn04,su15,lf52）",
    )
    p.add_argument("--tl", type=float, default=20.0, help="每局搜索秒数上限")
    p.add_argument(
        "--methods",
        default="bfs,iddfs",
        help="bfs,iddfs 或其一",
    )
    args = p.parse_args()

    print("=" * 60)
    print("BUILD_TAG:", BUILD_TAG)
    lj = _try_import_lingjing_explore()
    if lj is None:
        print("WARNING: C++ lingjing_explore 未加载，将走 Python 回退")
    else:
        print("C++ explore:", lj.version())
    print("games:", args.game, "tl=", args.tl)
    print("=" * 60)

    methods = []
    for m in args.methods.split(","):
        m = m.strip().lower()
        if m == "bfs":
            methods.append("_bfs_search")
        elif m == "iddfs":
            methods.append("_iddfs_search")

    rows = []
    for gid in [g.strip() for g in args.game.split(",") if g.strip()]:
        for method in methods:
            print(f"\n>>> {gid} / {method}  t_limit={args.tl}s")
            try:
                row = run_one(gid, method, args.tl)
            except Exception as exc:
                print(f"ERROR {type(exc).__name__}: {exc}")
                import traceback

                traceback.print_exc()
                row = {
                    "game": gid,
                    "method": method,
                    "found": False,
                    "path_len": 0,
                    "path_preview": None,
                    "sec": -1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            print("RESULT:", row)

    print("\n" + "=" * 60)
    print("SUMMARY")
    for r in rows:
        flag = "WIN" if r.get("found") else "MISS"
        err = r.get("error", "")
        print(
            f"  {r['game']:6s} {r['method']:14s} {flag:4s} "
            f"len={r.get('path_len', 0):3d}  t={r.get('sec')}s  {err}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
