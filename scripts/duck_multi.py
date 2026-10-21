"""多局轮转：把动作预算花在还有产出的局上。

和 `duck_play.py` 的区别只在"谁决定下一步跑哪局"：单局版一局跑到死，本脚本让
几条 lane 同时挂着，由 `duck/scheduler.py` 按边际产出分配步数。

    # 离线验证管线：三个假动作轮着喂，看轮转和落盘对不对
    python scripts/duck_multi.py --games ar25,cn04,su15 --steps 30 --backend script

    # 真模型
    python scripts/duck_multi.py --games ar25,cn04 --steps 200 \
        --backend openai --url http://127.0.0.1:8001/v1 --model local

注意 `--steps` 是**全局**预算，不是每局预算。这正是调度器存在的意义：总预算固定，
分配方式不同，产物就不同。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc_adaptor"))
sys.path.insert(0, str(ROOT / "scripts"))

from duck.chat import Chat, Endpoint                       # noqa: E402
from duck.scheduler import PriorityScheduler, build_lane    # noqa: E402
from duck_play import SCRIPT, ScriptChat                    # noqa: E402


def _default_lanes() -> int:
    """本地并发上限。显存和文件句柄是瓶颈，不是 CPU，所以取核数的一半。"""
    cpu = os.cpu_count() or 4
    return max(1, min(6, cpu // 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="ar25,cn04",
                    help="逗号分隔的 game_id")
    ap.add_argument("--lanes", type=int, default=0,
                    help=f"同时挂几局（默认 {_default_lanes()}，上限 6）")
    ap.add_argument("--steps", type=int, default=60, help="全局动作预算")
    ap.add_argument("--turns", type=int, default=40, help="每局模型回合上限")
    ap.add_argument("--backend", choices=["script", "openai"], default="script")
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--pin-policy", choices=["drain", "level"], default="level")
    ap.add_argument("--trace", default=str(ROOT / "runs" / "duck_multi" / "trace.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    cap = args.lanes or _default_lanes()
    if len(games) > min(cap, 6):
        print(f"[multi] 请求 {len(games)} 局，并发上限 {min(cap, 6)}；"
              f"截断为 {games[:min(cap, 6)]}", flush=True)
        games = games[:min(cap, 6)]

    lanes = []
    for i, game in enumerate(games):
        if args.backend == "script":
            chat = ScriptChat(SCRIPT * 6)      # 重复几轮，免得一局早早把脚本用完
        else:
            chat = Chat(Endpoint(args.url, args.model), think=args.think)
        lanes.append(build_lane(
            i, game, chat,
            max_steps=max(1, args.steps), max_turns=max(1, args.turns),
            pin_policy=args.pin_policy, verbose=not args.quiet,
        ))

    print(f"[multi] {len(lanes)} 局 / {args.steps} 步 / backend={args.backend} "
          f"/ trace={args.trace}", flush=True)
    sched = PriorityScheduler(
        lanes, total_steps=args.steps,
        log=(lambda s: None) if args.quiet else print,
        trace_path=args.trace,
    )
    report = sched.run()
    print("[multi] " + json.dumps(report, ensure_ascii=False), flush=True)

    for lane in report["lanes"]:
        print(f"[multi]   {lane['game']:<6} 步={lane['steps']:<4} 关={lane['levels']} "
              f"配额={lane['quota']:<4} 产出率={lane['rate']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
