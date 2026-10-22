# -*- coding: utf-8 -*-
"""并发度阶梯实测：并发度 -> 总分/关数/满分局。

为什么必须做
------------
§8.10 实测：同一个 agent、同一个网关、同一批 25 局，
  顺序   -> 1510 分 / 108 关 / 13 局 WIN
  并发25 -> 408.20 分 / 38 关 / 3 局 WIN
**并发把 13 局 WIN 砍成 3 局 WIN。** 而竞赛 rerun 是 25~110 局并发，
比本地更狠。本脚本画出退化曲线，用来给
  ARC_MAX_ACTIONS（每局动作上限）
  ARC_GAME_TIME_LIMIT_MIN（每局墙钟上限）
这两个旋钮定值。

用法（starter venv python）：
    .venv\\Scripts\\python.exe scripts\\concurrency_sweep.py --levels 1,4,8,16,25
    .venv\\Scripts\\python.exe scripts\\concurrency_sweep.py --levels 8 --max-actions 400
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

STARTER = Path(__file__).resolve().parents[1]
CANON = STARTER.parents[1]
VENDOR = STARTER / "vendor" / "ARC-AGI-3-Agents"
TREE_SRC = STARTER / "_sweep_tree"
PORT = int(os.environ.get("SWEEP_PORT", "8033"))

# 需要一起提交的本地脚本（engine solver 依赖 lingjing_solo / arc_adaptor）
EXTRA_MODULES = ["lingjing_solo", "arc_adaptor"]


def _prepare_tree() -> Path:
    if TREE_SRC.exists():
        shutil.rmtree(TREE_SRC, ignore_errors=True)
    shutil.copytree(VENDOR, TREE_SRC)
    shutil.copy2(STARTER / "agent" / "my_agent.py", TREE_SRC / "agents" / "templates" / "my_agent.py")
    for name in ("llm_policy.py", "llm_episode.py"):
        src = STARTER / "agent" / name
        if src.is_file():
            shutil.copy2(src, TREE_SRC / "agents" / "templates" / name)
    for mod in EXTRA_MODULES:
        src = CANON / mod
        if src.is_dir():
            shutil.copytree(src, TREE_SRC / mod)
    (TREE_SRC / "agents" / "__init__.py").write_text(
        "from typing import Type\n"
        "from dotenv import load_dotenv\n"
        "from .agent import Agent, Playback\n"
        "from .swarm import Swarm\n"
        "from .templates.random_agent import Random\n"
        "from .templates.my_agent import MyAgent\n"
        "\n"
        "load_dotenv()\n"
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n"
        "    'random': Random,\n"
        "    'myagent': MyAgent,\n"
        "}\n",
        encoding="utf-8",
    )
    return TREE_SRC


def _run_level(conc: int, max_actions: int, time_limit_min: int, run_seconds: int,
               out_dir: Path) -> dict:
    """跑一个并发度；返回 {conc, total_score, levels, wins, ...}。

    关键设计：**总工作量恒定 = 全部 25 局**，只把并发度从 25 降到 N
    （分批跑，每批 N 局同时开）。这样测到的差异才是纯并发效应，
    而不是"局数变少"。

    分批的代价是每批一个 scorecard（而不是一个 25 局的），
    所以这里把各批的分数直接相加 —— 对"并发度 vs 产出"这个比较是等价的。
    """
    tree = TREE_SRC
    rec = out_dir / f"rec_{conc}"
    (tree / ".env").write_text(
        "SCHEME=http\nHOST=127.0.0.1\n"
        f"PORT={PORT}\n"
        "ARC_API_KEY=test-key-123\n"
        f"ARC_BASE_URL=http://127.0.0.1:{PORT}/\n"
        "OPERATION_MODE=online\n"
        "ENVIRONMENTS_DIR=\n"
        f"RECORDINGS_DIR={rec}\n"
        "AETHER_LLM=0\n"
        "LINGJING_SCRIPT_DEBUG=0\n"
        "SELFLEARN_LLM=0\n"
        "LINGJING_LLM=off\n"
        "LINGJING_SCRIPTS=on\n"
        f"LINGJING_LEXICON_DIR={tree / 'arc_adaptor'}\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(STARTER))
    from arc_agi import Arcade, OperationMode
    from arc_agi.server import create_app

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(STARTER / "environment_files"),
        recordings_dir=str(rec),
    )
    app, rest = create_app(arc, competition_mode=True, save_all_recordings=False)
    holder: dict = {}
    rest.on_scorecard_close = lambda sc: holder.__setitem__("sc", sc)
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False),
        daemon=True,
    ).start()
    for _ in range(60):
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                break
        except OSError:
            time.sleep(0.5)

    env = dict(os.environ)
    env["MPLBACKEND"] = "agg"
    env["PYTHONPATH"] = (f"{tree / 'arc_adaptor'}{os.pathsep}{tree}{os.pathsep}"
                          f"{env.get('PYTHONPATH', '')}")
    env["SELFLEARN_LLM"] = "0"
    env["LINGJING_LLM"] = "off"
    env["LINGJING_SCRIPTS"] = "on"
    env["ARC_MAX_ACTIONS"] = str(max_actions)
    if time_limit_min:
        env["ARC_GAME_TIME_LIMIT_MIN"] = str(time_limit_min)

    games = sorted(p.name for p in (STARTER / "environment_files").iterdir() if p.is_dir())
    waves = [games[i:i + conc] for i in range(0, len(games), conc)]

    total = 0.0
    levels = 0
    wins = 0
    per_game = []
    elapsed = 0.0
    for wi, wave in enumerate(waves):
        holder.clear()
        cmd = [sys.executable, "main.py", "--agent", "myagent", "--game", ",".join(wave)]
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(tree), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        budget = max(60, run_seconds // max(len(waves), 1))
        try:
            proc.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        elapsed += time.time() - t0
        sc = holder.get("sc")
        if sc is None:
            continue
        for e in (sc.model_dump().get("environments") or []):
            for r in (e.get("runs") or []):
                s = float(r.get("score") or 0)
                total += s
                levels += int(r.get("levels_completed") or 0)
                if s > 0:
                    wins += 1
                per_game.append({"game": e.get("id"), "score": s,
                                 "levels": r.get("levels_completed"),
                                 "actions": sum(r.get("level_actions") or [])})
        print(f"    wave {wi + 1}/{len(waves)} ({len(wave)} games) done", flush=True)

    return {"conc": conc, "max_actions": max_actions,
            "time_limit_min": time_limit_min, "games": len(games),
            "total_score": round(total, 2), "levels": levels, "games_with_score": wins,
            "elapsed_sec": round(elapsed, 1), "per_game": per_game}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,8,16,25")
    ap.add_argument("--max-actions", type=int, default=800)
    ap.add_argument("--time-limit-min", type=int, default=0)
    ap.add_argument("--run-seconds", type=int, default=2400)
    ap.add_argument("--out", default="docs/concurrency_sweep.json")
    args = ap.parse_args()

    print("cpu_count:", os.cpu_count(), flush=True)
    tree = _prepare_tree()
    out_dir = STARTER / "_sweep_out"
    out_dir.mkdir(exist_ok=True)
    results = []
    for conc in [int(x) for x in args.levels.split(",") if x.strip()]:
        print(f"=== concurrency {conc} (max_actions={args.max_actions}) ===", flush=True)
        r = _run_level(conc, args.max_actions, args.time_limit_min, args.run_seconds, out_dir)
        results.append(r)
        print(f"  -> score={r['total_score']} levels={r['levels']} "
              f"games_with_score={r['games_with_score']} elapsed={r['elapsed_sec']}s", flush=True)
    out = Path(args.out)
    if not out.is_absolute():
        out = CANON / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
