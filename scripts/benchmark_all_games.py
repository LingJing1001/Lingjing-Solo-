"""25 局跑分基准：官方 Agent 口径驱动 SmartRouter（my_agent.py）实跑。

用法（对齐团队约定 CLI）：
  python scripts/benchmark_all_games.py 400 \
      --agent agent/my_agent.py \
      --out ui/static/games_benchmark_latest.json \
      --label smart-router-local

口径照抄 vendor/ARC-AGI-3-Agents 的 Agent.main() 循环：
  while not is_done(frames, frames[-1]) and steps < N:
      action = choose_action(frames, frames[-1])
      frame  = do_action_request(action)   # env.step(action, data=action.action_data...)
  差异仅两点：步数上限用 CLI 参数（官方是 MAX_ACTIONS=800）；离线本地引擎
  （Arcade OFFLINE, seed=0，与 tests/test_route_replay.py 同款）。

输出 JSON：{label, agent, n_steps, build_tag, generated_at,
            results:[{game_id, max_levels, final_levels, state, steps, done}],
            summary:{total_levels, nonzero_games, wins}}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "vendor" / "ARC-AGI-3-Agents"))

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import FrameData, GameState  # noqa: E402

from lingjing_solo.transfer.miner import ENV_DIR, gid_to_full_id  # noqa: E402

STARTER = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
VENDOR = STARTER / "vendor" / "ARC-AGI-3-Agents"

AGENT_CANDIDATES = (
    STARTER / "agent" / "my_agent.py",
    ROOT / "my_agent.py",
    VENDOR / "agents" / "templates" / "my_agent.py",
)


def resolve_agent(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / path).resolve()
    if p.is_file():
        return p
    for cand in AGENT_CANDIDATES:
        if cand.is_file():
            print(f"[bench] --agent {path} 不存在，回退 {cand}")
            return cand
    raise FileNotFoundError(f"agent 文件找不到: {path}")


def load_agent_class(agent_py: Path):
    """按路径加载 MyAgent；先补 sys.path（vendors 的 agents 包 + lingjing_solo）。"""
    for extra in (VENDOR, agent_py.parents[1], ROOT):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location("bench_my_agent", agent_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # 不注册的话 BUILD_TAG 等模块常量取不到
    spec.loader.exec_module(mod)
    return mod.MyAgent


def convert_raw(raw: Any) -> Optional[FrameData]:
    if raw is None:
        return None
    return FrameData(
        game_id=raw.game_id,
        frame=[arr.tolist() for arr in raw.frame],
        state=raw.state,
        levels_completed=raw.levels_completed,
        win_levels=raw.win_levels,
        guid=raw.guid,
        full_reset=raw.full_reset,
        available_actions=raw.available_actions,
    )


def action_kwargs(action: Any) -> dict:
    """复刻官方 do_action_request 的 data/reasoning 提取。"""
    data: dict = {}
    ad = getattr(action, "action_data", None)
    if ad is not None and hasattr(ad, "model_dump"):
        try:
            data = ad.model_dump() or {}
        except Exception:
            data = {}
    reasoning = getattr(action, "reasoning", None)
    if reasoning is not None and not isinstance(reasoning, dict):
        reasoning = {"text": str(reasoning)}
    return {"data": data, "reasoning": reasoning}


def run_one(arc: Any, agent_cls: type, full_id: str, n_steps: int) -> dict:
    env = arc.make(full_id, seed=0, save_recording=False)
    agent = agent_cls("bench-card", full_id, "smart-router-bench", "local", False, env)
    frames: list[FrameData] = agent.frames
    frames[0] = convert_raw(env.reset()) or frames[0]
    max_levels = int(frames[0].levels_completed)
    steps = none_frames = 0
    t0 = time.time()
    done = False
    while not agent.is_done(frames, frames[-1]) and steps < n_steps:
        action = agent.choose_action(frames, frames[-1])
        steps += 1
        agent.action_counter += 1
        raw = env.step(action, **action_kwargs(action))
        frame = convert_raw(raw)
        if frame is None:
            none_frames += 1
            if none_frames >= 5:  # 引擎连续拒绝：止损，避免死循环
                break
            continue
        none_frames = 0
        agent.append_frame(frame)
        max_levels = max(max_levels, int(frame.levels_completed))
        if frame.state is GameState.WIN:
            done = True
            break
    return {
        "game_id": full_id,
        "max_levels": max_levels,
        "final_levels": int(frames[-1].levels_completed),
        "state": frames[-1].state.name,
        "steps": steps,
        "win": frames[-1].state is GameState.WIN,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n_steps", type=int, nargs="?", default=400)
    ap.add_argument("--agent", default="agent/my_agent.py")
    ap.add_argument("--out", default="ui/static/games_benchmark_latest.json")
    ap.add_argument("--label", default="smart-router-local")
    ap.add_argument("--games", default="", help="逗号分隔 gid 子集（冒烟用）")
    args = ap.parse_args()

    logging.getLogger("arc_agi").setLevel(logging.CRITICAL)
    agent_py = resolve_agent(args.agent)
    agent_cls = load_agent_class(agent_py)
    arc = Arcade(environments_dir=str(ENV_DIR), logger=logging.getLogger("bench"),
                 operation_mode=OperationMode.OFFLINE)

    gids = ([g.strip() for g in args.games.split(",") if g.strip()]
            if args.games else sorted(p.name for p in ENV_DIR.iterdir() if p.is_dir()))
    results = []
    for gid in gids:
        full = gid_to_full_id(gid)
        if full is None:
            print(f"[bench] 跳过 {gid}: 目录里没有 hash 子目录", flush=True)
            continue
        try:
            r = run_one(arc, agent_cls, full, args.n_steps)
        except Exception as exc:  # noqa: BLE001 — 单局崩了记失败，继续跑
            r = {"game_id": full, "max_levels": 0, "final_levels": 0,
                 "state": f"ERROR:{type(exc).__name__}: {exc}", "steps": 0,
                 "win": False, "elapsed_s": 0.0}
        results.append(r)
        print(f"  {r['game_id']:18s} levels={r['max_levels']:2d} "
              f"state={r['state']:<14s} steps={r['steps']:3d} "
              f"({r['elapsed_s']}s)", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    build_tag = getattr(sys.modules.get("bench_my_agent"), "BUILD_TAG", None)
    payload = {
        "label": args.label,
        "agent": str(agent_py.relative_to(ROOT)) if agent_py.is_relative_to(ROOT) else str(agent_py),
        "n_steps": args.n_steps,
        "build_tag": build_tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "summary": {
            "total_levels": sum(r["max_levels"] for r in results),
            "nonzero_games": sum(1 for r in results if r["max_levels"] > 0),
            "wins": sum(1 for r in results if r["win"]),
            "n_games": len(results),
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    s = payload["summary"]
    print(f"\n== {args.label} ==\n总分(关) {s['total_levels']}  破零 {s['nonzero_games']}/{s['n_games']}"
          f"  WIN {s['wins']}  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
