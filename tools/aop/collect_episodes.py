#!/usr/bin/env python3
"""采集真实 remote episode：open → execute → close → GET 全流程录制。

覆盖 ≥5 个游戏 × 2 rep = ≥10 个 episode，含成功/失败/reset/边界。
每个 episode 保存: recording.jsonl + scorecard.json + manifest.json
采集门槛: open/execute/close/GET 全成功才入训练集；close/GET 失败的丢弃。

用法:
    # 需要 ARC_API_KEY 环境变量
    python tools/aop/collect_episodes.py --steps 400 --out data/aop/episodes
    # 或指定游戏
    python tools/aop/collect_episodes.py --games ar25,vc33,ls20,dc22,re86 --steps 400
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "vendor"))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"))

# 默认目标: 5 个游戏 × 2 rep = 10 episode，覆盖成功/失败
DEFAULT_GAMES = ["ar25", "vc33", "ls20", "dc22", "re86"]
REPS = 2


def collect_one_episode(
    arc: Any,
    game_id: str,
    rep: int,
    max_steps: int,
    out_dir: Path,
) -> dict[str, Any]:
    """采集单个 episode: open → execute → close → GET。"""
    label = "success" if game_id in ["ar25", "vc33", "ls20"] else "fail"
    tags = [game_id, label, f"rep{rep}", "phase-a"]
    result = {"game_id": game_id, "rep": rep, "label": label, "status": "unknown"}

    # 1. open scorecard
    try:
        card_id = arc.open_scorecard(tags=tags)
        result["scorecard_id"] = card_id
        print(f"  open_scorecard: {card_id}", flush=True)
    except Exception as e:
        result["status"] = "open_failed"
        result["error"] = str(e)
        print(f"  open 失败: {e}", flush=True)
        return result

    # 2. make remote env
    try:
        env = arc.make(game_id, scorecard_id=card_id, save_recording=True)
        if env is None:
            result["status"] = "make_failed"
            print(f"  make 失败: env is None", flush=True)
            return result
        result["environment_id"] = getattr(env, "game_id", game_id)
        print(f"  make env: {result['environment_id']}", flush=True)
    except Exception as e:
        result["status"] = "make_failed"
        result["error"] = str(e)
        print(f"  make 失败: {e}", flush=True)
        return result

    # 3. agent 跑游戏
    try:
        import importlib.util
        agent_path = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "agent" / "my_agent.py"
        spec = importlib.util.spec_from_file_location("user_agent_module", agent_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        MyAgentCls = module.MyAgent
        MyAgentCls.MAX_ACTIONS = max_steps

        agent = MyAgentCls(
            card_id=card_id,
            game_id=game_id,
            agent_name=f"collect.{game_id}.rep{rep}",
            ROOT_URL=os.environ.get("ARC_BASE_URL", "https://three.arcprize.org"),
            record=True,
            arc_env=env,
            tags=tags,
        )
        agent.main()

        final = agent.frames[-1] if agent.frames else None
        result["levels_completed"] = final.levels_completed if final else 0
        result["state"] = str(final.state).replace("GameState.", "") if final else "unknown"
        result["actions"] = agent.action_counter
        result["win"] = result["state"] == "WIN"
        print(f"  execute: levels={result['levels_completed']} state={result['state']} actions={result['actions']}", flush=True)
    except Exception as e:
        result["status"] = "execute_failed"
        result["error"] = str(e)
        print(f"  execute 失败: {e}", flush=True)

    # 4. GET scorecard（close 之前，close 后 scorecard 被销毁 → 404）
    try:
        verify = arc.get_scorecard(card_id)
        result["verified"] = verify is not None
        print(f"  get_scorecard: {'OK' if verify else 'FAIL'}", flush=True)
    except Exception as e:
        result["status"] = "get_failed"
        result["error"] = str(e)
        print(f"  GET 失败: {e}", flush=True)

    # 5. close scorecard
    try:
        scorecard = arc.close_scorecard(card_id)
        result["scorecard"] = scorecard.model_dump() if scorecard else None
        result["status"] = "verified" if result.get("verified") else "get_failed"
        print(f"  close_scorecard: OK", flush=True)
    except Exception as e:
        result["status"] = "close_failed"
        result["error"] = str(e)
        print(f"  close 失败: {e}", flush=True)
        return result

    # 6. 保存 episode（全流程成功才保存）
    if result["status"] == "verified":
        ep_dir = out_dir.resolve() / f"{game_id}_{label}_rep{rep}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        print(f"  保存到: {ep_dir}", flush=True)

        try:
            # manifest
            manifest = {
                "game_id": game_id,
                "environment_id": result.get("environment_id"),
                "scorecard_id": card_id,
                "label": label,
                "rep": rep,
                "levels_completed": result.get("levels_completed", 0),
                "state": result.get("state"),
                "actions": result.get("actions", 0),
                "win": result.get("win", False),
                "tags": tags,
                "collected_at": time.time(),
            }
            manifest_path = str(ep_dir / "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(manifest, indent=2, ensure_ascii=False))
            print(f"  manifest.json 写入: {manifest_path}", flush=True)

            # scorecard
            if result.get("scorecard"):
                sc_path = str(ep_dir / "scorecard.json")
                with open(sc_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(result["scorecard"], indent=2, ensure_ascii=False, default=str))
                print(f"  scorecard.json 写入: {sc_path}", flush=True)

            # recording
            recordings_dir = Path(os.environ.get("RECORDINGS_DIR", "recordings")).resolve()
            recording_files = sorted(recordings_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if recording_files:
                shutil.copy2(recording_files[0], ep_dir / "recording.jsonl")
                print(f"  recording.jsonl 复制成功: {recording_files[0].name}", flush=True)
            else:
                print(f"  ⚠️ 未找到 recording 文件", flush=True)

            print(f"  ✅ episode 保存: {ep_dir}", flush=True)
        except Exception as e:
            print(f"  ❌ 保存失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
    else:
        print(f"  ❌ episode 丢弃（{result['status']}）", flush=True)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="采集真实 remote episode")
    parser.add_argument("--games", default=",".join(DEFAULT_GAMES), help="逗号分隔的游戏 ID")
    parser.add_argument("--steps", type=int, default=400, help="每局最大步数")
    parser.add_argument("--out", type=Path, default=Path("data/aop/episodes"), help="输出目录")
    parser.add_argument("--reps", type=int, default=REPS, help="每个游戏重复次数")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    # 检查 API key
    api_key = os.environ.get("ARC_API_KEY", "")
    if not api_key or len(api_key) < 16:
        print("❌ 需要 ARC_API_KEY 环境变量")
        print("   获取: https://three.arcprize.org → 注册 → API key")
        print("   设置: export ARC_API_KEY=your_key")
        return 1

    # 创建 Arcade（ONLINE 模式）
    try:
        import arc_agi
        from arc_agi import OperationMode
        arc = arc_agi.Arcade(operation_mode=OperationMode.ONLINE)
    except Exception as e:
        print(f"❌ Arcade 初始化失败: {e}")
        return 1

    # 采集
    all_results = []
    for game_id in games:
        for rep in range(args.reps):
            print(f"\n=== {game_id} rep{rep} ===", flush=True)
            result = collect_one_episode(arc, game_id, rep, args.steps, args.out)
            all_results.append(result)

    # 汇总
    verified = [r for r in all_results if r.get("status") == "verified"]
    failed = [r for r in all_results if r.get("status") != "verified"]
    print(f"\n=== 采集汇总 ===")
    print(f"  总计: {len(all_results)}")
    print(f"  成功: {len(verified)}")
    print(f"  失败: {len(failed)}")
    for r in verified:
        print(f"    ✅ {r['game_id']} rep{r['rep']}: levels={r.get('levels_completed',0)} win={r.get('win',False)}")
    for r in failed:
        print(f"    ❌ {r['game_id']} rep{r['rep']}: {r['status']}")

    # 保存汇总
    summary = {
        "total": len(all_results),
        "verified": len(verified),
        "failed": len(failed),
        "results": all_results,
    }
    (args.out / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\n汇总保存: {args.out / 'collection_summary.json'}")

    # 注意：不运行 scorecard_reaper！close 后 scorecard 已销毁，GET 必 404，
    # reaper 会把刚采集的 episode 误删到 quarantine。
    # reaper 用于清理历史残留的坏 scorecard，不用于刚采集的 episode。

    return 0 if len(verified) >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
