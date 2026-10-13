# -*- coding: utf-8 -*-
"""自动生成 episodes 数据 - 从官网API获取游戏记录

使用 save_recording=True 保存游戏记录到 episodes 目录
"""
import sys
import os
import time
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput


def generate_episodes(game_id: str, num_reps: int = 2, max_levels: int = 3):
    """为指定游戏生成 episodes 数据"""
    print(f"\n{'='*60}")
    print(f"生成 episodes: {game_id} (reps={num_reps}, levels={max_levels})")
    print(f"{'='*60}")
    
    episodes_dir = ROOT / "data" / "aop" / "episodes"
    recordings_dir = ROOT / "recordings"
    
    success_count = 0
    fail_count = 0
    
    for rep in range(num_reps):
        print(f"\n--- Rep {rep} ---")
        
        # 创建环境，启用记录保存
        arc = arc_agi.Arcade(
            operation_mode=OperationMode.NORMAL,
            recordings_dir=str(recordings_dir),
        )
        
        env = arc.make(
            game_id,
            save_recording=True,
            include_frame_data=True,
        )
        
        if env is None:
            print(f"无法创建 {game_id} 环境")
            continue
        
        g = env._game
        num_levels = len(g._levels)
        levels_to_play = min(max_levels, num_levels)
        
        print(f"总关卡数: {num_levels}, 目标: {levels_to_play} 关")
        
        # 使用 CSP 求解器（如果是 cn04）
        if game_id == "cn04":
            import lingjing_solo.planning.arc_shadow_cn04 as shadow
            shadow._ensure()
            solver_game = shadow._game
        else:
            solver_game = g
        
        all_actions = []
        all_ok = True
        
        for lvl in range(levels_to_play):
            print(f"\n  L{lvl+1}:", end=" ", flush=True)
            
            # 设置关卡
            if game_id == "cn04":
                solver_game._current_level_index = lvl
                solver_game.level_reset()
                sol = shadow.solve_level(t_limit=60.0)
                
                if sol:
                    # 转换动作格式
                    actions = []
                    for a in sol:
                        if a[0] == 'c':
                            actions.append(('click', a[1], a[2]))
                        elif a[0] == 'm':
                            actions.append(('move', a[1]))
                        elif a[0] == 'r':
                            actions.append(('rotate',))
                        elif a[0] == 'i':
                            actions.append(('move', a[1]))
                    
                    # 在真实环境上执行
                    g._current_level_index = lvl
                    g.level_reset()
                    env.reset()
                    
                    for action in actions:
                        if action[0] == 'click':
                            frame = env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
                        elif action[0] == 'move':
                            frame = env.step(GameAction.from_id(action[1]), data={})
                        elif action[0] == 'rotate':
                            frame = env.step(GameAction.ACTION5, data={})
                    
                    state = frame.state.name if hasattr(frame, 'state') else ''
                    levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
                    ok = state == 'WIN' or levels > lvl
                    
                    print(f"{len(actions)} actions -> {'OK' if ok else 'FAIL'}", flush=True)
                    all_actions.extend(actions)
                    
                    if not ok:
                        all_ok = False
                        break
                else:
                    print("NO SOLUTION", flush=True)
                    all_ok = False
                    break
            else:
                # 其他游戏使用 A* 搜索
                from sk48_astar import solve_level_astar, is_win, apply_action
                
                g._current_level_index = lvl
                g.level_reset()
                env.reset()
                
                sol = solve_level_astar(g, lvl, max_time=120, max_nodes=50000)
                
                if sol:
                    for action in sol:
                        if action[0] == 'click':
                            frame = env.step(GameAction.ACTION6, data={'x': action[1], 'y': action[2]})
                        elif action[0] == 'move':
                            frame = env.step(GameAction.from_id(action[1]), data={})
                        elif action[0] == 'rotate':
                            frame = env.step(GameAction.ACTION5, data={})
                    
                    ok = is_win(g, lvl)
                    print(f"{len(sol)} actions -> {'OK' if ok else 'FAIL'}", flush=True)
                    all_actions.extend(sol)
                    
                    if not ok:
                        all_ok = False
                        break
                else:
                    print("NO SOLUTION", flush=True)
                    all_ok = False
                    break
        
        # 保存 episode 数据
        if all_ok:
            label = "success"
            success_count += 1
        else:
            label = "fail"
            fail_count += 1
        
        episode_name = f"{game_id}_{label}_rep{rep}"
        episode_dir = episodes_dir / episode_name
        
        print(f"\n  保存 episode: {episode_name}")
        
        # 创建目录
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找记录文件
        if recordings_dir.exists():
            for scorecard_dir in recordings_dir.iterdir():
                if scorecard_dir.is_dir():
                    for recording_file in scorecard_dir.glob(f"{game_id}-*.jsonl"):
                        # 复制 recording.jsonl
                        shutil.copy2(recording_file, episode_dir / "recording.jsonl")
                        print(f"  recording.jsonl: {recording_file.name}")
                        break
                    break
        
        # 创建 manifest.json
        manifest = {
            "game_id": game_id,
            "environment_id": game_id,
            "scorecard_id": f"auto-{rep}-{int(time.time())}",
            "label": label,
            "rep": rep,
            "levels_completed": len(all_actions),
            "state": "WIN" if all_ok else "NOT_FINISHED",
            "actions": len(all_actions),
            "win": all_ok,
            "tags": [game_id, label, f"rep{rep}", "phase-a"],
            "collected_at": time.time()
        }
        
        with open(episode_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # 创建 scorecard.json
        scorecard = {
            "source_url": None,
            "tags": [game_id, label, f"rep{rep}", "phase-a", "agent"],
            "opaque": None,
            "card_id": manifest["scorecard_id"],
            "api_key": None,
            "score": 100.0 if all_ok else 0.0,
            "environments": [{
                "id": f"{game_id}-auto",
                "runs": [{
                    "id": None,
                    "guid": f"auto-{rep}",
                    "score": 100.0 if all_ok else 0.0,
                    "levels_completed": levels_to_play if all_ok else 0,
                    "actions": len(all_actions),
                    "resets": 0,
                    "state": "WIN" if all_ok else "NOT_FINISHED",
                    "completed": all_ok,
                }],
                "score": 100.0 if all_ok else 0.0,
                "actions": len(all_actions),
                "levels_completed": levels_to_play if all_ok else 0,
                "completed": all_ok,
                "level_count": num_levels,
                "resets": 0
            }],
            "competition_mode": False,
            "total_environments_completed": 1 if all_ok else 0,
            "total_environments": 1,
            "total_levels_completed": levels_to_play if all_ok else 0,
            "total_levels": num_levels,
            "total_actions": len(all_actions)
        }
        
        with open(episode_dir / "scorecard.json", 'w') as f:
            json.dump(scorecard, f, indent=2)
        
        print(f"  manifest.json: 已创建")
        print(f"  scorecard.json: 已创建")
    
    print(f"\n{'='*60}")
    print(f"生成完成: {game_id}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"{'='*60}")
    
    return success_count, fail_count


def main():
    """批量生成 episodes 数据"""
    print("=" * 60)
    print("批量生成 episodes 数据")
    print("=" * 60)
    
    # 要生成 episodes 的游戏列表
    games = [
        ("cn04", 2, 6),   # cn04: 2次重复, 6关
        ("sk48", 2, 1),   # sk48: 2次重复, 1关（只通过第1关）
    ]
    
    total_success = 0
    total_fail = 0
    
    for game_id, reps, levels in games:
        success, fail = generate_episodes(game_id, reps, levels)
        total_success += success
        total_fail += fail
    
    print(f"\n{'='*60}")
    print(f"全部完成!")
    print(f"  总成功: {total_success}")
    print(f"  总失败: {total_fail}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
