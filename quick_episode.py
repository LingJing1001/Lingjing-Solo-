# -*- coding: utf-8 -*-
"""快速生成 episode - 只运行第1关"""
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
from arcengine import GameAction
import lingjing_solo.planning.arc_shadow_cn04 as shadow


def main():
    # 创建环境，启用记录保存
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        recordings_dir=str(ROOT / "recordings"),
    )
    
    # cn04 第1关
    env = arc.make("cn04", save_recording=True, include_frame_data=True)
    g = env._game
    
    # 使用 CSP 求解
    shadow._ensure()
    solver_game = shadow._game
    solver_game._current_level_index = 0
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
        g._current_level_index = 0
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
        ok = state == 'WIN' or levels > 0
        
        print(f'L1: {len(actions)} actions -> {"OK" if ok else "FAIL"}')
        print(f'State: {state}, Levels: {levels}')
        
        # 保存 episode
        episodes_dir = ROOT / "data" / "aop" / "episodes"
        episode_dir = episodes_dir / "cn04_success_rep0"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找记录文件
        recordings_dir = ROOT / "recordings"
        if recordings_dir.exists():
            for scorecard_dir in recordings_dir.iterdir():
                if scorecard_dir.is_dir():
                    for recording_file in scorecard_dir.glob("cn04-*.jsonl"):
                        shutil.copy2(recording_file, episode_dir / "recording.jsonl")
                        print(f"recording.jsonl: {recording_file.name}")
                        break
                    break
        
        # 创建 manifest.json
        manifest = {
            "game_id": "cn04",
            "environment_id": "cn04",
            "scorecard_id": f"auto-{int(time.time())}",
            "label": "success" if ok else "fail",
            "rep": 0,
            "levels_completed": 1 if ok else 0,
            "state": "WIN" if ok else "NOT_FINISHED",
            "actions": len(actions),
            "win": ok,
            "tags": ["cn04", "success" if ok else "fail", "rep0", "phase-a"],
            "collected_at": time.time()
        }
        
        with open(episode_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"manifest.json: 已创建")
        print(f"Episode 保存到: {episode_dir}")
        
        # 检查 recording.jsonl
        recording_path = episode_dir / "recording.jsonl"
        if recording_path.exists():
            with open(recording_path, 'r') as f:
                lines = f.readlines()
            print(f"recording.jsonl: {len(lines)} 行")
        else:
            print("recording.jsonl: 未找到")
    else:
        print("NO SOLUTION")


if __name__ == "__main__":
    main()
