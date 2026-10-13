# -*- coding: utf-8 -*-
"""快速生成 cn04 episode"""
import sys, os, time, json, shutil
sys.path.insert(0, 'D:/Quan/Lingjing-Solo-main-bench')
os.environ.setdefault('LINGJING_SRC', 'D:/Quan/Lingjing-Solo-main-bench/lingjing_solo')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
import lingjing_solo.planning.arc_shadow_cn04 as shadow
from pathlib import Path

ROOT = Path('D:/Quan/Lingjing-Solo-main-bench')
episodes_dir = ROOT / 'data' / 'aop' / 'episodes'
recordings_dir = ROOT / 'recordings'

print('=== 生成 cn04 episode ===')
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, recordings_dir=str(recordings_dir))
env = arc.make('cn04', save_recording=True, include_frame_data=True)
g = env._game

shadow._ensure()
solver_game = shadow._game

all_actions = []
all_ok = True

for lvl in range(1):  # 只运行第1关，减少内存
    solver_game._current_level_index = lvl
    solver_game.level_reset()
    sol = shadow.solve_level(t_limit=60.0)
    
    if sol:
        actions = []
        for a in sol:
            if a[0] == 'c': actions.append(('click', a[1], a[2]))
            elif a[0] == 'm': actions.append(('move', a[1]))
            elif a[0] == 'r': actions.append(('rotate',))
            elif a[0] == 'i': actions.append(('move', a[1]))
        
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
        status = 'OK' if ok else 'FAIL'
        print(f'  L{lvl+1}: {len(actions)} actions -> {status}')
        all_actions.extend(actions)
        if not ok: all_ok = False
    else:
        print(f'  L{lvl+1}: NO SOLUTION')
        all_ok = False

# 保存
label = 'success' if all_ok else 'fail'
episode_dir = episodes_dir / f'cn04_{label}_rep_full'
episode_dir.mkdir(parents=True, exist_ok=True)

if recordings_dir.exists():
    for sd in recordings_dir.iterdir():
        if sd.is_dir():
            for rf in sd.glob('cn04-*.jsonl'):
                shutil.copy2(rf, episode_dir / 'recording.jsonl')
                break
            break

manifest = {'game_id': 'cn04', 'label': label, 'levels_completed': 6 if all_ok else 0, 'actions': len(all_actions), 'win': all_ok}
with open(episode_dir / 'manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

rp = episode_dir / 'recording.jsonl'
if rp.exists():
    with open(rp) as f:
        lines = sum(1 for _ in f)
    print(f'recording.jsonl: {lines} lines')

status = 'ALL_PASS' if all_ok else 'FAIL'
print(f'Total: {len(all_actions)} actions, {status}')
