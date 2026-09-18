"""探查 AR25 L1-L5 结构（搜索-执行分离）。"""
import sys, os, time, heapq
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"F:/pro/Lingjing-Solo-/arc_adaptor")
sys.path.insert(0, r"F:/pro/Lingjing-Solo-backup-20260916-125530/arc_adaptor")
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput, GameState
from arc_shadow import _snapshot, _restore, _state_key, _heuristic

ACT_MAP = {n: getattr(GameAction, "ACTION%d" % n) for n in range(1, 8)}

def is_won(g):
    w = g.vplrhaovhr()
    return w is True or (hasattr(w, '__len__') and len(w) == 1 and bool(w))

def search_level(env, t_limit=30):
    """搜索当前关，返回通关动作序列或 None。搜索后恢复状态。"""
    g = env._game
    if is_won(g): return []
    li0 = int(g._current_level_index)
    snap0 = _snapshot(g)
    try: g.next_level = lambda: None
    except: pass
    try:
        seq=0; heap=[(int(_heuristic(g)),0,0,seq,snap0,[])]; seen={_state_key(g)}; t0=time.time()
        while heap:
            if time.time()-t0 > t_limit: return None
            h,_,_,_,snap,path = heapq.heappop(heap)
            mx = int(g.lelsvjlwneo.ilqnjlrnkk)-1
            if len(path) >= mx: continue
            for a in [1,2,3,4,5]:
                _restore(g, snap)
                try: g.perform_action(ActionInput(id=ACT_MAP[a],data={},reasoning=None),raw=True)
                except: continue
                if int(g._current_level_index) > li0 or is_won(g): return path+[a]
                if g._state == GameState.GAME_OVER: continue
                k = _state_key(g)
                if k in seen: continue
                seen.add(k); seq+=1
                heapq.heappush(heap,(int(_heuristic(g))*10+len(path)+1,len(path)+1,0,seq,_snapshot(g),path+[a]))
        return None
    finally:
        try: del g.next_level
        except: pass
        _restore(g, snap0)

arcade = Arcade(environments_dir=r"F:/pro/Lingjing-Solo-/environment_files", operation_mode=OperationMode.OFFLINE)
gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ar25")][0]
env = arcade.make(gid); env.reset(); g = env._game

for level in range(6):
    if int(g._current_level_index) < level: break
    axes = [(int(ax.x),int(ax.y),[t for t in (ax.tags or []) if 'kgxr' in t or 'nugu' in t]) for ax in g.jtkyjqznbnp]
    n_switch = len(g.ayyvxqrhnzw)
    axis_ids = set(id(a) for a in g.jtkyjqznbnp)
    spr_info = [(int(o.x),int(o.y),"轴" if id(o) in axis_ids else "块") for o in g.ayyvxqrhnzw]
    targets = [(int(t.x),int(t.y)) for t in g.fswikrcrdmx]
    budget = int(g.lelsvjlwneo.ilqnjlrnkk)
    print(f"L{level+1}: budget={budget} n_switch={n_switch} 目标={len(targets)} 轴={axes} 可切换={spr_info}")
    print(f"  目标: {targets}")
    if level < 5:
        path = search_level(env, t_limit=60)
        if path is None: print(f"  L{level+1}搜索超时"); break
        for a in path: g.perform_action(ActionInput(id=ACT_MAP[a],data={},reasoning=None),raw=True)
        print(f"  → L{level+1}通关 {len(path)}步, 进L{int(g._current_level_index)+1}")
