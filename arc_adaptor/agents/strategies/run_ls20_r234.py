"""LS20 L1: env.step + reset/重放 BFS (可靠但慢)。"""
import sys, os, time, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))  # arc_adaptor
from paths import add_to_sys_path, environments_dir  # noqa: E402

add_to_sys_path()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput, GameState

arcade = Arcade(environments_dir=environments_dir("ls20"), operation_mode=OperationMode.OFFLINE)
gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ls20")][0]
env = arcade.make(gid)

def state_key(g):
    return (int(g.gudziatsk.x), int(g.gudziatsk.y), g._step_counter_ui.current_steps)

def is_won(g, li0):
    return int(g._current_level_index) > li0 or g._state == GameState.WIN

def replay(env, actions):
    env.reset()
    g = env._game
    for a in actions:
        g.perform_action(ActionInput(id=GameAction["ACTION%d"%a], data={}, reasoning=None), raw=True)
    return g

def bfs_solve(env, t_limit=300, max_depth=20):
    env.reset()
    g = env._game
    li0 = int(g._current_level_index)
    if is_won(g, li0):
        return [], 1
    start = state_key(g)
    queue = collections.deque([[]])
    seen = {start}
    t0 = time.time()
    while queue:
        if time.time() - t0 > t_limit:
            return None, len(seen)
        path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for a in [1, 2, 3, 4]:
            g = replay(env, path + [a])
            if is_won(g, li0):
                return path + [a], len(seen)
            if g._state == GameState.GAME_OVER:
                continue
            k = state_key(g)
            if k not in seen:
                seen.add(k)
                if len(seen) % 500 == 0:
                    print(f"    状态={len(seen)} 深度={len(path)+1} 时间={time.time()-t0:.0f}s")
                queue.append(path + [a])
    return None, len(seen)

# 跑 L1
env.reset()
g = env._game
p = g.gudziatsk
print(f"L1: 玩家=({p.x},{p.y}) 步数={g._step_counter_ui.current_steps}")
t0 = time.time()
path, n_seen = bfs_solve(env, t_limit=300, max_depth=15)
elapsed = time.time() - t0

if path:
    g = replay(env, path)
    won = is_won(g, 0)
    print(f"  结果: {len(path)}步 {elapsed:.1f}s 状态={n_seen} {'OK' if won else 'FAIL'}")
    print(f"  路径: {path}")
else:
    print(f"  结果: 失败 {elapsed:.1f}s 状态={n_seen}")
