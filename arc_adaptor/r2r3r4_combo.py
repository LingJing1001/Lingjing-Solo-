"""R2+R3+R4 组合智能播放引擎

组合流程:
  R2 (感知): 帧画面 → 块/目标/状态哈希/转移表/胜利检测
  R4 (剪枝): 信息增益 + 目标推断 + 反循环 + 探测预算 → 选择最优动作
  R3 (寻路): BFS/A* 状态转移图搜索 → 规划完整路径

用于 AR25 随机播放的智能决策替代纯随机选择。
"""
import sys
import os
import json
import hashlib
from collections import defaultdict, deque
import heapq

if __name__ == '__main__' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads'))


# ═══════════════════════════════════════════════════════════
# R2: 感知模块
# ═══════════════════════════════════════════════════════════

def r2_state_hash(grid):
    s = json.dumps(grid)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def r2_find_cells(grid, color):
    return [(x, y) for y in range(len(grid)) for x in range(len(grid[0])) if grid[y][x] == color]


def r2_find_connected_regions(cells):
    if not cells:
        return []
    cell_set = set(cells)
    visited = set()
    regions = []
    for cell in cells:
        if cell not in visited:
            queue = deque([cell])
            region = []
            while queue:
                x, y = queue.popleft()
                if (x, y) not in visited:
                    visited.add((x, y))
                    region.append((x, y))
                    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in cell_set and (nx, ny) not in visited:
                            queue.append((nx, ny))
            regions.append(region)
    return regions


def r2_detect_victory(grid, game_state=None):
    # 颜色邻近关系不能证明通关；以环境终态为准。
    return getattr(game_state, 'name', game_state) == 'WIN'


def r2_perceive(grid, game_state=None):
    sh = r2_state_hash(grid)
    players = r2_find_cells(grid, 5)
    targets = r2_find_cells(grid, 11)
    axes = r2_find_cells(grid, 10)
    victory = r2_detect_victory(grid, game_state)
    player_regions = r2_find_connected_regions(players)
    player_pos = None
    if player_regions:
        main = max(player_regions, key=len)
        player_pos = (
            int(sum(r[0] for r in main) / len(main)),
            int(sum(r[1] for r in main) / len(main)),
        )
    target_center = None
    if targets:
        target_center = (
            int(sum(t[0] for t in targets) / len(targets)),
            int(sum(t[1] for t in targets) / len(targets)),
        )
    return {
        'state_hash': sh,
        'player_pos': player_pos,
        'target_center': target_center,
        'player_count': len(players),
        'target_count': len(targets),
        'axis_count': len(axes),
        'victory': victory,
    }


# ═══════════════════════════════════════════════════════════
# R4: 剪枝模块
# ═══════════════════════════════════════════════════════════

TRANSFER_STORE = 'data/r2_transfers.json'


def _load_transfers():
    if os.path.exists(TRANSFER_STORE):
        try:
            with open(TRANSFER_STORE) as f:
                return json.load(f)
        except Exception:
            pass
    return {'transfers': []}


def _save_transfers(data):
    """落盘转移表。

    这里绝不能抛异常: 它在每次智能播放决策时都会调用, 一旦写失败(磁盘满/
    权限/文件被占用), 异常会穿透 smart_play_step 让接口 500, 前端只会静默
    退化成随机动作 —— 反而更难排查。失败只记日志, 决策照常返回。
    """
    try:
        os.makedirs(os.path.dirname(TRANSFER_STORE) or '.', exist_ok=True)
        tmp = TRANSFER_STORE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, TRANSFER_STORE)  # 原子替换, 避免写一半被读到
    except Exception as e:
        print(f"[r2r3r4] 转移表落盘失败(不影响决策): {e!r}")


def r4_info_gain(action, transfers):
    usage_count = sum(1 for t in transfers.get('transfers', []) if t.get('action') == action)
    return 1.0 / (1 + usage_count)


def r4_loop_penalty(action, recent_actions, window=3):
    if len(recent_actions) < window:
        return 0.0
    recent = recent_actions[-window:]
    return recent.count(action) * 0.3


def r4_infer_goal(player_pos, target_center):
    if not target_center or not player_pos:
        return 'unknown'
    dx = target_center[0] - player_pos[0]
    dy = target_center[1] - player_pos[1]
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'


def r4_score_action(action, player_pos, target_center, transfers, recent_actions):
    info = r4_info_gain(action, transfers)
    loop = r4_loop_penalty(action, recent_actions)

    dx, dy = 0, 0
    if action == 1:
        dy = -1
    elif action == 2:
        dy = 1
    elif action == 3:
        dx = -1
    elif action == 4:
        dx = 1

    goal_proximity = 0.0
    if player_pos and target_center:
        new_pos = (player_pos[0] + dx, player_pos[1] + dy)
        old_dist = abs(player_pos[0] - target_center[0]) + abs(player_pos[1] - target_center[1])
        new_dist = abs(new_pos[0] - target_center[0]) + abs(new_pos[1] - target_center[1])
        goal_proximity = (old_dist - new_dist) * 0.5

    return info - loop + goal_proximity


def r4_choose_action(grid, player_pos, target_center, recent_actions=None):
    if recent_actions is None:
        recent_actions = []
    transfers = _load_transfers()
    candidates = []
    for action in [1, 2, 3, 4, 5]:
        score = r4_score_action(action, player_pos, target_center, transfers, recent_actions)
        candidates.append((action, score))
    candidates.sort(key=lambda x: -x[1])
    best_action = candidates[0][0]
    return best_action, candidates


# ═══════════════════════════════════════════════════════════
# R3: 寻路模块
# ═══════════════════════════════════════════════════════════

def r3_bfs(grid, start, goal, passable):
    h, w = len(grid), len(grid[0])
    visited = {start: None}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            path = []
            current = (x, y)
            while visited[current] is not None:
                prev = visited[current]
                dx, dy = current[0] - prev[0], current[1] - prev[1]
                if dy == -1:
                    path.append(1)
                elif dy == 1:
                    path.append(2)
                elif dx == -1:
                    path.append(3)
                elif dx == 1:
                    path.append(4)
                current = prev
            return list(reversed(path))
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in visited:
                if grid[ny][nx] in passable:
                    visited[(nx, ny)] = (x, y)
                    queue.append((nx, ny))
    return None


def r3_astar(grid, start, goal, passable):
    h, w = len(grid), len(grid[0])

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(0, heuristic(start, goal), start)]
    came_from = {start: None}
    g_score = {start: 0}
    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while came_from[current] is not None:
                prev = came_from[current]
                dx, dy = current[0] - prev[0], current[1] - prev[1]
                if dy == -1:
                    path.append(1)
                elif dy == 1:
                    path.append(2)
                elif dx == -1:
                    path.append(3)
                elif dx == 1:
                    path.append(4)
                current = prev
            return list(reversed(path))
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] in passable:
                tentative_g = g_score[current] + 1.0
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f, heuristic(neighbor, goal), neighbor))
    return None


def r3_plan_path(grid, start, goal, use_astar=True):
    passable = {0, 5, 10, 11}
    if use_astar:
        return r3_astar(grid, start, goal, passable)
    else:
        return r3_bfs(grid, start, goal, passable)


# ═══════════════════════════════════════════════════════════
# 组合入口: R2+R3+R4
# ═══════════════════════════════════════════════════════════

_recent_actions = []
_path_cache = []
_path_index = 0


def vc33_click_candidates(grid):
    """VC33-5430563c 的紫色按钮和已激活交换器，坐标为显示像素。"""
    candidates = []
    for color in (9, 12):
        regions = r2_find_connected_regions(r2_find_cells(grid, color))
        for region in regions:
            cx = sum(x for x, y in region) / len(region)
            cy = sum(y for x, y in region) / len(region)
            x, y = min(region, key=lambda p: ((p[0]-cx)**2 + (p[1]-cy)**2, p))
            candidates.append({'action': 6, 'data': {'x': x, 'y': y}})
    return candidates


def r3_search_click_path(reset, step, candidates, state_key, max_seconds=30,
                         max_depth=40, max_states=10000):
    """通过公开 reset/step 重放 BFS；不跳关，不读取引擎私有状态。"""
    import time

    started = time.monotonic()
    initial = reset()
    initial_level = initial.levels_completed
    queue = deque([[]])
    seen = {state_key(initial)}
    calls = 0

    def result(reason, path=None):
        return {'reason': reason, 'path': path, 'states': len(seen),
                'search_actions': calls, 'seconds': round(time.monotonic()-started, 3)}

    def expired():
        return time.monotonic()-started >= max_seconds

    if getattr(initial.state, 'name', initial.state) in ('WIN', 'GAME_OVER'):
        return result('terminal_initial_state')
    depth_limited = False
    while queue:
        if expired():
            return result('time_limit')
        path = queue.popleft()
        frame = reset()
        for action in path:
            if expired():
                return result('time_limit')
            frame = step(action)
            calls += 1
        if len(path) >= max_depth:
            depth_limited = True
            continue
        for action in candidates(frame):
            if expired():
                return result('time_limit')
            branch = reset()
            for previous in path:
                if expired():
                    return result('time_limit')
                branch = step(previous)
                calls += 1
            branch = step(action)
            calls += 1
            status = getattr(branch.state, 'name', branch.state)
            if status == 'WIN' or branch.levels_completed > initial_level:
                return result('solved', path + [action])
            if status == 'GAME_OVER':
                continue
            key = state_key(branch)
            # BFS 最先发现的相同画面路径更短，保留更多动作预算。
            if key in seen:
                continue
            if len(seen) >= max_states:
                return result('state_limit')
            seen.add(key)
            queue.append(path + [action])
    return result('depth_limit' if depth_limited else 'frontier_exhausted')


_click_visits = defaultdict(int)


def smart_play_step(grid, recent_actions=None, *, game_state=None,
                    available_actions=None, game_id=None):
    """R2+R3+R4 组合决策：返回下一步动作

    策略:
      1. R2 感知当前帧
      2. 如果已胜利 → 返回 None
      3. 如果有缓存路径 → 沿路径走
      4. R3 尝试 A* 寻路到目标
      5. 寻路成功 → 缓存路径，走第一步
      6. 寻路失败 → R4 剪枝选最优动作
      7. 记录转移并返回动作
    """
    global _path_cache, _path_index

    if recent_actions is not None:
        global _recent_actions
        _recent_actions = recent_actions

    # 旧调用返回方向动作整数；VC33 调用返回含坐标的动作请求。
    if game_id and game_id.startswith('vc33'):
        if getattr(game_state, 'name', game_state) in ('WIN', 'GAME_OVER'):
            return None
        allowed = {getattr(a, 'value', a) for a in (available_actions or [])}
        if 6 not in allowed:
            raise ValueError('VC33 requires ACTION6 in available_actions')
        candidates = vc33_click_candidates(grid)
        if not candidates:
            raise ValueError('No visible VC33 click controls')
        state = r2_state_hash(grid[1:])
        def usage(action):
            return _click_visits[(state, action['data']['x'], action['data']['y'])]
        action = min(candidates, key=usage)
        _click_visits[(state, action['data']['x'], action['data']['y'])] += 1
        return action

    # R2: 感知
    perception = r2_perceive(grid, game_state)

    if perception['victory'] or getattr(game_state, 'name', game_state) == 'GAME_OVER':
        _path_cache = []
        _path_index = 0
        return None

    player_pos = perception['player_pos']
    target_center = perception['target_center']

    if not player_pos:
        return 5

    # 沿缓存路径走
    if _path_cache and _path_index < len(_path_cache):
        action = _path_cache[_path_index]
        _path_index += 1
        _recent_actions.append(action)
        _record_transfer(r2_state_hash(grid), action, 'path')
        return action

    # R3: 寻路
    if target_center:
        path = r3_plan_path(grid, player_pos, target_center, use_astar=True)
        if path and len(path) > 0:
            _path_cache = path
            _path_index = 0
            action = _path_cache[_path_index]
            _path_index += 1
            _recent_actions.append(action)
            _record_transfer(r2_state_hash(grid), action, 'astar')
            return action

    # R4: 剪枝选动作
    action, candidates = r4_choose_action(
        grid, player_pos, target_center, _recent_actions
    )
    _recent_actions.append(action)
    _record_transfer(r2_state_hash(grid), action, 'prune')
    return action


def _record_transfer(state_hash, action, method):
    data = _load_transfers()
    data.setdefault('transfers', []).append({
        'state_hash': state_hash,
        'action': action,
        'method': method,
    })
    if len(data['transfers']) > 5000:
        data['transfers'] = data['transfers'][-3000:]
    _save_transfers(data)


def reset_smart_play():
    global _recent_actions, _path_cache, _path_index
    _click_visits.clear()
    _recent_actions = []
    _path_cache = []
    _path_index = 0


if __name__ == '__main__':
    print("=" * 60)
    print("R2+R3+R4 组合智能播放引擎")
    print("=" * 60)

    import arc_shadow_ar25
    arc_shadow_ar25.reset()

    frame = arc_shadow_ar25.get_frame()
    print(f"\n初始帧: {len(frame)}×{len(frame[0])}")

    perception = r2_perceive(frame)
    print(f"R2 感知: 玩家={perception['player_pos']}, 目标={perception['target_center']}, 胜利={perception['victory']}")

    action, candidates = r4_choose_action(
        frame, perception['player_pos'], perception['target_center'], []
    )
    action_names = {1: 'UP ↑', 2: 'DOWN ↓', 3: 'LEFT ←', 4: 'RIGHT →', 5: 'TOGGLE ⟳'}
    print(f"\nR4 剪枝结果:")
    for a, s in candidates:
        print(f"  {action_names.get(a, str(a))}: 评分={s:.4f}")
    print(f"  → 选择: {action_names.get(action, str(action))}")

    if perception['player_pos'] and perception['target_center']:
        path = r3_plan_path(frame, perception['player_pos'], perception['target_center'])
        print(f"\nR3 寻路: {'找到' if path else '未找到'} {len(path) if path else 0}步路径")
        if path:
            print(f"  动作序列: {[action_names.get(a, str(a)) for a in path[:10]]}{'...' if len(path) > 10 else ''}")

    print(f"\n组合决策 (10步模拟):")
    reset_smart_play()
    for i in range(10):
        action = smart_play_step(frame)
        if action is None:
            print(f"  步骤{i+1}: 🎉 胜利!")
            break
        arc_shadow_ar25.apply(action)
        frame = arc_shadow_ar25.get_frame()
        p = r2_perceive(frame)
        print(f"  步骤{i+1}: {action_names.get(action, str(action))} → 玩家={p['player_pos']}, 胜利={p['victory']}")

    print("\n" + "=" * 60)
    print("✅ R2+R3+R4 组合引擎就绪!")
    print("=" * 60)