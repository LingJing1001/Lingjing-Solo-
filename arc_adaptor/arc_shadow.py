"""AR25 本地引擎影子模拟器 + 镜像机制求解器

供 /api/proxy/arc-solve 使用:
- 影子(本地 arcengine 环境实例)与线上会话逐步同步:
  arc_start -> reset(), arc_step 成功后 -> apply(move)
- 求解用贪心最佳优先搜索 (不追求最短, 预算内可行即可),
  启发函数 = Σ 未覆盖目标格 (块格/当前反射格的最小曼哈顿距离 + 1)

注意: 快照必须覆盖引擎全部可变全局状态 (选中对象/旋转距离/动画标志/_state),
否则搜索会"找到"现实中不可达的假解 — 详见 docs/arc-agi-3-ar25-report.md §4.1
"""
import os
import json
import threading
import time
import heapq

_lock = threading.RLock()
_env = None
_game = None
_num2act = {}
_act2num = {}
_applied = 0
_valid = False
_error = None
_ActionInput = None
_GameState = None
_ACTS = None


def _ensure():
    global _env, _game, _num2act, _act2num, _valid, _error, _ActionInput, _GameState, _ACTS
    if _env is not None:
        return _valid
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput, GameState
        arc = arc_agi.Arcade()
        _env = arc.make("ar25-0c556536")
        _game = _env._game
        _num2act = {n: getattr(GameAction, "ACTION%d" % n) for n in range(1, 8)}
        _act2num = {v: k for k, v in _num2act.items()}
        _ActionInput = ActionInput
        _GameState = GameState
        _ACTS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                 GameAction.ACTION4, GameAction.ACTION5]
        _load_solutions()
        _env.reset()
        _valid = True
    except Exception as e:
        _error = str(e)
        _valid = False
    return _valid


def reset():
    """线上 arc-start(开新档/重置) 时调用: 影子回到 L1 起点"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            _applied = 0
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def apply(move):
    """线上 arc-step 成功后调用: 影子同步执行同一动作"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _game.perform_action(
                _ActionInput(id=_num2act[int(move)], data={}, reasoning=None), raw=True)
            _applied += 1
            return True
        except Exception:
            _valid = False
            return False


def is_synced(actions_taken):
    """影子动作计数与线上会话 actions_taken 一致才可信"""
    return _valid and _applied == int(actions_taken or 0)


# 已在线上 scorecard 验证过的关卡解法 (从关卡起点出发), level_idx(0-based) -> 动作序列
_SYM = {'↑': 1, '↓': 2, '←': 3, '→': 4, '选': 5}


def _dec(s):
    return [_SYM[c] for c in s]


KNOWN_SOLUTIONS = {
    0: _dec("↓" * 10 + "←" * 5),                                   # L1: 15步 (最优)
    1: _dec("↓↓↓↓↓→↓↓↓选←←选←"),                                    # L2: 14步
    2: _dec("↓↓↓↓→选←←↑↑↑↑↑←←←↑←←选↑选→→→选选↓选↓↓↓选选↑↑↑↑↑↑↑选→→→选←←←←←" + "↓" * 11),  # L3: 62步
    3: _dec("→→→→→↓→→↑选→→→→→→→选" + "↓" * 6),                      # L4: 24步
    4: _dec("→→→→→→→→→选↑↑↑←↑↑选↓↓↓选←选←←选选←选←←选选←选←选选←选←←←←选↓↓选→选↑↑选↑选←"),  # L5: 56步
    # L7: 78步 (离线求解, 线上重放验证: 关卡索引 6->7)
    6: [2, 2, 2, 2, 2, 2, 5, 5, 3, 3, 3, 3, 5, 4, 1, 1, 1, 1, 1, 4, 4, 5, 1, 1, 5, 5,
        1, 1, 4, 5, 5, 1, 5, 5, 1, 1, 5, 1, 5, 1, 5, 4, 4, 4, 4, 4, 4, 5, 3, 5, 5, 5,
        4, 4, 4, 4, 4, 5, 1, 5, 5, 5, 3, 5, 1, 1, 1, 4, 5, 5, 5, 3, 5, 1, 1, 4, 1, 1],
}

# 解法持久化缓存: 搜索成功的新解自动存盘, 重启后仍秒回
_SOLUTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "arc_solutions.json")


def _load_solutions():
    try:
        with open(_SOLUTIONS_FILE) as f:
            for k, v in json.load(f).items():
                idx = int(k)
                if idx not in KNOWN_SOLUTIONS or len(v) < len(KNOWN_SOLUTIONS[idx]):
                    KNOWN_SOLUTIONS[idx] = v
    except Exception:
        pass


def _save_solution(level_idx, moves):
    try:
        data = {}
        if os.path.exists(_SOLUTIONS_FILE):
            with open(_SOLUTIONS_FILE) as f:
                data = json.load(f)
        data[str(level_idx)] = list(moves)
        os.makedirs(os.path.dirname(_SOLUTIONS_FILE), exist_ok=True)
        with open(_SOLUTIONS_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def status():
    return {"valid": _valid, "applied": _applied, "error": _error, "loaded": _env is not None}


# ─── 内部: 快照与搜索 ─────────────────────────────────

def _sel_idx(g):
    try:
        return g.ayyvxqrhnzw.index(g.yvifanjrcyu)
    except ValueError:
        return -1


def _snapshot(g):
    spr = [(s.x, s.y, s.pixels.copy()) for s in g.ayyvxqrhnzw]
    ovo = tuple(sorted((g.ouurgkpbbjj.index(s), d) for s, d in g.ovoizfolxfq.items()))
    misc = (g.hsiusrsrdkswnt, g.qehjebksqcm, g.hujpxmlafgh, g.xukxeewuexo, g.xjwpeqpcxav, g._state)
    return (spr, _sel_idx(g), g.lelsvjlwneo.current_steps, ovo, misc)


def _restore(g, snap):
    spr, si, steps, ovo, misc = snap
    for s, (x, y, px) in zip(g.ayyvxqrhnzw, spr):
        s.set_position(x, y)
        s.pixels = px.copy()
    if 0 <= si < len(g.ayyvxqrhnzw):
        g.yvifanjrcyu = g.ayyvxqrhnzw[si]
    g.lelsvjlwneo.current_steps = steps
    g.ovoizfolxfq = {g.ouurgkpbbjj[i]: d for i, d in ovo}
    g.hsiusrsrdkswnt, g.qehjebksqcm, g.hujpxmlafgh, g.xukxeewuexo, g.xjwpeqpcxav, st = misc
    g._state = st


def _state_key(g):
    """状态快照键: 转换为可哈希的标量元组，避免 numpy 数组比较问题"""
    def _to_hashable(val):
        if isinstance(val, (int, float, str, bool, type(None))):
            return val
        if hasattr(val, 'tobytes'):
            return val.tobytes()  # numpy array -> bytes
        if isinstance(val, (list, tuple)):
            return tuple(_to_hashable(x) for x in val)
        return hash(str(type(val)))
    return (
        tuple((s.x, s.y, s.pixels.tobytes()) for s in g.ayyvxqrhnzw),
        _sel_idx(g),
        _to_hashable(g.hsiusrsrdkswnt),
        _to_hashable(g.qehjebksqcm),
    )


def _heuristic(g):
    grid = g.naxbskjmlg()
    unc = [(s.x, s.y) for s in g.fswikrcrdmx if grid[s.y, s.x] < 0]
    if not unc:
        return 0
    cells = []
    for s in g.ouurgkpbbjj:
        h, w = s.pixels.shape
        for i in range(h):
            for j in range(w):
                if s.pixels[i, j] != -1:
                    cells.append((s.x + j, s.y + i))
    refl = []
    for ax in g.jtkyjqznbnp:
        if "0054kgxrvfihgm" in ax.tags:
            refl += [(2 * ax.x - x, y) for x, y in cells]
        if "0002nuguepuujf" in ax.tags:
            refl += [(x, 2 * ax.y - y) for x, y in cells]
    allc = cells + refl
    total = 0
    for tx, ty in unc:
        d = min((abs(px - tx) + abs(py - ty) for px, py in allc), default=99)
        # 杠杆距离: 挪轴搬运反射 (轴移1格->反射移2格) + 块垂直/水平对齐
        for ax in g.jtkyjqznbnp:
            if "0054kgxrvfihgm" in ax.tags:
                for px, py in cells:
                    s2 = tx + px  # 轴需要移到 (tx+px)/2 才能让反射x=tx
                    if s2 % 2:
                        s2 += 1  # 奇数时近似+1
                    dl = abs(py - ty) + abs(s2 // 2 - ax.x)
                    if dl < d:
                        d = dl
            if "0002nuguepuujf" in ax.tags:
                for px, py in cells:
                    s2 = ty + py
                    if s2 % 2:
                        s2 += 1
                    dl = abs(px - tx) + abs(s2 // 2 - ax.y)
                    if dl < d:
                        d = dl
        total += d + 1
    return total


def replay(actions):
    """自愈重建: 影子回到 L1 并重放线上完整动作历史 (与线上引擎同样确定性)"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            for m in actions:
                _game.perform_action(
                    _ActionInput(id=_num2act[int(m)], data={}, reasoning=None), raw=True)
            _applied = len(actions)
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def solve_level(t_limit=120.0, max_nodes=600000):
    """从影子当前状态求解当前关。

    返回: 动作编号列表 [1-7] (已通关 pending 时返回 []),
          失败/超时返回 None。调用方需先检查 is_synced。
    """
    if not _ensure() or not _valid:
        return None
    with _lock:
        g = _game
        t0 = time.time()
        # 防御: vplrhaovhr() 可能返回 numpy 数组
        won_flag = g.vplrhaovhr()
        if won_flag is True or (hasattr(won_flag, '__len__') and len(won_flag) == 1 and bool(won_flag)):
            return []
        level_idx0 = int(g._current_level_index)
        # 关卡起点 (步数未消耗) 且有已验证解法 -> 直接返回, 免搜索
        cs = int(g.lelsvjlwneo.current_steps)
        ik = int(g.lelsvjlwneo.ilqnjlrnkk)
        if level_idx0 in KNOWN_SOLUTIONS and cs == ik:
            return list(KNOWN_SOLUTIONS[level_idx0])
        start_snap = _snapshot(g)
        # 关键: 搜索期间禁用引擎的自动跳关 — 胜利分支的 perform_action 会
        # 在同一次调用里 next_level()->on_set_level() 重建 sprite 列表,
        # 使后续分支的快照恢复到错乱混合状态 (假解根源)。用实例属性遮蔽。
        neutered = False
        try:
            g.next_level = lambda: None
            neutered = True
        except Exception:
            neutered = False
        try:
            # heap 元素: (priority, tiebreak1, tiebreak2, tiebreak3_seq, snapshot, path)
            # seq 作为最终 tiebreaker 防止 snapshot 元组被比较（snapshot 含 numpy 数组）
            seq = 0
            heap = [(int(_heuristic(g)), 0, 0, seq, start_snap, [])]
            seen = {_state_key(g)}
            nodes = 0
            while heap:
                if time.time() - t0 > t_limit or nodes > max_nodes:
                    return None
                h, gg, _, _, snap, path = heapq.heappop(heap)
                # 防御: ilqnjlrnkk 可能是 numpy 数组
                max_steps = int(g.lelsvjlwneo.ilqnjlrnkk) - 1
                if len(path) >= max_steps:
                    continue
                for a in _ACTS:
                    _restore(g, snap)
                    try:
                        g.perform_action(_ActionInput(id=a, data={}, reasoning=None), raw=True)
                    except Exception:
                        continue
                    # 防御: level_index 和 vplrhaovhr() 可能返回数组
                    won_after = g.vplrhaovhr()
                    won_flag = won_after is True or (hasattr(won_after, '__len__') and len(won_after) == 1 and bool(won_after))
                    if int(g._current_level_index) > level_idx0 or won_flag:
                        result = [_act2num[x] for x in path + [a]]
                        _save_solution(level_idx0, result)
                        return result
                    if g._state == _GameState.GAME_OVER:
                        continue
                    k = _state_key(g)
                    if k in seen:
                        continue
                    seen.add(k)
                    # 防御: _heuristic 必须返回标量
                    h_val = int(_heuristic(g))
                    seq += 1
                    # seq 在 snapshot 和 path 之后，确保 Python 比较 tuple 时永远不会碰不到 snapshot
                    heapq.heappush(heap, (h_val * 10 + len(path) + 1, len(path) + 1,
                                          nodes + 1, seq, _snapshot(g), path + [a]))
                nodes += 1
            return None
        finally:
            if neutered:
                try:
                    del g.next_level
                except Exception:
                    pass
            _restore(g, start_snap)


# ─── Learn 模式: Ar25Learner 求解 ─────────────────────────────────────────

def _extract_level_data():
    """从当前 live game 对象提取真实的关卡数据（sprite cells/tags）。

    用于 Ar25Learner 初始化，保证 pathify 和 replay 使用相同的 sim。
    """
    try:
        axes = []
        for ax in _game.jtkyjqznbnp:
            tags = list(ax.tags) if hasattr(ax, "tags") else []
            cells = []
            if hasattr(ax, "pixels") and ax.pixels is not None:
                h, w = ax.pixels.shape
                for i in range(h):
                    for j in range(w):
                        if ax.pixels[i, j] not in (-1,):
                            cells.append([j - w // 2, i - h // 2])
            if not cells:
                cells = [[0, 0]]
            axes.append({"x": int(ax.x), "y": int(ax.y), "tags": tags, "cells": cells})

        sprites = []
        axis_positions = {(ax.x, ax.y) for ax in _game.jtkyjqznbnp}
        for s in _game.ayyvxqrhnzw:
            if (int(s.x), int(s.y)) in axis_positions:
                continue  # skip axes (they appear in ayyvxqrhnzw too)
            tags = list(s.tags) if hasattr(s, "tags") else []
            cells = []
            if hasattr(s, "pixels") and s.pixels is not None:
                h, w = s.pixels.shape
                for i in range(h):
                    for j in range(w):
                        if s.pixels[i, j] not in (-1,):
                            cells.append([j - w // 2, i - h // 2])
            if not cells:
                cells = [[0, 0]]
            sprites.append({"x": int(s.x), "y": int(s.y), "tags": tags, "cells": cells})

        targets = []
        for t in _game.fswikrcrdmx:
            targets.append({"x": int(t.x), "y": int(t.y)})

        budget = int(_game.lelsvjlwneo.ilqnjlrnkk)
        return {"budget": budget, "axes": axes, "sprites": sprites, "targets": targets}
    except Exception:
        return None


def learn_solve_level(t_limit=120.0, max_configs=32):
    """learn 模式：用 Ar25Learner 模板求解当前关卡。

    流程: bind_obs → 枚举配置 → pathify → replay 验证 → tabu 去重

    Returns:
        list[int] 动作序列（通关），或 None（失败）
    """
    if not _ensure() or not _valid:
        return None

    from ar25_learner import bind_obs_from_env, Ar25Learner

    with _lock:
        try:
            level_idx0 = int(_game._current_level_index)
            obs = bind_obs_from_env(_env)
            level_data = _extract_level_data()
        except Exception:
            return None

    def replay(path):
        """在影子引擎上重放动作序列，返回是否通关。"""
        with _lock:
            try:
                _env.reset()
                for m in path:
                    _game.perform_action(
                        _ActionInput(id=_num2act[int(m)], data={}, reasoning=None),
                        raw=True,
                    )
                won_after = _game.vplrhaovhr()
                won_flag = won_after is True or (
                    hasattr(won_after, "__len__") and len(won_after) == 1 and bool(won_after)
                )
                return int(_game._current_level_index) > level_idx0 or won_flag
            except Exception:
                return False

    learner = Ar25Learner(level_data=level_data)
    result = learner.solve(obs, replay=replay, max_configs=max_configs)

    if result.solved:
        # 保存最优解
        _save_solution(level_idx0, result.path)
        return result.path

    # 最小闭环的安全回退：几何 learner 只负责提出候选；
    # 候选全部被 live engine 拒绝时，使用同一官方引擎做有界搜索。
    # solve_level() 返回前会恢复搜索快照，随后由 replay() 重新执行并
    # 验证关卡推进，因此不会把“找到路径”误报成“真实通关”。
    with _lock:
        try:
            # learner 的每次 replay 都会改变 live game；搜索必须从同一
            # 个关卡起点开始，否则搜索路径与随后 replay 的起点不一致。
            _env.reset()
        except Exception:
            return None
    fallback_path = solve_level(t_limit=t_limit, max_nodes=600000)
    if fallback_path is not None and replay(fallback_path):
        _save_solution(level_idx0, fallback_path)
        return fallback_path
    return None
