"""FT09 本地引擎影子模拟器 + 线性代数求解器

FT09 机制 (点灯解谜):
- 4x4 网格, 每格 3x3 像素, 点击 (ACTION6 + 显示坐标) 循环颜色
- Hkx 格: 点击只循环自身颜色 (模板 elp 中心)
- NTi 格: 点击循环自身 + 像素值 6 标记的邻居 (模板编码在外观中)
- bsT 检查器: 8 邻居约束, 像素==0 → 邻居颜色必须等于检查器中心色, 否则必须不等
- 每关有步数限制 (kCv), 每次动作递减, 归零失败
- 全部约束满足 → 过关

求解策略:
1. 引擎探测构建点击影响矩阵 A (点击 j 影响哪些格)
2. 从 bsT 约束推导每格目标色
3. 模 k 线性方程组 A^T x = b 求解 (k = 色环长度)
4. 引擎验证: 应用点击序列后 cgj() 为真
"""
import os
import json
import threading
import time

os.environ.setdefault("ARC_API_KEY", "f1d9f364-5ac8-4486-a4e9-6b9286de351f")

_lock = threading.RLock()
_env = None
_game = None
_applied = 0
_valid = False
_error = None
_ActionInput = None
_GameState = None
_ACTION6 = None


def _ensure():
    global _env, _game, _applied, _valid, _error, _ActionInput, _GameState, _ACTION6
    if _env is not None:
        return _valid
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput, GameState
        arc = arc_agi.Arcade()
        _env = arc.make("ft09-0d8bbf25")
        _game = _env._game
        _ActionInput = ActionInput
        _GameState = GameState
        _ACTION6 = GameAction.ACTION6
        _env.reset()
        _applied = 0
        _valid = True
    except Exception as e:
        _error = str(e)
        _valid = False
    return _valid


def reset():
    """线上 arc-start 时调用: 影子回到 L1 起点"""
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


def apply(x, y=None):
    """兼容双参 (x, y) 与字典 {"a":6,"x","y"} 两种调用形式"""
    if isinstance(x, dict):
        y = int(x.get("y", 0))
        x = int(x.get("x", 0))
    else:
        x, y = int(x), int(y if y is not None else 0)
    """线上点击成功后调用: 影子同步执行同一点击 (显示坐标)"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _game.perform_action(
                _ActionInput(id=_ACTION6, data={"x": int(x), "y": int(y)}, reasoning=None), raw=True)
            _applied += 1
            return True
        except Exception:
            _valid = False
            return False


def is_synced(actions_taken):
    return _valid and _applied == int(actions_taken or 0)


def replay(history):
    """自愈重建: 影子回到 L1 并重放完整点击历史
    兼容: {"x","y"} 字典 / [6,x,y] 三元组 / 扁平 [6,x,y,...] 列表"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            clicks = []
            hs = list(history)
            i = 0
            while i < len(hs):
                h = hs[i]
                if isinstance(h, dict):
                    clicks.append((int(h.get("x", 0)), int(h.get("y", 0))))
                    i += 1
                elif isinstance(h, (list, tuple)):
                    if len(h) >= 3:
                        clicks.append((int(h[1]), int(h[2])))
                    elif len(h) == 2:
                        clicks.append((int(h[0]), int(h[1])))
                    i += 1
                else:
                    clicks.append((int(hs[i + 1]), int(hs[i + 2])))
                    i += 3
            _env.reset()
            for (x, y) in clicks:
                _game.perform_action(
                    _ActionInput(id=_ACTION6, data={"x": x, "y": y}, reasoning=None), raw=True)
            _applied = len(clicks)
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def status():
    return {"valid": _valid, "applied": _applied, "error": _error, "loaded": _env is not None}


# ─── 内部: 快照与求解 ─────────────────────────────────

def _cells(g):
    """当前关所有可变色格 (Hkx + NTi)"""
    lv = g.current_level
    return [s for s in lv._sprites if ("Hkx" in s.tags) or ("NTi" in s.tags)]


def _center(s):
    return int(s.pixels[1][1])


def _snapshot(g):
    spr = [(s.x, s.y, s.pixels.copy()) for s in g.current_level._sprites]
    zth_idx = None
    zth = getattr(g, "zth", None)
    if zth is not None:
        for i, s in enumerate(g.current_level._sprites):
            if s is zth:
                zth_idx = i
                break
    misc = (int(g.lpw.dzy), int(getattr(g, "our", 0)), g._state)
    return (spr, zth_idx, misc)


def _restore(g, snap):
    spr, zth_idx, misc = snap
    sprites = g.current_level._sprites
    for s, (x, y, px) in zip(sprites, spr):
        s.set_position(x, y)
        s.pixels = px.copy()
    if zth_idx is not None and 0 <= zth_idx < len(sprites):
        g.zth = sprites[zth_idx]
    g.lpw.dzy, g.our, st = misc
    g._state = st


def _state_key(g):
    return tuple(_center(s) for s in _cells(g))


def _click_disp(g, s):
    """格子精灵 (3x3) -> 显示坐标 (64x64 空间).

    display = world * scale + letterbox_padding, 取精灵中心 (x+1, y+1).
    """
    cam = g._camera
    scale = max(1, int(64 / cam.width))
    pad_x = int((64 - cam.width * scale) / 2)
    pad_y = int((64 - cam.height * scale) / 2)
    wx, wy = int(s.x) + 1, int(s.y) + 1
    return (wx * scale + pad_x, wy * scale + pad_y)


def _extract_constraints(g):
    """从 bsT 检查器提取约束.

    返回: {cell_sprite: {"eq": color 或 None, "ne": [colors...]}}
    eq: 必须等于的颜色; ne: 必须不等于的颜色列表
    """
    lv = g.current_level
    pos_map = {}
    for s in _cells(g):
        pos_map[(int(s.x), int(s.y))] = s

    cons = {}
    for etf in lv.get_sprites_by_tag("bsT"):
        nRq = _center(etf)
        for j in range(3):
            for i in range(3):
                if j == 1 and i == 1:
                    continue  # 检查器自身位置
                dx, dy = (i - 1) * 4, (j - 1) * 4
                nb = pos_map.get((int(etf.x) + dx, int(etf.y) + dy))
                if nb is None:
                    continue
                require_eq = (etf.pixels[j][i] == 0)
                c = cons.setdefault(nb, {"eq": None, "ne": []})
                if require_eq:
                    if c["eq"] is None:
                        c["eq"] = nRq
                else:
                    if nRq not in c["ne"]:
                        c["ne"].append(nRq)
    return cons


def _targets(g, palette, cons):
    """推导每格目标色. 返回 {sprite: target_color}; 无约束格目标 = 当前色."""
    targets = {}
    for s in _cells(g):
        c = cons.get(s)
        cur = _center(s)
        if not c:
            targets[s] = cur
            continue
        if c["eq"] is not None:
            targets[s] = c["eq"]
        elif len(palette) == 2:
            # 两色: != c 即为另一色
            others = [p for p in palette if p not in c["ne"]]
            targets[s] = others[0] if others else cur
        else:
            # 多色且只有 != 约束: 任选一个允许色 (优先当前色)
            allowed = [p for p in palette if p not in c["ne"]]
            targets[s] = cur if cur in allowed else (allowed[0] if allowed else cur)
    return targets


def _probe_matrix(g, cells):
    """引擎探测: 点击每格一次, 记录影响哪些格 (各 +1).

    返回: {j: {i: 1}} 邻接 (cells[j] 点击影响 cells[i])
    """
    A = {}
    base = _snapshot(g)
    base_colors = [_center(s) for s in cells]
    pal = list(g.gqb)
    for j, cj in enumerate(cells):
        try:
            _restore(g, base)
            dx, dy = _click_disp(g, cj)
            g.perform_action(_ActionInput(
                id=_ACTION6, data={"x": dx, "y": dy}, reasoning=None), raw=True)
        except Exception:
            A[j] = {}
            continue
        after = [_center(s) for s in cells]
        eff = {}
        for i, (b, a) in enumerate(zip(base_colors, after)):
            if b != a:
                # 确认是 +1 步进 (色环循环)
                try:
                    step = (pal.index(a) - pal.index(b)) % len(pal)
                except ValueError:
                    step = 1
                if step:
                    eff[i] = step
        A[j] = eff
    _restore(g, base)
    return A


def _solve_mod(A, b, k, n):
    """解模 k 线性方程组: sum_j x_j * A[j][i] = b[i] (mod k).

    A: {j: {i: step}}; b: [n]; 未知数 x_j ∈ [0, k).
    返回 x 列表或 None.
    """
    # 增广矩阵: n 行 (方程 i), n+1 列 (x_0..x_{n-1}, b_i)
    rows = []
    for i in range(n):
        row = [0] * (n + 1)
        for j in range(n):
            row[j] = A.get(j, {}).get(i, 0) % k
        row[n] = b[i] % k
        rows.append(row)

    # 高斯消元 (模 k, k 很小; 用扩展欧几里得求逆, 仅处理可逆系数)
    def inv(a):
        for x in range(1, k):
            if (a * x) % k == 1:
                return x
        return None

    piv_col = 0
    piv_rows = []
    for r in range(n):
        # 找主元
        pr = None
        for rr in range(r, n):
            if rows[rr][piv_col] % k != 0 and inv(rows[rr][piv_col] % k) is not None:
                pr = rr
                break
        if pr is None:
            piv_col += 1
            if piv_col > n:
                break
            continue
        rows[r], rows[pr] = rows[pr], rows[r]
        iv = inv(rows[r][piv_col] % k)
        rows[r] = [(v * iv) % k for v in rows[r]]
        for rr in range(n):
            if rr != r and rows[rr][piv_col] % k != 0:
                f = rows[rr][piv_col] % k
                rows[rr] = [(v - f * w) % k for v, w in zip(rows[rr], rows[r])]
        piv_rows.append(piv_col)
        piv_col += 1
        if piv_col > n:
            break

    # 检查矛盾行 0=非0
    for row in rows:
        if all(v % k == 0 for v in row[:n]) and row[n] % k != 0:
            return None

    x = [0] * n
    for r, col in enumerate(piv_rows):
        x[col] = rows[r][n] % k
    # 免变量默认 0 — 验证阶段兜底
    return x


def solve_level(t_limit=60.0):
    """求解影子当前关 — 多算法自动级联.

    1. 线性代数 (模 k 方程组) — 极快, 适合线性可解关卡
    2. BFS — 状态空间 < 50 万 时适用
    3. IDDFS (迭代加深) — 状态空间大时的深度优先搜索
    4. 随机重启爬山 — 极大赛空间的最后兜底

    返回: 点击坐标列表 [[x, y], ...] (显示坐标), 已通关 pending 返回 [],
          失败返回 None.
    """
    if not _ensure() or not _valid:
        return None
    with _lock:
        g = _game
        t0 = time.time()
        level_idx0 = int(g._current_level_index)

        cells = _cells(g)
        if not cells:
            return []

        palette = list(g.gqb)
        k = len(palette)
        cons = _extract_constraints(g)
        targets = _targets(g, palette, cons)
        n = len(cells)

        clickable = [s for s in cells if _center(s) in palette]
        click_disps = [_click_disp(g, s) for s in cells]

        # 预探测 A 矩阵 (耗时, 只做一次)
        A = _probe_matrix(g, cells)

        pal_idx = {c: i for i, c in enumerate(palette)}
        b = [0] * n
        for i, s in enumerate(cells):
            t = targets[s]
            c = _center(s)
            if c in pal_idx and t in pal_idx:
                b[i] = (pal_idx[t] - pal_idx[c]) % k
            elif c != t:
                b[i] = 1
            else:
                b[i] = 0

        snap0 = _snapshot(g)

        def neuter():
            try:
                g.next_level = lambda: None
                return True
            except Exception:
                return False

        def restore():
            try:
                del g.next_level
            except Exception:
                pass
            _restore(g, snap0)

        def verify(clicks):
            ng = neuter()
            try:
                for cx, cy in clicks:
                    g.perform_action(_ActionInput(id=_ACTION6, data={"x": cx, "y": cy}, reasoning=None), raw=True)
                return bool(g.cgj())
            except Exception:
                return False
            finally:
                restore()

        # ═══ 算法 1: 线性代数 ═══
        sub_A = {jj: {i: st for i, st in A.get(jj, {}).items()} for jj in range(n)}
        x = _solve_mod(sub_A, b, k, n)
        if x is not None:
            clicks = []
            for j, cnt in enumerate(x):
                for _ in range(int(cnt) % k):
                    if j < n and cells[j] in clickable:
                        dx, dy = _click_disp(g, cells[j])
                        clicks.append([dx, dy])
            if clicks and verify(clicks):
                return clicks

        # ═══ 算法 2: BFS (放宽状态限制) ═══
        result = _solve_bfs(g, t_limit, level_idx0, max_states=500000)
        if result is not None:
            if result and verify(result):
                return result
            restore()

        # ═══ 算法 3: IDDFS (迭代加深, 无状态空间限制) ═══
        result = _solve_iddfs(g, t_limit, level_idx0, max_depth=30)
        if result is not None and result and verify(result):
            return result
        restore()

        # ═══ 算法 4: 随机重启爬山 (极大赛空间) ═══
        result = _solve_random(g, t_limit, level_idx0, max_restarts=50)
        if result is not None and result and verify(result):
            return result
        restore()

        return None


def _solve_bfs(g, t_limit, level_idx0, max_states=500000, max_nodes=None):
    """引擎 BFS 搜索.

    max_states: 状态去重表上限 (超过则提前退出)
    max_nodes: 可选, 搜索节点数上限
    """
    cells = _cells(g)
    n = len(cells)
    total_possible = len(g.gqb) ** n
    # 完全禁止搜索的硬上限 (状态空间爆炸)
    if total_possible > 5_000_000:
        return None

    start_snap = _snapshot(g)
    start_key = _state_key(g)
    from collections import deque
    prev = {start_key: None}
    click_disps = [_click_disp(g, s) for s in cells]
    queue = deque([start_key])
    keys_snap = {start_key: start_snap}

    neutered = False
    try:
        g.next_level = lambda: None
        neutered = True
    except Exception:
        neutered = False

    t0 = time.time()
    found = None
    nodes_visited = 0
    try:
        while queue and time.time() - t0 < t_limit:
            if max_nodes is not None and nodes_visited >= max_nodes:
                break
            key = queue.popleft()
            _restore(g, keys_snap[key])
            for j in range(n):
                if max_nodes is not None and nodes_visited >= max_nodes:
                    break
                try:
                    g.perform_action(_ActionInput(
                        id=_ACTION6, data={"x": click_disps[j][0], "y": click_disps[j][1]}, reasoning=None), raw=True)
                except Exception:
                    _restore(g, keys_snap[key])
                    continue
                nodes_visited += 1
                nk = _state_key(g)
                if nk not in prev:
                    prev[nk] = (key, j)
                    keys_snap[nk] = _snapshot(g)
                    if g.cgj():
                        found = nk
                        break
                    if len(prev) < max_states:
                        queue.append(nk)
                _restore(g, keys_snap[key])
            if found:
                break
    finally:
        if neutered:
            try:
                del g.next_level
            except Exception:
                pass
        _restore(g, start_snap)

    if not found:
        return None
    clicks = []
    k = found
    while prev[k] is not None:
        pk, j = prev[k]
        clicks.append(list(click_disps[j]))
        k = pk
    clicks.reverse()
    return clicks


def _solve_iddfs(g, t_limit, level_idx0, max_depth=30):
    """迭代加深 DFS — 无状态空间限制, 按深度逐步搜索."""
    cells = _cells(g)
    n = len(cells)
    click_disps = [_click_disp(g, s) for s in cells]
    start_snap = _snapshot(g)

    neutered = False
    try:
        g.next_level = lambda: None
        neutered = True
    except Exception:
        neutered = False

    t0 = time.time()
    found_path = None

    def dfs(depth, path):
        nonlocal found_path
        if time.time() - t0 > t_limit:
            return True  # 超时, 停止搜索
        if g.cgj():
            found_path = path[:]
            return True
        if depth >= max_depth:
            return False
        for j in range(n):
            try:
                g.perform_action(_ActionInput(
                    id=_ACTION6, data={"x": click_disps[j][0], "y": click_disps[j][1]}, reasoning=None), raw=True)
                found = dfs(depth + 1, path + [list(click_disps[j])])
                _restore(g, start_snap)
                if found:
                    return True
            except Exception:
                _restore(g, start_snap)
                continue
        return False

    try:
        for d in range(max_depth + 1):
            if time.time() - t0 > t_limit:
                break
            _restore(g, start_snap)
            if dfs(d, []):
                break
    finally:
        if neutered:
            try:
                del g.next_level
            except Exception:
                pass
        _restore(g, start_snap)

    return found_path


def _solve_random(g, t_limit, level_idx0, max_restarts=50, max_steps=100):
    """随机重启爬山 — 极大赛空间的最后兜底.

    每次重启: 随机打乱初始状态, 然后贪心爬山
    """
    cells = _cells(g)
    n = len(cells)
    click_disps = [_click_disp(g, s) for s in cells]
    import random

    start_snap = _snapshot(g)
    neutered = False
    try:
        g.next_level = lambda: None
        neutered = True
    except Exception:
        neutered = False

    t0 = time.time()
    best = None
    best_unsat = float('inf')

    try:
        for restart in range(max_restarts):
            if time.time() - t0 > t_limit:
                break
            _restore(g, start_snap)

            # 随机初始化点击
            init_clicks = random.randint(0, n * 2)
            for _ in range(init_clicks):
                j = random.randint(0, n - 1)
                try:
                    g.perform_action(_ActionInput(
                        id=_ACTION6, data={"x": click_disps[j][0], "y": click_disps[j][1]}, reasoning=None), raw=True)
                except Exception:
                    _restore(g, start_snap)
                    break

            # 贪心爬山: 每步找最能减少约束违背的动作
            path = []
            for step in range(max_steps):
                if time.time() - t0 > t_limit:
                    break
                if g.cgj():
                    return path
                # 找 best move
                best_j = -1
                best_score = _count_unsatisfied(g)
                saved_snap = _snapshot(g)
                for j in range(n):
                    try:
                        g.perform_action(_ActionInput(
                            id=_ACTION6, data={"x": click_disps[j][0], "y": click_disps[j][1]}, reasoning=None), raw=True)
                        score = _count_unsatisfied(g)
                        if score < best_score:
                            best_score = score
                            best_j = j
                        _restore(g, saved_snap)
                    except Exception:
                        _restore(g, saved_snap)
                        continue
                if best_j >= 0:
                    g.perform_action(_ActionInput(
                        id=_ACTION6, data={"x": click_disps[best_j][0], "y": click_disps[best_j][1]}, reasoning=None), raw=True)
                    path.append(list(click_disps[best_j]))
                    if best_score == 0:
                        return path
                else:
                    break

            # 记录本轮最优
            unsat = _count_unsatisfied(g)
            if unsat < best_unsat:
                best_unsat = unsat
                best = path[:]
                if unsat == 0:
                    return path
    finally:
        if neutered:
            try:
                del g.next_level
            except Exception:
                pass
        _restore(g, start_snap)

    return best if best_unsat < float('inf') else None


def _count_unsatisfied(g):
    """计算当前违背的约束数量 (用于爬山法评估)."""
    try:
        cons = _extract_constraints(g)
        palette = list(g.gqb)
        pal_idx = {c: i for i, c in enumerate(palette)}
        cells = _cells(g)
        pos_map = {(int(s.x), int(s.y)): s for s in cells}
        violations = 0
        for etf in g.current_level.get_sprites_by_tag("bsT"):
            center_c = _center(etf)
            for j in range(3):
                for i in range(3):
                    if j == 1 and i == 1:
                        continue
                    dx, dy = (i - 1) * 4, (j - 1) * 4
                    nb = pos_map.get((int(etf.x) + dx, int(etf.y) + dy))
                    if nb is None:
                        continue
                    nb_c = _center(nb)
                    if etf.pixels[j][i] == 0:
                        if nb_c != center_c:
                            violations += 1
                    else:
                        if nb_c == center_c:
                            violations += 1
        return violations
    except Exception:
        return 0
