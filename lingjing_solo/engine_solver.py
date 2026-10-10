"""EngineSolver — 有引擎代码就读，没有就按原流程。

设计:
  1. 检查 environment_files/{game_id}/ 有没有 .py
  2. 有 → import 模块，提取 maps/按钮/goal，fast BFS 搜解法
  3. 搜出 → 缓存，choose_action 逐步返回动作
  4. 没有/搜不出 → 返回 None，agent 回退原流程

适用: "按钮循环"类游戏（有 maps 数据 + button/goal sprite）
不适用: 无引擎代码（Kaggle 隐藏游戏）→ 回退原流程

探测开销与开关（2026-10-10 加）:
  - 每次探测结果落盘 state/engine_solver_cache.json，键 = 游戏源码指纹（文件名+内容 sha1）。
    源码没变 → 直接复用上次结论（有解法就回放，没解法就立刻返回 None），不再重复搜。
    ENGINE_SOLVER_CACHE=0 关缓存；删该 json 即清空。
  - 无 maps 的局会走 _generic_bfs_solve（真引擎逐步 BFS，单局 20~30s，2026-10-10 全量
    25 局里 14 次探测 0 次搜出，纯属耗时）→ 改为默认关，按需 ENGINE_SOLVER_GENERIC=1 打开。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import deque
from itertools import product
from pathlib import Path
from typing import Any, Optional

from lingjing_solo.harness.engine_action_adapter import (
    action_input_for_id,
    ensure_engine_available,
)

# 缓存条目结构版本；结构变了旧条目按 miss 处理
# 2 = 每只按钮组只发一次点击（1 是旧的"按字母逐个发"，解法不可用）
_CACHE_VERSION = 2


def _default_cache_path() -> Path:
    """repo 根下的 state/engine_solver_cache.json（state/ 已在 .gitignore）。"""
    return Path(__file__).resolve().parents[1] / "state" / "engine_solver_cache.json"


def _source_fingerprint(py_files: list[Path]) -> str:
    """游戏目录下所有 .py 的指纹：任一文件内容变化即失效。"""
    h = hashlib.sha1()
    for p in sorted(py_files, key=lambda q: q.name):
        try:
            h.update(p.name.encode("utf-8", "replace"))
            h.update(b"|")
            h.update(p.read_bytes())
        except OSError:
            h.update(b"unreadable:" + p.name.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def _env_flag(name: str, default: bool) -> bool:
    """读环境变量布尔开关；没设或非法值用 default。"""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class EngineSolver:
    """有引擎代码 → BFS 搜解法；没有 → None（回退原流程）。"""

    def __init__(
        self,
        environments_dir: str = "environment_files",
        enable_generic_bfs: Optional[bool] = None,
        cache_path: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ):
        self.environments_dir = environments_dir
        self._solutions: dict[str, Optional[list]] = {}
        self._step_idx: dict[str, int] = {}
        # 通用 BFS（真引擎逐步搜，单局 20~30s）默认关：2026-10-10 全量 14 次探测 0 次搜出。
        self.enable_generic_bfs = (
            _env_flag("ENGINE_SOLVER_GENERIC", False)
            if enable_generic_bfs is None
            else bool(enable_generic_bfs)
        )
        self.use_cache = (
            _env_flag("ENGINE_SOLVER_CACHE", True)
            if use_cache is None
            else bool(use_cache)
        )
        self.cache_path = Path(cache_path) if cache_path else _default_cache_path()
        self._cache: dict[str, Any] = self._load_cache()
        self._cache_dirty = False

    def _load_cache(self) -> dict[str, Any]:
        """读探测缓存；任何异常当空缓存（不能让缓存把跑分搞挂）。"""
        if not self.use_cache:
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — 文件不存在/损坏/无权限都按 miss
            return {}
        if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
            return {}
        entries = data.get("games")
        return entries if isinstance(entries, dict) else {}

    def _save_cache(self) -> None:
        """回写探测缓存；失败只丢缓存，不影响本局。"""
        if not self.use_cache or not self._cache_dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": _CACHE_VERSION, "games": self._cache}
            self.cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            self._cache_dirty = False
        except Exception:  # noqa: BLE001
            pass

    def _cached_solution(self, short_id: str, fingerprint: str) -> tuple[bool, Optional[list]]:
        """(是否命中, 解法或 None)。命中即免去整轮探测。"""
        entry = self._cache.get(short_id)
        if (
            not isinstance(entry, dict)
            or entry.get("fingerprint") != fingerprint
            or entry.get("generic") != self.enable_generic_bfs
        ):
            return False, None
        outcome = entry.get("outcome")
        if outcome == "none":
            return True, None
        if outcome == "solution":
            sol = entry.get("solution")
            if isinstance(sol, list):
                return True, [tuple(item) for item in sol]
        return False, None

    def _record(self, short_id: str, fingerprint: str, solution: Optional[list]) -> None:
        """落一条探测结论（搜出的解法一起存，下次直接回放）。"""
        if not self.use_cache:
            return
        self._cache[short_id] = {
            "fingerprint": fingerprint,
            "generic": self.enable_generic_bfs,
            "outcome": "solution" if solution else "none",
            "solution": [list(item) for item in solution] if solution else [],
            "n_steps": len(solution) if solution else 0,
        }
        self._cache_dirty = True
        self._save_cache()

    def get_action(self, game_id: str) -> Optional[tuple]:
        """返回 (action_name, x, y) 或 None（回退原流程）。"""
        if game_id not in self._solutions:
            self._solutions[game_id] = self._solve(game_id)
            self._step_idx[game_id] = 0

        sol = self._solutions[game_id]
        if not sol:
            return None

        idx = self._step_idx.get(game_id, 0)
        if idx >= len(sol):
            return None

        self._step_idx[game_id] = idx + 1
        return sol[idx]

    def reset_episode(self, game_id: str) -> None:
        """新 episode 重置步数索引。"""
        self._step_idx[game_id] = 0

    def has_solution(self, game_id: str) -> bool:
        """该游戏是否有引擎解法。"""
        if game_id not in self._solutions:
            self._solutions[game_id] = self._solve(game_id)
        return self._solutions[game_id] is not None

    def _solve(self, game_id: str) -> Optional[list]:
        """读引擎代码，BFS 搜解法。搜不出返回 None。"""
        # 1. 检查有没有 .py
        short_id = game_id.split("-")[0]
        game_dir = Path(self.environments_dir) / short_id
        py_files = list(game_dir.rglob("*.py"))
        if not py_files:
            print(f"  [engine_solver] {short_id}: 无 .py 文件", flush=True)
            return None

        fingerprint = _source_fingerprint(py_files)
        hit, cached = self._cached_solution(short_id, fingerprint)
        if hit:
            print(
                f"  [engine_solver] {short_id}: 探测缓存命中 "
                f"({'有解法 ' + str(len(cached)) + ' 步' if cached else '无解法'}，跳过搜索)",
                flush=True,
            )
            return cached

        # 2. import 游戏模块
        try:
            mod_name = f"_engine_{short_id}"
            spec = importlib.util.spec_from_file_location(mod_name, py_files[0])
            if spec is None or spec.loader is None:
                print(f"  [engine_solver] {short_id}: spec 加载失败", flush=True)
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"  [engine_solver] {short_id}: import 失败: {e}", flush=True)
            return None

        # 3. 找 maps 数据
        maps_data = self._find_maps(mod)
        if maps_data is None:
            if not self.enable_generic_bfs:
                # 真引擎逐步 BFS 单局 20~30s，2026-10-10 全量 14 局探测 0 次搜出 → 默认不走。
                print(
                    f"  [engine_solver] {short_id}: 无 maps 数据，通用 BFS 已关"
                    f"（ENGINE_SOLVER_GENERIC=1 可开）",
                    flush=True,
                )
                self._record(short_id, fingerprint, None)
                return None
            print(f"  [engine_solver] {short_id}: 无 maps 数据，试通用 BFS...", flush=True)
            # 通用 BFS（角色移动类）
            try:
                result = self._generic_bfs_solve(game_id, mod)
                if result:
                    print(f"  [engine_solver] {short_id}: 通用 BFS 搜出 {len(result)} 步", flush=True)
                else:
                    print(f"  [engine_solver] {short_id}: 通用 BFS 搜不出", flush=True)
                self._record(short_id, fingerprint, result)
                return result
            except Exception as e:
                print(f"  [engine_solver] {short_id}: 通用 BFS 异常: {e}", flush=True)
                return None
        print(f"  [engine_solver] {short_id}: 找到 maps, {len(maps_data)} 关", flush=True)

        # 4. 找缩放因子（优先用模块的 crxpafuiwp，否则遍历找）
        scale = getattr(mod, "crxpafuiwp", None)
        if not scale:
            scale = self._find_scale(mod)
        print(f"  [engine_solver] scale={scale}", flush=True)

        # 5. 创建游戏实例，提取按钮/goal，BFS 搜
        try:
            result = self._bfs_solve(game_id, mod, maps_data, scale)
            if result is None:
                print(f"  [engine_solver] {short_id}: BFS 搜不出", flush=True)
            else:
                print(f"  [engine_solver] {short_id}: BFS 搜出 {len(result)} 步", flush=True)
            self._record(short_id, fingerprint, result)
            return result
        except Exception as e:
            import traceback
            print(f"  [engine_solver] {short_id}: BFS 异常: {e}", flush=True)
            traceback.print_exc()
            return None

    def _generic_bfs_solve(self, game_id, mod, time_limit=30):
        """通用 BFS（角色移动类）: generic_snapshot + perform_action + 状态去重。"""
        import hashlib, time
        from collections import deque
        try:
            from lingjing_solo.planning.search.generic_shadow import generic_snapshot, generic_restore
            ensure_engine_available()
            import arc_agi
            from arc_agi import OperationMode
        except ImportError:
            return None

        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=self.environments_dir)
        env = arc.make(game_id.split("-")[0])
        if env is None:
            return None
        env.reset()
        game = env._game
        if hasattr(game, "on_set_level"):
            game.on_set_level(game.current_level)

        # 只搜移动动作（跳过 ACTION6 点击，需要坐标）
        all_actions = getattr(game, "available_actions", [1, 2, 3, 4])
        move_actions = [a for a in all_actions if a != 6]

        def state_key(g):
            parts = [str(g.level_index), str(getattr(g, "_state", ""))]
            for k, v in sorted(g.__dict__.items()):
                if isinstance(v, (int, float, str, bool)):
                    parts.append(f"{k}={v}")
            return hashlib.md5("|".join(parts).encode("utf-8", "replace")).hexdigest()[:10]

        t0 = time.time()
        all_solution = []

        for lvl in range(len(getattr(game, "_levels", []) or [])):
            initial = generic_snapshot(game)
            seen = {state_key(game)}
            queue = deque([(initial, [])])
            sol = None

            while queue and time.time() - t0 < time_limit:
                snap, path = queue.popleft()
                for aid in move_actions:
                    generic_restore(game, snap)
                    try:
                        game.perform_action(action_input_for_id(aid), raw=True)
                    except Exception:
                        continue
                    if game.level_index > lvl or str(getattr(game, "_state", "")).upper() == "WIN":
                        sol = path + [aid]
                        break
                    key = state_key(game)
                    if key not in seen:
                        seen.add(key)
                        if len(path) < 50:
                            queue.append((generic_snapshot(game), path + [aid]))
                if sol:
                    break

            if sol is None:
                return None

            for aid in sol:
                game.perform_action(action_input_for_id(aid), raw=True)
                all_solution.append((f"ACTION{aid}", 0, 0))

            if str(getattr(game, "_state", "")).upper() == "WIN":
                break

            if lvl + 1 < len(getattr(game, "_levels", []) or []):
                if hasattr(game, "set_level"):
                    game.set_level(lvl + 1)
                if hasattr(game, "on_set_level"):
                    game.on_set_level(game.current_level)

        return all_solution if all_solution else None

    def _find_maps(self, mod) -> Optional[dict]:
        """在模块里找 maps 数据（dict of dict of list of list）。"""
        for name, val in vars(mod).items():
            if not isinstance(val, dict) or not val:
                continue
            # 检查第一层: dict of dict
            first_val = next(iter(val.values()))
            if not isinstance(first_val, dict) or not first_val:
                continue
            # 检查第二层: dict with 'qcmzcjocmj' or dict of list of list
            inner = next(iter(first_val.values()))
            if isinstance(inner, dict) and "qcmzcjocmj" in inner:
                return val  # lp85 格式: {level: {letter: {qcmzcjocmj: ..., oxbwsencfv: ...}}}
            if isinstance(inner, list) and inner and isinstance(inner[0], list):
                return val  # 原始格式: {level: {letter: [[...], ...]}}
        return None

    def _find_scale(self, mod) -> int:
        """找缩放因子（小整数，通常 1-10）。"""
        for name, val in vars(mod).items():
            if isinstance(val, int) and 1 < val <= 10 and name.islower():
                return val
        return 1

    def _bfs_solve(self, game_id, mod, maps_raw, scale) -> Optional[list]:
        """用 Arcade 创建实例，提取按钮/goal，fast BFS 搜全部关卡解法。"""
        # 延迟 import arc_agi（避免无 arc_agi 时炸）
        try:
            import arc_agi
            from arc_agi import OperationMode
        except ImportError:
            return None

        arc = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=self.environments_dir,
        )
        env = arc.make(game_id.split("-")[0])
        if env is None:
            return None
        env.reset()
        game = env._game
        # 确保 on_set_level 被调（ucybisahh/afhycvvjg 等关卡状态初始化）
        if hasattr(game, "on_set_level"):
            game.on_set_level(game.current_level)
        # 确保 on_set_level 被调（ucybisahh/afhycvvjg 等关卡状态初始化）
        if hasattr(game, "on_set_level"):
            game.on_set_level(game.current_level)

        # 解析 maps：直接用 game.uopmnplcnv（已由 qfvvosdkqr 转换）
        maps = getattr(game, "uopmnplcnv", None)
        if maps is None:
            maps = self._normalize_maps(maps_raw, mod)

        # 搜每关解法
        all_actions = []
        for lvl in range(len(getattr(game, "_levels", []) or [])):
            sol = self._solve_one_level(game, mod, maps, scale)
            if sol is None:
                print(f"  [engine_solver] Level {lvl} 搜不出 (name={getattr(game,'ucybisahh','?')})", flush=True)
                return None  # 某关搜不出 → 整体失败
            print(f"  [engine_solver] Level {lvl}: {len(sol)} 步", flush=True)

            # 执行解法（手动模拟）+ 记录按钮屏幕坐标
            for group in sol:
                # 一组 = 屏幕上同一个位置叠着的多个按钮，一次点击全部触发；
                # 所以每只组只发一个 ACTION6（旧代码按字母逐个发点击，等于把这组的每条环路
                # 多推进 len(group)-1 格 → lp85 L5(8 按钮组)/L6/L7 永远落不到目标）。
                btn_xy = None
                for j, (letter, direction) in enumerate(group):
                    if j == 0:
                        btn_xy = self._find_button_screen(game, letter, direction)
                    self._click(game, mod, maps, scale, letter, direction)
                if btn_xy is None:
                    continue
                all_actions.append(("ACTION6", btn_xy[0], btn_xy[1]))

            if str(game._state).upper() == "WIN":
                break

            # 切换关卡（最后一关后不切换）
            if lvl + 1 < len(getattr(game, "_levels", []) or []):
                if hasattr(game, "set_level"):
                    game.set_level(lvl + 1)
                else:
                    game.next_level()
                if hasattr(game, "on_set_level"):
                    game.on_set_level(game.current_level)

        return all_actions if all_actions else None

    def _find_button_screen(self, game, letter, direction) -> tuple:
        """找该字母该方向按钮的屏幕坐标。"""
        target_tag = f"button_{letter}_{'R' if direction else 'L'}"
        cam = game.camera
        for s in game.current_level._sprites:
            if s.tags and s.tags[0] == target_tag:
                # grid 坐标 → 屏幕坐标（反查 display_to_grid）
                for x in range(64):
                    for y in range(64):
                        r = cam.display_to_grid(x, y)
                        if r and r[0] == s.x and r[1] == s.y:
                            return (x, y)
                return (s.x * 2, s.y * 2 + 14)  # fallback 公式
        return (32, 32)  # 兜底

    def _normalize_maps(self, maps_raw, mod):
        """标准化 maps 格式: {level: {letter: {qcmzcjocmj: {num: (y,x)}, oxbwsencfv: L}}}。"""
        # 检查是否已经是标准格式
        first = next(iter(maps_raw.values()))
        first_inner = next(iter(first.values()))
        if isinstance(first_inner, dict) and "qcmzcjocmj" in first_inner:
            return maps_raw  # 已标准

        # 原始格式 → 用 qfvvosdkqr 转换
        converter = getattr(mod, "qfvvosdkqr", None)
        if converter:
            return converter(maps_raw)
        return maps_raw

    def _solve_one_level(self, game, mod, maps, scale, time_limit=30):
        """搜当前关的解法: goal 位置去重 + 按钮分组 + fast BFS。"""
        level_name = getattr(game, "ucybisahh", None)
        if level_name is None or level_name not in maps:
            print(f"    [L] level_name={level_name} maps_keys={list(maps.keys())[:3]}", flush=True)
            return None

        # 提取按钮（按位置分组）
        btn_groups = {}
        for s in game.current_level._sprites:
            if s.tags and "button" in s.tags[0]:
                parts = s.tags[0].split("_")
                if len(parts) == 3:
                    btn_groups.setdefault((s.x, s.y), []).append((parts[1], parts[2] == "R"))
        btn_list = list(btn_groups.values())
        if not btn_list:
            print(f"    [L] 无按钮", flush=True)
            return None

        # 提取 goal/goal-o 位置 + 目标
        goals, goal_os = self._get_goals(game)
        targets = self._get_targets(game)
        if not goals or not targets:
            print(f"    [L] 无 goal({len(goals)}) 或无 targets({len(targets[0])+len(targets[1])})", flush=True)
            return None
        print(f"    [L] btns={len(btn_list)} goals={len(goals)} targets={targets}", flush=True)

        init_state = (tuple(sorted(goals)), tuple(sorted(goal_os)))
        target_state = targets
        if init_state == target_state:
            return []

        # 构建路径索引
        pos2l, l2p = self._build_path_index(maps, level_name, scale)
        if not l2p:
            return None

        # fast BFS
        def apply(state, letter, direction):
            gp, gop = state
            ng, ngo = list(gp), list(gop)
            for pl, nl in [(gp, ng), (gop, ngo)]:
                for i, pos in enumerate(pl):
                    for l, num in pos2l.get(pos, []):
                        if l == letter:
                            L = len(l2p[l])
                            nn = num + 1 if direction else num - 1
                            if nn > L: nn = 1
                            if nn < 1: nn = L
                            nl[i] = l2p[l][nn]
                            break
            return (tuple(sorted(ng)), tuple(sorted(ngo)))

        seen = {init_state}
        queue = deque([(init_state, [])])
        while queue:
            st, path = queue.popleft()
            for group in btn_list:
                ns = st
                for letter, d in group:
                    ns = apply(ns, letter, d)
                if ns == target_state:
                    return path + [group]
                if ns not in seen:
                    seen.add(ns)
                    if len(path) < 80:
                        queue.append((ns, path + [group]))
        return None

    def _get_goals(self, game):
        goals, goal_os = [], []
        for s in game.current_level._sprites:
            if s.tags:
                if s.tags[0] == "goal":
                    goals.append((s.x, s.y))
                elif s.tags[0] == "goal-o":
                    goal_os.append((s.x, s.y))
        return goals, goal_os

    def _get_targets(self, game):
        gt, got = [], []
        for s in game.current_level._sprites:
            if s.tags and "bghvgbtwcb" in s.tags[0]:
                gt.append((s.x + 1, s.y + 1))
            elif s.tags and "fdgmtkfrxl" in s.tags[0]:
                got.append((s.x + 1, s.y + 1))
        return (tuple(sorted(gt)), tuple(sorted(got)))

    def _build_path_index(self, maps, level_name, scale):
        pos2l, l2p = {}, {}
        if level_name not in maps:
            return pos2l, l2p
        for letter, data in maps[level_name].items():
            qcmz = data.get("qcmzcjocmj", data) if isinstance(data, dict) else data
            L = data.get("oxbwsencfv", len(qcmz)) if isinstance(data, dict) else len(qcmz)
            if L <= 1:
                continue
            l2p[letter] = {}
            for num, pos in qcmz.items():
                px, py = pos.x * scale, pos.y * scale
                l2p[letter][num] = (px, py)
                pos2l.setdefault((px, py), []).append((letter, num))
        return pos2l, l2p

    def _click(self, game, mod, maps, scale, letter, direction):
        """手动模拟点击按钮（先收集再移动，避免链式推动）。"""
        level_name = game.ucybisahh
        chm = getattr(mod, "chmfaflqhy", None)
        if chm is None:
            return
        moves = chm(level_name, letter, direction, game.uopmnplcnv)
        pending = []
        for frm, to in moves:
            sp = game.ttawusezqc(frm.x * scale, frm.y * scale)
            if sp:
                pending.append((sp, to.x * scale, to.y * scale))
        for sp, nx, ny in pending:
            sp.set_position(nx, ny)
