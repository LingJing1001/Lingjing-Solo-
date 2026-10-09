"""EngineSolver — 有引擎代码就读，没有就按原流程。

设计:
  1. 检查 environment_files/{game_id}/ 有没有 .py
  2. 有 → import 模块，提取 maps/按钮/goal，fast BFS 搜解法
  3. 搜出 → 缓存，choose_action 逐步返回动作
  4. 没有/搜不出 → 返回 None，agent 回退原流程

适用: "按钮循环"类游戏（有 maps 数据 + button/goal sprite）
不适用: 无引擎代码（Kaggle 隐藏游戏）→ 回退原流程
"""
from __future__ import annotations

import importlib.util
import os
import sys
from collections import deque
from itertools import product
from pathlib import Path
from typing import Any, Optional


class EngineSolver:
    """有引擎代码 → BFS 搜解法；没有 → None（回退原流程）。"""

    def __init__(self, environments_dir: str = "environment_files"):
        self.environments_dir = environments_dir
        self._solutions: dict[str, Optional[list]] = {}
        self._step_idx: dict[str, int] = {}

    def get_action(self, game_id: str) -> Optional[str]:
        """返回当前步的动作（如 'ACTION6'）或 None（回退原流程）。

        解法是按钮点击序列。每步返回 ACTION6（点击）。
        实际坐标由 agent 的 BubbleClickPlanner 或调用方决定。
        """
        if game_id not in self._solutions:
            self._solutions[game_id] = self._solve(game_id)
            self._step_idx[game_id] = 0

        sol = self._solutions[game_id]
        if not sol:
            return None

        idx = self._step_idx.get(game_id, 0)
        if idx >= len(sol):
            return None  # 解法用完

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
            return None

        # 2. import 游戏模块
        try:
            mod_name = f"_engine_{short_id}"
            spec = importlib.util.spec_from_file_location(mod_name, py_files[0])
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            return None

        # 3. 找 maps 数据（dict of dict of list of list）
        maps_data = self._find_maps(mod)
        if maps_data is None:
            return None  # 不是"按钮循环"类游戏

        # 4. 找缩放因子
        scale = self._find_scale(mod)

        # 5. 创建游戏实例，提取按钮/goal，BFS 搜
        try:
            return self._bfs_solve(game_id, mod, maps_data, scale)
        except Exception:
            return None

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

        # 解析 maps（如果是原始格式，需要 qfvvosdkqr 转换）
        maps = self._normalize_maps(maps_raw, mod)

        # 搜每关解法
        all_actions = []
        for lvl in range(len(getattr(game, "_levels", []) or [])):
            sol = self._solve_one_level(game, mod, maps, scale)
            if sol is None:
                return None  # 某关搜不出 → 整体失败

            # 执行解法（手动模拟）
            for letter, direction in sol:
                self._click(game, mod, maps, scale, letter, direction)
                all_actions.append("ACTION6")  # 点击动作

            if str(game._state).upper() == "WIN":
                break

            # 切换关卡
            if hasattr(game, "set_level"):
                game.set_level(lvl + 1)
            else:
                game.next_level()
            if hasattr(game, "on_set_level"):
                game.on_set_level(game.current_level)

        return all_actions if all_actions else None

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
            return None

        # 提取 goal/goal-o 位置 + 目标
        goals, goal_os = self._get_goals(game)
        targets = self._get_targets(game)
        if not goals or not targets:
            return None

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
