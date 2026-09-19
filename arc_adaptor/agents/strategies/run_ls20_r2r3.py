"""LS20：R2(内部状态感知) → R3(联合状态搜索) 联动 runner。

对应 docs/最小可运行闭环 §3.1/§3.2 的契约：
    R2 出 observation（含 state_hash / legal_actions / game_specific.triple_*）
    R3 出 plan（planner / input_state_hash / candidate_actions / validity / cost）
    动作名到引擎枚举的转换只在这里发生（R3 不返回引擎对象）

R3 分三级，「已验证者优先」（顺序见 R3Planner 的顺序教训）：
    R3-solver    Ls20Solver：当前关卡的离线 BFS 脚本 + 在线单步重规划
    R3-position  纯位置 A*：脚本/在线都给不出动作、且携带三元组已被目标接受时直达目标格
    R3-joint     位置×三元组的 replay BFS（只跑在影子环境上）：先找旋转台/调色台再进目标
                 （去重键是 R2.state_key = 玩家格 × shape × color × rot × 关卡；
                   只按坐标去重会把「同格不同携带块」错误合并）

产物：state/ls20_r2r3_<ts>/
    recording.jsonl  每 tick 的 observation / plan / transition（§5.2）
    manifest.json    证据身份字段（§5.1）
    report.json      验收结论（§5.3），tier=offline_engine，不是 live Scorecard
"""
from __future__ import annotations

import heapq
import json
import os
import pathlib
import sys
import time
from collections import deque
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # arc_adaptor
from paths import (PROJECT_ROOT, add_to_sys_path, environments_dir,  # noqa: E402
                    git_output, state_dir)


def _rel_to_repo(target) -> str:
    """证据文件会被提交：能相对仓库根表示就不写机器绝对路径。"""
    t, root = pathlib.Path(target), pathlib.Path(PROJECT_ROOT)
    for base in (root, root.parent):
        try:
            return str(t.relative_to(base))
        except ValueError:
            continue
    return str(t)


add_to_sys_path()

import numpy as np  # noqa: E402
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402

import evidence_compat as ev  # noqa: E402
import r2_ls20 as r2  # noqa: E402
from lingjing_solo.planning import plan_contract as pc  # noqa: E402
from lingjing_solo.planning.ls20_solver import Ls20Solver, _walkable  # noqa: E402

# 本运行器的 planner 分层（§7.1「引用 state hash」之外的另一半：统计口径不能各写各的）。
# 名字必须与 plan() 里实际吐出的保持一致，validate_plan 会拒未登记的名字。
pc.register_planner("r3_ls20_solver",
                    "已验证的 Ls20Solver：_script_mode 下是离线 BFS 罐头回放，否则是在线单步重规划")
pc.register_planner("r3_layout_wait", "换关布局渲染滞后：ACTION1 原地补一步，等当关布局出现")
pc.register_planner("r3_position_astar", "位置 A*：三元组已被目标接受时直达目标格（r2_ls20 观测驱动）")
pc.register_planner("r3_joint_bfs", "影子引擎上 位置×三元组 联合 BFS（reset+重放，最后兜底）")
pc.register_planner("r3_none", "本 tick 所有 planner 都拿不出动作（决策失败，不是没跑）")

sys.setrecursionlimit(10000)
ACT_BY_NUM = {n: getattr(GameAction, f"ACTION{n}") for n in (1, 2, 3, 4)}
NAME_BY_NUM = {n: f"ACTION{n}" for n in (1, 2, 3, 4)}
NUM_BY_NAME = {v: k for k, v in NAME_BY_NUM.items()}
DELTA = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
OPP = {1: 2, 2: 1, 3: 4, 4: 3}
STEP = 5
LEGAL = list(NAME_BY_NUM.values())


def _legal_names(frame: Any) -> list[str]:
    """frame.available_actions 可能是 int 列表、枚举或 None —— 统一成抽象动作名。"""
    raw = getattr(frame, "available_actions", None) or []
    names: list[str] = []
    for a in raw:
        num = getattr(a, "value", a)
        if isinstance(num, GameAction):
            num = num.value
        try:
            num = int(num)
        except (TypeError, ValueError):
            name = getattr(a, "name", str(a))
            if name in NUM_BY_NAME:
                names.append(name)
            continue
        if num in NAME_BY_NUM:
            names.append(NAME_BY_NUM[num])
    return names or LEGAL


# ─────────────────────────────────────────────────────────────
# R3-a：位置 A*（三元组已匹配时才允许把目标格当可走）
# ─────────────────────────────────────────────────────────────

def r3_position_astar(grid: np.ndarray, obs: dict[str, Any]) -> list[int] | None:
    """位置 A*，把步数预算当**硬约束**：走完就超预算的分支直接不展开。

    预算来自 R2 的引擎真值（steps_left / step_decrement，见 r2_ls20.moves_left）。
    读不到预算时（帧后端、归零闪光帧）退回「不管预算」的原始行为——
    宁可少一条剪枝，也不要因为缺信息就宣布无解。
    """
    spec = obs["game_specific"]
    player = spec.get("state_key")[:2] if spec.get("state_key") else obs["objects_or_features"]["player"]
    goals = [tuple(cell) for cell in obs["objects_or_features"]["goal_cells"]]
    if not player or not goals:
        return None
    start = (int(player[0]), int(player[1]))
    targets = set(goals)
    field = np.asarray(grid, dtype=np.int8)
    goal_walkable = r2.is_goal_ready(obs)
    budget = r2.moves_left(obs)          # 还能做几个动作（None = 无预算信息）

    def h(a: tuple[int, int], b: tuple[int, int]) -> float:
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) / STEP

    open_set: list[tuple[float, float, tuple[int, int]]] = [(h(start, min(targets, key=lambda t: abs(start[0]-t[0]) + abs(start[1]-t[1]))), 0.0, start)]
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score = {start: 0.0}
    while open_set:
        _, cost, cur = heapq.heappop(open_set)
        if cur in targets:
            path: list[int] = []
            while came[cur] is not None:
                prev = came[cur]
                d = (cur[0] - prev[0], cur[1] - prev[1])
                path.append(next(n for n, delta in DELTA.items() if delta[0] * STEP == d[0] and delta[1] * STEP == d[1]))
                cur = prev
            return list(reversed(path))
        for num, (dx, dy) in DELTA.items():
            nxt = (cur[0] + dx * STEP, cur[1] + dy * STEP)
            if not _walkable(field, nxt[0], nxt[1]):
                continue
            if nxt in targets and not goal_walkable:
                continue  # 引擎规则：三元组不匹配时目标格按墙处理，且不扣步
            new_cost = cost + 1
            if budget is not None and new_cost > budget:
                continue            # 走完这步就没步数了 → 这条分支没有意义
            best = min(targets, key=lambda t: abs(nxt[0]-t[0]) + abs(nxt[1]-t[1]))
            if budget is not None and new_cost + h(nxt, best) > budget:
                continue            # 乐观估计也超预算：连「最短可能路线」都走不到目标
            if nxt not in g_score or new_cost < g_score[nxt]:
                g_score[nxt] = new_cost
                came[nxt] = cur
                heapq.heappush(open_set, (new_cost + h(nxt, best), new_cost, nxt))
    return None


# ─────────────────────────────────────────────────────────────
# R3-b：位置×三元组 replay BFS（修饰台在状态空间里是普通转移）
# ─────────────────────────────────────────────────────────────

def _level_index(env: Any) -> int:
    return int(env._game._current_level_index)


def _advanced(env: Any, base_level: int) -> bool:
    g = env._game
    return _level_index(env) > base_level or g._state == GameState.WIN


def _replay(env: Any, path: list[int]) -> None:
    env.reset()
    g = env._game
    for num in path:
        g.perform_action(ActionInput(id=ACT_BY_NUM[num], data={}, reasoning=None), raw=True)


def _budget_of(game: Any) -> tuple[int | None, int | None, bool]:
    """从引擎读 (剩余步, 剩余生命, 是否重开闪光帧)；拿不到就返回 None 让调用方不加剪枝。"""
    counter = getattr(game, "_step_counter_ui", None)
    steps = getattr(counter, "current_steps", None) if counter is not None else None
    lives = getattr(game, "aqygnziho", None)
    return (int(steps) if steps is not None else None,
            int(lives) if lives is not None else None,
            bool(getattr(game, "ebfuxzbvn", False)))


def r3_joint_bfs(shadow: Any, history: list[int], *, max_nodes: int = 400,
                 max_depth: int = 8, t_limit: float = 30.0,
                 stats: dict[str, int] | None = None) -> list[int] | None:
    """位置×三元组的 replay BFS，且**带步数预算约束** —— 只在影子环境里跑，绝不碰主环境。

    引擎的 reset() 回到 L1，所以搜索必须把「开局到当前的完整历史」一起重放，
    否则第 2 关起算出来的路径不是相对当前状态的（见 run_ls20_r234.py 只解 L1 的教训）。
    代价是每个节点 O(|history|+depth) 次 perform_action，故只作为兜底、预算给得很小。

    预算剪枝（2026-09-19）：剩余步、生命、重开闪光帧都直接从影子引擎读，
    所以「踩 npxgalaybz 补给格 → 步数条复位」这条规则**不需要手写模型**，
    引擎自己就把预算加回来了；反过来，一段会把步数走到 0（=本关重开、掉一条命）
    的分支会被当场砍掉，不会占着 visited 名额。
    """
    _replay(shadow, history)
    base = _level_index(shadow)
    base_steps, base_lives, _ = _budget_of(shadow._game)
    start_key = r2.state_key(shadow._game)
    visited = {start_key}
    queue: deque[list[int]] = deque([[]])
    nodes = pruned = 0
    started = time.monotonic()
    while queue:
        if time.monotonic() - started > t_limit or nodes > max_nodes:
            if stats is not None:
                stats["pruned"] = pruned
            return None
        path = queue.popleft()
        for num in (1, 2, 3, 4):
            if path and num == OPP[path[-1]]:
                continue  # 立即反向 = 无操作，LS20 无重力，走回来只会浪费步数
            if len(path) + 1 > max_depth:
                continue
            _replay(shadow, history + path + [num])
            nodes += 1
            g = shadow._game
            if _advanced(shadow, base):
                if stats is not None:
                    stats["pruned"] = pruned
                return path + [num]
            if g._state == GameState.GAME_OVER:
                continue
            steps, lives, flash = _budget_of(g)
            if flash or (steps is not None and steps <= 0) \
                    or (base_lives is not None and lives is not None and lives < base_lives):
                pruned += 1
                continue            # 这段后缀把步数花光了：引擎会重开本关，感知当帧作废
            key = r2.state_key(g)
            if key in visited:
                continue
            visited.add(key)
            queue.append(path + [num])
    if stats is not None:
                stats["pruned"] = pruned
    return None


# ─────────────────────────────────────────────────────────────
# R3 统一出口：返回 §3.2 的 plan 结构（只给抽象动作名）
# ─────────────────────────────────────────────────────────────

class R3Planner:
    """三级 R3，「已验证者优先」；产出的 plan 只含抽象动作名（引擎转换留在 runner）。

    1) r3_ls20_solver     量规内已验证的 Ls20Solver（离线 BFS 脚本 + 在线单步重规划）
                          —— 这一级是 LS20 7/7 的基线，必须优先，不能被下面的搜索抢走
    2) r3_position_astar  脚本/在线规划都拿不出动作，且当前三元组已被目标接受 → 直达目标格
    3) r3_joint_bfs       最后兜底：影子环境上的 位置×三元组 replay BFS

    顺序教训（2026-09-19 首跑）：把 A* 放第一级时，它会在 Ls20Solver 已排好
    当前关卡脚本动作时抢决策，L3 出现 (14,40)↔(14,45) 来回 20 步，最后 r3_none 停住。
    """

    def __init__(self, arcade: Any, gid: str, budget: dict[str, Any]):
        self.arcade = arcade
        self.gid = gid
        self.budget = budget
        self.shadow: Any = None
        self.history: list[int] = []          # 从开局起到当前状态的完整动作序列
        self.solver = Ls20Solver()
        self._last_grid: np.ndarray | None = None
        self._last_action: str | None = None
        self.notes: list[str] = []
        self.last_joint_stats: dict[str, int] = {}

    def observe(self, grid: np.ndarray, levels: int, action: str | None = None) -> None:
        """每 tick 先把 (上一帧, 本帧, 其间的动作) 喂给内部规划器，再谈计划。"""
        self.solver.observe(self._last_grid, grid, self._last_action, levels)
        self._last_grid = grid
        self._last_action = action

    def note(self, num: int) -> None:
        self.history.append(num)

    # ── 各级 planner ─────────────────────────────────────────
    def _solver_action(self, grid: np.ndarray) -> tuple[str, int] | None:
        """已验证规划器 + 生产驱动层(lingjing_solo/agent.py)同样的两处补救。"""
        act = self.solver.plan(LEGAL)
        if act is None and not self.solver._await_level_layout \
                and not self.solver._layout_wait and not self.solver.active:
            self.solver.reset_level(grid)          # agent.py:234 同路径
            act = self.solver.plan(LEGAL)
        if act is None and self.solver._layout_wait and "ACTION1" in LEGAL:
            return "r3_layout_wait", 1            # 换关渲染滞后：原地补一步追上
        if act and act in NUM_BY_NAME:
            return "r3_ls20_solver", NUM_BY_NAME[act]
        return None

    def _joint_path(self) -> list[int]:
        if self.shadow is None:
            self.shadow = self.arcade.make(self.gid)
        stats: dict[str, int] = {}
        self.last_joint_stats = stats
        return r3_joint_bfs(self.shadow, list(self.history),
                            max_nodes=self.budget["max_nodes"],
                            max_depth=self.budget["max_depth"],
                            t_limit=self.budget["time_s"], stats=stats) or []

    # ── 统一出口 ─────────────────────────────────────────────
    def plan(self, obs: dict[str, Any], grid: np.ndarray) -> dict[str, Any]:
        started = time.monotonic()
        spec = obs["game_specific"]
        self.last_joint_stats = {}        # 没跑到 joint 就不该带上一次的剪枝数
        planner, path = "r3_none", []
        picked = self._solver_action(grid)
        if picked:
            planner, path = picked[0], [picked[1]]
        if not path and r2.is_goal_ready(obs):
            astar = r3_position_astar(grid, obs)
            if astar:
                planner, path = "r3_position_astar", astar
                self.notes.append(f"{obs['observation_id']}: A* 兜底接管 "
                                  f"({self.solver._phase})")
        if not path and not self.solver._layout_wait and not self.solver._await_level_layout:
            path = self._joint_path()
            if path:
                planner = "r3_joint_bfs"
                self.notes.append(f"{obs['observation_id']}: replay BFS 兜底接管")
        # validity 只声明能站得住的话：罐头解是离线跑出来的（`verified_offline`，附来源），
        # 在线单步重规划和 A*/BFS 搜索结果只是「当场算出来的候选」，等 `_verify_and_return`
        # 那种回放验证过了才升级——§10 禁止把没验过的说成验过。
        replayed_from_script = planner == "r3_ls20_solver" and bool(self.solver._script_mode)
        evidence_refs = [obs["observation_id"]]
        if replayed_from_script:
            evidence_refs.append(f"script_bank:ls20:L{self.solver.levels_seen}")
        plan = pc.build_plan(
            planner=planner,
            input_state_hash=obs["state_hash"],
            candidate_actions=[NAME_BY_NUM[n] for n in path],
            legal_actions=obs["legal_actions"],
            # 目标分解产物化（设计文档 §7.5 第二条缺口）：不再是常量占位，
            # 而是"这 tick 还差什么"——剩余目标 + 当前修饰台任务。
            expected_goal=(f"{planner}:remaining_goals={len(self.solver.goals) - self.solver.goal_idx}"
                           f":mod_task={self.solver.mod_kind if self.solver.mod_tasks else 'none'}"),
            cost=len(path) if path else None,
            search_budget=self.budget,
            # 预算约束的现场：还剩几个动作、还剩几条命、搜索为此砍掉了多少分支
            validity="verified_offline" if replayed_from_script
            else ("candidate" if path else "none"),
            evidence_refs=evidence_refs,
            search_seconds=round(time.monotonic() - started, 2),
            budget_moves=r2.moves_left(obs),
            steps_left=spec.get("steps_left"),
            lives=spec.get("lives"),
            joint_pruned_branches=self.last_joint_stats.get("pruned"),
            solver_phase=getattr(self.solver, "_phase", None),
        )
        return plan


# ─────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────

def _install_color_residue_fix() -> None:
    """诊断用打桩（不动仓库里的 solver）：给 _find_color_pad 补上 rot/shape 都有的可达性过滤。

    玩家一步走 5 像素，(x%5, y%5) 这个残差类在关内永不改变，所以只有和玩家同残差的
    台子才踩得到。_find_rot_pad / _find_shape_pad 都按 _reachable_from* 过滤了，
    _find_color_pad 只按曼哈顿距离取最近（ls20_solver.py:228-229），于是会定一个
    永远走不到的颜色台 —— _next_leg_cost 变 None，_refill_detour 又以「leg 未知」
    为由不绕路吃补给（ls20_solver.py:943），最后步数条耗尽卡死。见
    probe_ls20_g2_refill.py 与 probe_ls20_pad_residue.py 的逐 tick 证据。
    """
    import lingjing_solo.planning.ls20_solver as m

    def find_color_pad(grid, start=None):
        cands = [(x, y) for y in range(58) for x in range(58)
                 if {9, 14, 0, 8} <= set(int(v) for v in grid[y:y + 5, x:x + 5].flatten())
                 and m._walkable(grid, x, y)]
        if not cands:
            return None
        if start:
            reach = m._reachable_from(grid, start)
            # 与原实现唯一的差别：先在「踩得到」的候选里取最近，全都不通才退回原逻辑
            usable = [c for c in cands if c in reach]
            if usable:
                return min(usable, key=lambda t: abs(t[0] - start[0]) + abs(t[1] - start[1]))
        return min(cands, key=lambda t: abs(t[0] - start[0]) + abs(t[1] - start[1])) if start else cands[0]

    m._find_color_pad = find_color_pad


def main() -> int:
    argv = sys.argv[1:]
    no_scripts = "--no-scripts" in argv
    fix_color = "--fix-color-residue" in argv      # 仅诊断：验证「残差过滤」这一处的收益
    max_ticks = next((int(a) for a in argv if not a.startswith("--")), 200)
    budget = {"time_s": float(os.environ.get("LS20_R3_T_LIMIT", 30)),
              "max_nodes": int(os.environ.get("LS20_R3_MAX_NODES", 400)),
              "max_depth": int(os.environ.get("LS20_R3_MAX_DEPTH", 8))}
    if no_scripts:
        # G2 闸门（docs/统一R3搜索策略设计.md §4）：关掉离线罐头解，逼 planner 只靠搜索。
        # 补丁点必须是 ls20_solver 命名空间——它是 `from .script_bank import script_for_level`。
        import lingjing_solo.planning.ls20_solver as _solver_mod
        _solver_mod.script_for_level = lambda *a, **k: None
    if fix_color:
        _install_color_residue_fix()
    run_dir = state_dir() / time.strftime(
        "ls20_r2r3" + ("_noscript" if no_scripts else "")
        + ("_fixcolor" if fix_color else "") + "_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    recording = (run_dir / "recording.jsonl").open("w", encoding="utf-8")

    commit, branch = git_output("rev-parse", "HEAD"), git_output("branch", "--show-current")
    run_id = (f"{time.strftime('%Y-%m-%dT%H:%M:%S')}-ls20-r2r3"
              + ("-g2-no-scripts" if no_scripts else "")
              + ("-fix-color-residue" if fix_color else ""))
    manifest = ev.build_manifest(
        run_id=run_id, game_id="ls20", branch=branch, commit=commit,
        module_versions={"r2": r2.OBSERVATION_SCHEMA,
                         "r3": "ls20_solver+position_astar+joint_bfs",
                         "r4": "none", "evidence": ev.SCHEMA_VERSION},
        evidence_tier="offline_engine", mode="execute", seed=None,
        limits={"max_ticks": max_ticks, **budget, "scripts_disabled": no_scripts,
                "color_residue_fix": fix_color},
        source_recording=None,
        # 证据文件会被提交，里面只存相对仓库根的路径，不存机器绝对路径
        game_specific={"environments_dir": _rel_to_repo(environments_dir("ls20")),
                       "protocol_backend": ev.BACKEND,
                       "arc_commit": git_output("rev-parse", "HEAD",
                                                cwd=pathlib.Path(environments_dir("ls20")).parent)},
    )
    manifest["artifacts"] = ["recording.jsonl", "report.json"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    arcade = Arcade(environments_dir=environments_dir("ls20"), operation_mode=OperationMode.OFFLINE)
    gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ls20")][0]
    env = arcade.make(gid)
    frame = env.reset()
    grid = r2.to_grid(frame)
    obs = r2.observe_state(env._game, grid, game_id=gid, tick=0, legal_actions=_legal_names(frame),
                           run_id=run_id, episode_id=gid)
    print(f"game_id={gid}  run_dir={run_dir}")
    print(f"R2 reset: 玩家={obs['objects_or_features']['player']} 目标={obs['objects_or_features']['goal_cells']} "
          f"当前三元组={obs['game_specific']['triple_current']} 要求={obs['game_specific']['triple_required']} "
          f"剩余步={obs['game_specific']['steps_left']}")
    print(f"{'tick':>4} {'动作':<9} {'planner':<19} {'玩家':<10} {'lvl':>3} state")

    prev_obs, actions_taken, levels_seen = obs, 0, int(obs["levels_completed"])
    # 基线行：protocol.replay_recording 要求第一行的 requested_action 是 RESET（或缺省）
    baseline = ev.build_tick(
        run_id=run_id, episode_id=gid, tick=0, frame=grid.tolist(),
        requested_action={"name": "RESET", "payload": {}}, settled_frame=True,
        state="RESET", levels_completed=levels_seen, score=None,
        legal_actions=obs["legal_actions"], state_hash=obs["state_hash"],
        evidence_refs=[obs["observation_id"]],
        game_specific={"triple_current": obs["game_specific"]["triple_current"],
                       "triple_required": obs["game_specific"]["triple_required"]})
    recording.write(json.dumps({"data": baseline, "observation": _slim(obs)},
                               ensure_ascii=False) + "\n")
    recording.flush()
    planner = R3Planner(arcade, gid, budget)
    planner.observe(grid, levels_seen)
    actions_legal = True
    # §7.5「planner 是否真的接管」要能被机器读到：只写"跑通了几个关卡"看不出是哪层在决策。
    plan_tally: dict[str, int] = {}
    validity_tally: dict[str, int] = {}
    hash_linked = True
    verdict = "FAIL"
    for tick in range(1, max_ticks + 1):
        try:
            plan = planner.plan(prev_obs, grid)
        except pc.PlanContractError as exc:
            # 契约不满足 ≠ 可以继续跑：宁可 BLOCKED，也不写一份"plan 没自检过"的证据。
            planner.notes.append(f"tick {tick}: plan 契约校验失败 -> {exc}")
            print(f"{tick:>4} {'-':<9} {'契约失败':<19} {exc} → 停")
            verdict = "BLOCKED"
            break
        if not plan["candidate_actions"]:
            print(f"{tick:>4} {'-':<9} {plan['planner'] + ' 无解':<19} 计划失败 → 停")
            verdict = "BLOCKED"
            break
        num = NUM_BY_NAME[plan["candidate_actions"][0]]
        action_name = NAME_BY_NUM[num]
        plan_tally[plan["planner"]] = plan_tally.get(plan["planner"], 0) + 1
        validity_tally[plan["validity"]] = validity_tally.get(plan["validity"], 0) + 1
        # 「引用 state hash」不能只靠字段存在：它必须就是上一 tick 真被记录过的那个状态哈希，
        # 否则 plan 与观测之间没有任何绑定，九字段就成了自说自话。
        hash_linked &= prev_obs["state_hash"] == plan["input_state_hash"]
        actions_legal &= action_name in prev_obs["legal_actions"]
        frame = env.step(ACT_BY_NUM[num], data=None)
        grid = r2.to_grid(frame)
        obs = r2.observe_state(env._game, grid, game_id=gid, tick=tick,
                               legal_actions=_legal_names(frame),
                               run_id=run_id, episode_id=gid)
        trans = r2.transition(prev_obs, action_name, obs)
        actions_taken += 1
        planner.note(num)
        planner.observe(grid, int(obs["levels_completed"]), action=action_name)
        tick_rec = ev.build_tick(
            run_id=run_id, episode_id=gid, tick=tick, frame=grid.tolist(),
            requested_action={"name": action_name, "id": num, "payload": {}},
            settled_frame=True, state=obs["state"],
            levels_completed=int(obs["levels_completed"]), score=None,
            legal_actions=obs["legal_actions"], state_hash=obs["state_hash"],
            plan_id=plan["plan_id"], plan=plan, evidence_refs=[prev_obs["observation_id"]],
            game_specific={"triple_current": obs["game_specific"]["triple_current"],
                           "triple_required": obs["game_specific"]["triple_required"],
                           "steps_left": obs["game_specific"]["steps_left"],
                           "state_key": obs["game_specific"]["state_key"]})
        record = {"data": tick_rec, "observation": _slim(prev_obs), "plan": plan,
                  "transition": trans}
        recording.write(json.dumps(record, ensure_ascii=False) + "\n")
        recording.flush()
        print(f"{tick:>4} {action_name:<9} {plan['planner']:<19} "
              f"{str(obs['objects_or_features']['player']):<10} {obs['levels_completed']:>3} {obs['state']}"
              f"{'  ←过关' if trans['levels_delta'] else ''}")
        levels_seen = max(levels_seen, int(obs["levels_completed"]))
        prev_obs = obs
        if obs["state"] in ("WIN", "GAME_OVER"):
            verdict = "PASS" if obs["state"] == "WIN" else "FAIL"
            print(f"\n终态 {obs['state']} levels={obs['levels_completed']}")
            break
    recording.close()

    # ── 自证：把刚写的 recording 交给协议层回读，判据不许手填 True ──────────
    replay_info: dict[str, Any] = {}
    replay_ok = False
    try:
        res = ev.replay_recording(run_dir / "recording.jsonl", legal_actions=LEGAL)
        replay_ok = len(res.transitions) == actions_taken
        replay_info = {"transitions": len(res.transitions), "reset_count": res.reset_count,
                       "final_state": res.final_state,
                       "levels_completed": res.levels_completed}
    except ev.EvidenceValidationError as exc:
        replay_info = {"error": str(exc)}

    schema_ok, schema_err = True, ""
    with (run_dir / "recording.jsonl").open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            try:
                ev.validate_tick(json.loads(line)["data"])
            except (ev.EvidenceValidationError, KeyError, json.JSONDecodeError) as exc:
                schema_ok, schema_err = False, f"line {line_number}: {exc}"
                break

    if verdict in ("PASS", "FAIL") and not (replay_ok and schema_ok and actions_legal and hash_linked):
        verdict = "BLOCKED"          # §10：证据不自洽就当没通过，不做替换
    report = ev.build_verification_report(
        run_id=run_id, tier="offline_engine", verdict=verdict,
        criteria={"schema_valid": schema_ok, "actions_legal": actions_legal,
                  "replay_complete": replay_ok,
                  "plan_hash_linked": hash_linked,
                  "terminal_verified": verdict in ("PASS", "FAIL"),
                  "levels_completed": levels_seen},
        # 引擎的 _current_level_index 是「当前关卡下标」；最后一关通关时它不再 +1，
        # 所以 WIN 时的实际通关数 = levels_seen + 1，两个量都写清楚，别让读的人猜。
        metrics={"actions": actions_taken, "scripts_disabled": no_scripts,
                 "level_index": levels_seen,
                 "plans_by_planner": plan_tally, "plans_by_validity": validity_tally,
                 "levels_cleared": levels_seen + 1 if verdict == "PASS" else levels_seen,
                 "levels_completed": levels_seen},
        evidence_refs=["recording.jsonl", "manifest.json"],
        limitations=(["G2 闸门：已关掉离线罐头解，关数不代表 LS20 基线成绩"]
                     if no_scripts else [])
        + ["R4/R5 未接入", "非 live Scorecard",
                     "r3_joint_bfs 用 reset+重放（每个节点重放整段历史），仅作兜底",
                     "r3_ls20_solver 复用已验证的 Ls20Solver：含离线脚本关卡时它是回放而非搜索"]
        + ([] if schema_ok else [f"tick schema: {schema_err}"]) + planner.notes,
        game_specific={"r2_schema": r2.OBSERVATION_SCHEMA, "protocol_backend": ev.BACKEND,
                       "replay": replay_info},
    )
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["status"] = {"PASS": "completed", "FAIL": "completed",
                          "BLOCKED": "blocked"}[verdict]
    manifest["artifacts"] = ["recording.jsonl", "report.json"]
    manifest["per_tick_recording"] = {
        "path": "recording.jsonl", "ticks": actions_taken + 1,
        "format": "lingjing-evidence-v1 build_tick（含基线行），每行包在 data 字段里",
        "backend": ev.BACKEND}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\nverdict={verdict} actions={actions_taken} levels={levels_seen}")
    print(f"plans={dict(sorted(plan_tally.items()))} validity={dict(sorted(validity_tally.items()))} "
          f"hash_linked={hash_linked}")
    print(f"RESULT_DIR={run_dir}")
    return 0 if verdict == "PASS" else 1


def _slim(obs: dict[str, Any]) -> dict[str, Any]:
    """recording 里不重复塞邻域像素，留关键字段就够重建。"""
    copy = dict(obs)
    copy["objects_or_features"] = {k: v for k, v in obs["objects_or_features"].items() if k != "goals"}
    return copy


if __name__ == "__main__":
    raise SystemExit(main())
