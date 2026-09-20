"""R2+R3+R4 组合跑 AR25 闯关 (v5: 增强搜索预算 + 已知解法优先 + 双轴穷举)。

R2 感知: 从游戏内部状态提取轴/拼块/目标/覆盖信息
R3 搜索: 目标分解 → beam search → 状态空间搜索
R4 剪枝: 动作效果反馈 + 覆盖进度 + 反循环 (贪心回退)

策略 (v5 改进):
  0. 已知解法优先 (arc_shadow.KNOWN_SOLUTIONS, 秒级)
  1. R2 感知关卡结构 → 决定搜索策略
  2. R3 目标分解 (单轴关卡, 秒级)
  3. R3 双轴目标分解 (双轴关卡, 含穷举拼块排列)
  4. R3 beam search (中复杂度, L6+ 加宽beam加长预算)
  5. R3 arc_shadow (L6+, 600s/2M节点)
  6. R4 贪心步进 (最终回退, 逐动作评分)

预算策略:
  L1-L5: 总60s (目标分解通常秒级)
  L6:    总900s (双轴+复杂, beam 300s + arc_shadow 600s)
  L7:    总900s (双轴+旋转块, 同上)
  L8:    总900s (最高难度)

R3 出口（设计文档 §8.8 / 团队规范 §3.2）:
  各 planner 一律返回 abstract action name（"ACTION1".."ACTION5"）序列，
  name → GameAction/ActionInput 的转换只发生在下面的边界块；
  `r234_solve` 返回 `(path, method, plan)`，plan 是当场过 `validate_plan` 的九字段产物，
  每关一份，随 `result.json` 的 `levels[i]["plan"]` 落盘。
"""
import sys, os, time, heapq, itertools, json, hashlib, pathlib, contextlib
os.environ.setdefault("MPLBACKEND", "Agg")
sys.stdout.reconfigure(encoding='utf-8')
# 用自身位置推导 arc_adaptor，绝不写死成员本地 checkout 路径（§3.2）
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from paths import (PROJECT_ROOT, add_to_sys_path, environments_dir,  # noqa: E402
                   state_dir, git_output)

add_to_sys_path()

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput, GameState
from arc_shadow import _snapshot, _restore, _state_key, _heuristic, KNOWN_SOLUTIONS
from lingjing_solo.planning import plan_contract as pc  # noqa: E402
import evidence_compat as ev                            # noqa: E402  §5.1/§5.2/§5.3 schema
import r2_ar25                                          # noqa: E402  §3.1 观测 + 正式状态哈希
from tick_trail import TickTrail                        # noqa: E402  §5.2 逐 tick 写入器

# ═══════════════════════════════════════════════════════════
# §3.2 边界：planner 只认识 abstract action name（"ACTIONn" 字符串），
# name → GameAction/ActionInput 的转换只发生在本块（团队规范 §3.2 兼容要求第一条 + 规则 5）。
# `tests/test_abstract_action_boundary.py` 的 AST 闸门只扫 `lingjing_solo/**`，管不到本文件，
# 所以这里的收口靠"除本块之外不得出现 ActionInput(...)"这一条约定 + 8.8 的回归测试。
# ═══════════════════════════════════════════════════════════

#: 仅供 arc_shadow 的 int 键 API（`_num2act` / `_act2num` / `_ACTS`）使用，新代码别拿它转枚举。
ACT_MAP = {n: getattr(GameAction, "ACTION%d" % n) for n in range(1, 8)}
NAME_BY_NUM = {n: "ACTION%d" % n for n in range(1, 8)}
ENUM_BY_NAME = {NAME_BY_NUM[n]: ACT_MAP[n] for n in range(1, 8)}
#: AR25 只用得到这五个，实测 `is_complex()` 全为 False（只有 ACTION6 复杂、需要 x/y payload），
#: 所以设计文档 §8.4 那条"复杂动作的 payload 会不会被 plan 丢掉"在本游戏不适用。
AR25_ACTION_NAMES = tuple(NAME_BY_NUM[n] for n in (1, 2, 3, 4, 5))
#: name → 棋盘位移，供 R4 评分；ACTION5=TOGGLE 只切换选中对象、不平移，故位移为 0。
ACTION_DELTA = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0),
                "ACTION4": (1, 0), "ACTION5": (0, 0)}

def to_enum(name):
    """abstract name → `GameAction`：全模块唯一转换点，未知名字抛错而不是降级放行。"""
    try:
        return ENUM_BY_NAME[name]
    except KeyError:
        raise pc.PlanContractError(f"未知 abstract action name: {name!r}") from None

def act_input(name):
    """abstract name → `ActionInput`；所有 `perform_action` 调用都从这里出来。"""
    return ActionInput(id=to_enum(name), data={}, reasoning=None)

def run_names(g, names, at=None):
    """在真引擎上按序执行 abstract name 序列。

    给了 `at`（一个 `_attempt()` 段）就逐动作登记成 §5.2 证据；不给则是纯回放。
    """
    for a in names:
        if at is None:
            g.perform_action(act_input(a), raw=True)
        else:
            at.step(g, a)

def nums_to_names(seq):
    """历史 int 序列（`KNOWN_SOLUTIONS`、`arc_shadow.solve_level` 的返回值）→ name 列表。"""
    return [NAME_BY_NUM[int(n)] for n in seq]

# ═══════════════════════════════════════════════════════════
# §5.2 逐 tick 证据：只有"真留在引擎上的动作"才写行
# ═══════════════════════════════════════════════════════════
#
# AR25 的引擎推进分散在四处（罐头解法、目标分解校验、R4 贪心、主循环补放），另外三层
# （beam / 状态搜索 / arc_shadow）在 finally 里 `_restore` 回入口帧。前者才是"环境发生过"，
# 后者只是"搜索尝试过"——按 §10 不能混写，所以证据只在 `_Attempt` 提交后落盘。

#: 逐 tick 写入器；None = 只跑关不采证（被 import 当库用、单测直接点 planner 时的默认）。
_TRAIL: TickTrail | None = None
#: 本轮 run 的身份，`main()` 建 trail 时一并设好，供 observation / evidence_refs 引用。
_RUN_ID = ""
_EPISODE_ID = ""

def _engine_frame(g):
    """§5.2 每行的 `frame`：现读引擎 21×21 渲染网格。拿不到就抛错，绝不填占位帧。"""
    return g.naxbskjmlg().tolist()

def _observe(g, perc=None, *, tick=None):
    """感知 → `r2_ar25.enrich` → §3.1 observation；`tick` 缺省用 trail 的下一个行号。"""
    if perc is None:
        perc = r2_perceive(g)
    r2_ar25.enrich(g, perc)
    if tick is None:
        tick = _TRAIL.next_tick if _TRAIL is not None else 0
    return r2_ar25.observe(perc, tick=tick, frame=_engine_frame(g), run_id=_RUN_ID,
                           episode_id=_EPISODE_ID, legal_actions=list(AR25_ACTION_NAMES))

class _Attempt:
    """一段"打上去、可能要整段回滚"的引擎推进：commit 才进 recording，abort 一行不留。"""

    def __init__(self):
        self.active = _TRAIL is not None
        self.done = not self.active          # 不采证时视为已结清，contextmanager 不用收尾
        if self.active:
            _TRAIL.begin_attempt()

    def step(self, g, name):
        """执行一个动作；采证时 before 取 trail 的当前终态，after 从动作后的引擎重采。"""
        before = _TRAIL.last_obs if self.active else None
        g.perform_action(act_input(name), raw=True)
        if self.active:
            _TRAIL.record(before, name, _observe(g), _engine_frame(g))

    def commit(self):
        """引擎确实前进到了这里（关卡被解开/动作没回滚）：整段转入本关缓冲。"""
        if self.active:
            _TRAIL.commit_attempt()
        self.done = True

    def abort(self):
        """引擎已 `_restore` 回入口帧：整段丢弃，`last_obs` 跟着回到尝试之前。"""
        if self.active:
            _TRAIL.abort_attempt()
        self.done = True

@contextlib.contextmanager
def _attempt():
    """`with _attempt() as at:` —— 出口没 settle 就按 abort 处理（宁可少写，不虚报）。"""
    at = _Attempt()
    try:
        yield at
    finally:
        if not at.done:
            at.abort()

def _flush_level(plan):
    """本关已提交的转移连同"整关一份"的 plan 落盘；不采证时是空操作。"""
    if _TRAIL is None:
        return 0
    _TRAIL.attach_plan(plan)
    return _TRAIL.flush_level()

def _rel_to_repo(target):
    """证据文件会被提交：能相对仓库根表示就不写机器绝对路径（同 `run_ls20_r2r3.py:37`）。"""
    t = pathlib.Path(target)
    for base in (pathlib.Path(PROJECT_ROOT), pathlib.Path(PROJECT_ROOT).parent):
        try:
            return str(t.relative_to(base))
        except ValueError:
            continue
    return str(t)

H_AXIS_TAG = "0002nuguepuujf"
V_AXIS_TAG = "0054kgxrvfihgm"
BOARD = 21
#: 目标分解的组合数上限：单轴 = 固定轴位后拼块组合上限（超过则每块截断 top-30）；
#: 双轴 = `_dual_exhaustive_pick` 的 max_combos。plan 里的 `search_budget` 报的就是这两个数。
GOAL_DECOMB_COMBO_CAP = 500000
DUAL_COMBO_CAP = 100000

# 本运行器的 planner 分层。名字必须登记进注册表，`validate_plan` ⑤ 会拒未登记的名字
# （§7.5 命名漂移条：各运行器自造 planner 名会让统计口径无法对账）。
pc.register_planner("ar25_goal_decomp",
                    "R3 单轴目标分解：枚举 轴位×各拼块平移位 组合，取全覆盖最小成本路径")
pc.register_planner("ar25_goal_decomp_dual",
                    "R3 双轴目标分解：枚举 (h_y,v_x)×拼块组合，覆盖判定按引擎反射级联闭包")
pc.register_planner("ar25_shadow_state_search",
                    "R3 arc_shadow 贪心最佳优先（L6+ 兜底；int 出口在调用点转 abstract name）")
pc.register_planner("r4_greedy_step",
                    "R4 贪心步进：动作效果反馈 + 覆盖进度 + 反循环，逐动作评分（最终回退）")

#: 历史 method 串 → 注册表 planner 名。保留 method 串本身，是为了让既有
#: `state/*/report.json` 的分布还能跟新数据对账（§8.6 风险三）。
AR25_METHOD_TO_PLANNER = {
    "R0-已知解法": "t0_known_solution",
    "R3-目标分解": "ar25_goal_decomp",
    "R3-双轴分解": "ar25_goal_decomp_dual",
    "R3-beam": "beam",
    "R3-arc_shadow": "ar25_shadow_state_search",
    "R3-搜索": "state_search",
    "R4-贪心": "r4_greedy_step",
}

def is_won(g):
    w = g.vplrhaovhr()
    return w is True or (hasattr(w, '__len__') and len(w) == 1 and bool(w))

def get_cells(obj):
    h, w = obj.pixels.shape
    return [(j, i) for i in range(h) for j in range(w) if obj.pixels[i, j] != -1]

# ═══════════════════════════════════════════════════════════
# R2: 感知 (从内部状态提取结构)
# ═══════════════════════════════════════════════════════════

def r2_perceive(g):
    """从游戏内部状态提取完整关卡信息。"""
    axis_ids = set(id(a) for a in g.jtkyjqznbnp)
    axes = []
    for ax in g.jtkyjqznbnp:
        tags = list(ax.tags) if hasattr(ax, "tags") else []
        atype = 'h' if H_AXIS_TAG in tags else ('v' if V_AXIS_TAG in tags else '?')
        axes.append({"x": int(ax.x), "y": int(ax.y), "type": atype, "cells": get_cells(ax)})
    switch_order = []
    for idx, obj in enumerate(g.ayyvxqrhnzw):
        is_axis = id(obj) in axis_ids
        switch_order.append({"idx": idx, "is_axis": is_axis,
                             "x": int(obj.x), "y": int(obj.y),
                             "cells": get_cells(obj) if not is_axis else None})
    sel_idx = g.ayyvxqrhnzw.index(g.yvifanjrcyu) if g.yvifanjrcyu in g.ayyvxqrhnzw else 0
    targets = {(int(t.x), int(t.y)) for t in g.fswikrcrdmx}
    budget = int(g.lelsvjlwneo.ilqnjlrnkk)
    steps_left = int(g.lelsvjlwneo.current_steps)
    grid = g.naxbskjmlg()
    uncovered = [(int(t.x), int(t.y)) for t in g.fswikrcrdmx if grid[t.y, t.x] < 0]
    movable = [s for s in switch_order if not s["is_axis"]]
    n_axes = len(axes)
    atype = axes[0]["type"] if n_axes == 1 else '?'
    return {
        "axes": axes, "switch_order": switch_order, "sel_idx": sel_idx,
        "targets": targets, "budget": budget, "steps_left": steps_left,
        "n_switch": len(switch_order), "movable": movable,
        "n_axes": n_axes, "atype": atype,
        "covered": len(targets) - len(uncovered), "total_targets": len(targets),
        "uncovered": uncovered, "won": is_won(g),
        "level": int(g._current_level_index),
        "state_key": _state_key(g),
    }

# ═══════════════════════════════════════════════════════════
# §3.2 plan 出口：把 R3/R4 的产出包成九字段契约
# ═══════════════════════════════════════════════════════════

def ar25_state_hash(perc):
    """输入状态哈希（§3.2 字段③：plan 必须说明"相对哪个局面成立"）。

    ② 的临时实现是 `sha256(repr(_state_key(g)))` 前 16 位——那是**搜索去重键**，不是状态身份：
    `arc_shadow._snapshot` 把旋转距离和三个标志位当可变全局状态保存，`_state_key` 却没收它们。
    ③ 起改用 `r2_ar25.state_hash`：逐字段命名、排序确定、不含步数条（理由见该模块 docstring）。
    旧搜索键哈希只留在观测的 `game_specific.engine_key_hash` 里作**对照**，由 `tick_trail`
    在真实跑测里双向审计「新哈希是否比搜索键粗」，结论进 `report.json`。

    `perc` 必须经 `r2_ar25.enrich(g, perc)`：没 enrich 时 `state_hash` 抛 `R2PerceptionError`，
    而不是静默算出一个少收了一类可变状态的哈希。
    """
    return r2_ar25.state_hash(perc)

def ar25_axis_types(perc):
    """plan 里报的轴构成：直接从 `axes` 汇总。

    没有用 `perc["atype"]`——那个字段按 R2 的老约定只在**单轴**关卡填，双轴关卡一律 '?'
    （`r2_perceive` 里 `axes[0]["type"] if n_axes == 1 else '?'`）。② 是"只改出口不动 R2"，
    所以这里自己算，避免 plan 的 `expected_goal` 写成 `axes=2(?)` 这种读起来像 bug 的串。
    """
    ts = sorted({a["type"] for a in perc["axes"]} - {'?'})
    return "+".join(ts) if ts else '?'

def ar25_expected_goal(perc):
    """目标分解的产物，不是常量占位：这关要「全覆盖」，并带上输入态的进度与结构。"""
    return (f"cover_all_targets:{perc['covered']}->{perc['total_targets']}"
            f";axes={perc['n_axes']}({ar25_axis_types(perc)})"
            f";movable={len(perc['movable'])}"
            f";steps_left={perc['steps_left']}/{perc['budget']}")

def ar25_subgoals(perc):
    """把「全覆盖」拆成逐目标点：subgoals 是目标分解的中间产物，不是装饰字段。

    输入态里已覆盖的目标不必再分解；极端情况下（全已覆盖）退回完整目标集，
    保证 `subgoals` 不会因为关卡结构而变成空列表。
    """
    pts = perc["uncovered"] or sorted(perc["targets"])
    return [f"cover({x},{y})" for x, y in sorted(pts)] + ["all_targets_covered"]

def _pos_int(v):
    """`search_budget` 的每一项都得是正整数（validate_plan ④）。

    把 0/None 兜成 1 而不是删掉键：预算为 0 的 planner 等于没跑，报 1 是"至少给了界"，
    真出现这个值说明上游算错了，应该从 plan 的其余字段里查而不是让校验放行。
    """
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return 1

def ar25_plan(perc, method, path, search_budget, **extra):
    """把一次成功的 planner 产出包成 §3.2 plan，并当场过 `validate_plan`。

    `input_state_hash` / `expected_goal` / `subgoals` 全部取自分层的 `perc`——那是
    `r234_solve` 在动任何动作**之前**采的观测，正是这份 plan 成立时的局面。

    validity 口径与设计文档 §8.7（LS20 侧已采纳）一致：只有"离线算好、这轮原样重放又通过"
    的罐头解配 `verified_offline`；搜索当场算出的路径一律 `candidate`——§10 禁止把没验过的
    说成验过，而"搜索里引擎前进了一帧"不等于整条路径在落盘前被完整重放过。
    """
    planner = AR25_METHOD_TO_PLANNER[method]
    canned = planner == "t0_known_solution"
    state_hash = ar25_state_hash(perc)
    return pc.build_plan(
        planner=planner,
        input_state_hash=state_hash,
        candidate_actions=list(path),
        legal_actions=list(AR25_ACTION_NAMES),
        expected_goal=ar25_expected_goal(perc),
        subgoals=ar25_subgoals(perc),
        cost=len(path),
        search_budget=search_budget,
        validity="verified_offline" if canned else "candidate",
        # 罐头解：给出可定位的来源（解法表在本仓库的位置 + 本轮实测回放）。
        # 搜索产物：只给"这份 plan 是在哪一帧算出来的"的指针，不作任何验证声明。
        evidence_refs=([f"arc_adaptor/arc_shadow.py:KNOWN_SOLUTIONS[{perc['level']}]",
                        f"offline_engine:replay_win:L{perc['level'] + 1}"]
                       if canned else [f"ar25:L{perc['level'] + 1}:input_state_hash={state_hash}"]),
        ar25_method=method,
        level=perc["level"] + 1,
        **extra,
    )

# ═══════════════════════════════════════════════════════════
# R3-A: 目标分解搜索
# ═══════════════════════════════════════════════════════════

def enum_useful_positions(sprite, axis_pos, targets, atype):
    cells = sprite["cells"]
    sx, sy = sprite["x"], sprite["y"]
    positions = set()
    for tx, ty in targets:
        for cx, cy in cells:
            positions.add((tx - cx, ty - cy))
            if atype == 'h':
                positions.add((tx - cx, 2 * axis_pos - ty - cy))
            else:
                positions.add((2 * axis_pos - tx - cx, ty - cy))
    by_cov = {}
    for ox, oy in positions:
        abs_cells = {(ox + cx, oy + cy) for cx, cy in cells}
        if atype == 'h':
            refl = {(x, 2 * axis_pos - y) for x, y in abs_cells}
        else:
            refl = {(2 * axis_pos - x, y) for x, y in abs_cells}
        cov = frozenset(targets & (abs_cells | refl))
        if cov:
            d = abs(ox - sx) + abs(oy - sy)
            if cov not in by_cov or d < by_cov[cov][2]:
                by_cov[cov] = (ox, oy, d)
    return [(v[0], v[1], cov) for cov, v in by_cov.items()]

def solve_goal_decomp(g, perc):
    """R3 目标分解: 枚举轴位置×拼块位置组合, 找全覆盖最小成本。"""
    targets = perc["targets"]
    axes = perc["axes"]
    movable = perc["movable"]
    if perc["n_axes"] != 1 or not movable or perc["atype"] == '?':
        return None
    ax = axes[0]
    atype = ax["type"]
    ax_idx = next((s["idx"] for s in perc["switch_order"] if s["is_axis"]), None)
    n_sw = perc["n_switch"]
    sel0 = perc["sel_idx"]
    print(f"  [R3 目标分解] 轴类型={atype} 轴初始=({ax['x']},{ax['y']}) "
          f"拼块数={len(movable)} 目标={len(targets)}")
    best = None
    axis_range = range(0, BOARD)
    for ap in axis_range:
        all_pos = [enum_useful_positions(s, ap, targets, atype) for s in movable]
        if any(not p for p in all_pos):
            continue
        total_combos = 1
        for p in all_pos:
            total_combos *= len(p)
        if total_combos == 0:
            continue
        lists = all_pos
        if total_combos > GOAL_DECOMB_COMBO_CAP:
            lists = [sorted(p, key=lambda x: -len(x[2]))[:30] for p in all_pos]
        for combo in itertools.product(*lists):
            all_cov = set()
            for _, _, cov in combo:
                all_cov |= cov
            if all_cov == targets:
                cost = _compute_cost(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw)
                if best is None or cost < best[0]:
                    best = (cost, ap, combo)
    if best is None:
        return None
    cost, ap, combo = best
    print(f"  [R3 目标分解] ✓ 全覆盖! 轴→{ap} cost={cost}")
    return _generate_actions(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw)

def _compute_cost(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw):
    if atype == 'h':
        axis_cost = abs(ap - ax["y"])
    else:
        axis_cost = abs(ap - ax["x"])
    spr_cost = sum(abs(ox - s["x"]) + abs(oy - s["y"]) for s, (ox, oy, _) in zip(movable, combo))
    cur = sel0
    switch_cost = 0
    if ax_idx is not None:
        switch_cost += (ax_idx - cur) % n_sw
        cur = ax_idx
    for s in movable:
        switch_cost += (s["idx"] - cur) % n_sw
        cur = s["idx"]
    return axis_cost + spr_cost + switch_cost

def _generate_actions(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw):
    actions = []
    cur = sel0
    if ax_idx is not None:
        for _ in range((ax_idx - cur) % n_sw):
            actions.append(5)
        cur = ax_idx
        if atype == 'h':
            dy = ap - ax["y"]
            actions.extend([2] * dy if dy > 0 else [1] * (-dy))
        else:
            dx = ap - ax["x"]
            actions.extend([4] * dx if dx > 0 else [3] * (-dx))
    for s, (ox, oy, _) in zip(movable, combo):
        for _ in range((s["idx"] - cur) % n_sw):
            actions.append(5)
        cur = s["idx"]
        dx, dy = ox - s["x"], oy - s["y"]
        actions.extend([4] * dx if dx > 0 else [3] * (-dx))
        actions.extend([2] * dy if dy > 0 else [1] * (-dy))
    # §3.2：planner 出口只给 abstract name（内部仍按编号拼路径，出口统一转一次）
    return nums_to_names(actions)

# ═══════════════════════════════════════════════════════════
# R3-A2: 双轴目标分解 (L6+ 的 h+v 双镜像轴关卡)
# ═══════════════════════════════════════════════════════════

def _dual_reflect_chain(cells, vx, hy, max_depth=12):
    """双轴反射闭包: 与 ar25.py skqtojxvbv 相同的 BFS 级联 (深度≤12)。"""
    seen = set(cells)
    stack = [(x, y, 0) for x, y in cells]
    while stack:
        x, y, d = stack.pop()
        if d >= max_depth:
            continue
        for nx, ny in ((2 * vx - x, y), (x, 2 * hy - y)):
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                stack.append((nx, ny, d + 1))
    return seen

def _dual_enum_positions(sprite, vx, hy, targets):
    """枚举拼块平移位置, 返回 {覆盖集: (ox, oy, 距离)} 的去重字典。"""
    cells = sprite["cells"]
    sx, sy = sprite["x"], sprite["y"]
    cand = {}
    for tx, ty in targets:
        for cx, cy in cells:
            ox, oy = tx - cx, ty - cy
            abs_cells = {(ox + dx, oy + dy) for dx, dy in cells}
            cov = frozenset(_dual_reflect_chain(abs_cells, vx, hy) & targets)
            if not cov:
                continue
            d = abs(ox - sx) + abs(oy - sy)
            if cov not in cand or d < cand[cov][2]:
                cand[cov] = (ox, oy, d)
    return cand

def _dual_greedy_pick(movable, cand_list, targets):
    """贪心拼接: 每块选剩余覆盖最大的放置。"""
    uncovered = set(targets)
    picked = []
    spr_cost = 0
    for s, cand in zip(movable, cand_list):
        ranked = sorted(cand.items(), key=lambda kv: -len(kv[0] & uncovered))
        cov, (ox, oy, d) = ranked[0]
        uncovered -= cov
        picked.append((s["idx"], ox, oy))
        spr_cost += d
    return uncovered, picked, spr_cost

def _dual_exhaustive_pick(movable, cand_list, targets, max_combos=DUAL_COMBO_CAP):
    """穷举拼接: 枚举所有拼块位置组合, 找全覆盖最小成本。

    仅在拼块数≤4且组合数≤max_combos时启用, 否则回退贪心。
    """
    n = len(movable)
    if n > 5:
        return None
    top_per_sprite = []
    total = 1
    for s, cand in zip(movable, cand_list):
        ranked = sorted(cand.items(), key=lambda kv: -len(kv[0]))
        top_n = ranked[:8]
        top_per_sprite.append(top_n)
        total *= len(top_n)
        if total > max_combos:
            return None
    best = None
    for combo in itertools.product(*top_per_sprite):
        all_cov = set()
        cost = 0
        picked = []
        for s, (cov, (ox, oy, d)) in zip(movable, combo):
            all_cov |= cov
            cost += d
            picked.append((s["idx"], ox, oy))
        if all_cov >= targets:
            if best is None or cost < best[0]:
                best = (cost, picked)
    return best

def solve_goal_decomp_dual(g, perc):
    """R3 双轴目标分解: 枚举 (h_y, v_x) × 拼块位置, 找全覆盖组合。

    v5 改进: 先尝试穷举拼块排列 (组合数≤10万时), 失败再贪心。
    覆盖判定与引擎 skqtojxvbv 的反射级联一致。
    """
    if perc["n_axes"] != 2:
        return None
    h_axis = next((a for a in perc["axes"] if a["type"] == 'h'), None)
    v_axis = next((a for a in perc["axes"] if a["type"] == 'v'), None)
    movable = perc["movable"]
    if h_axis is None or v_axis is None or not movable:
        return None
    targets = perc["targets"]
    h_idx = next((s["idx"] for s in perc["switch_order"] if s["is_axis"] and s["x"] == h_axis["x"] and s["y"] == h_axis["y"]), None)
    v_idx = next((s["idx"] for s in perc["switch_order"] if s["is_axis"] and s["x"] == v_axis["x"] and s["y"] == v_axis["y"]), None)
    n_sw = perc["n_switch"]
    sel0 = perc["sel_idx"]
    budget = perc["budget"]
    print(f"  [R3 双轴分解] h@({h_axis['x']},{h_axis['y']}) v@({v_axis['x']},{v_axis['y']}) "
          f"拼块数={len(movable)} 目标={len(targets)}")
    best = None
    for hy in range(BOARD):
        for vx in range(BOARD):
            cand_list = [_dual_enum_positions(s, vx, hy, targets) for s in movable]
            if any(not c for c in cand_list):
                continue
            axis_cost = abs(hy - h_axis["y"]) + abs(vx - v_axis["x"])
            # v5: 先穷举, 再贪心
            ex = _dual_exhaustive_pick(movable, cand_list, targets)
            if ex is not None:
                spr_cost, picked = ex
            else:
                uncovered, picked, spr_cost = _dual_greedy_pick(movable, cand_list, targets)
                if uncovered:
                    continue
            # 切换成本: 两轴 + 拼块按序
            order = []
            if h_idx is not None:
                order.append(h_idx)
            if v_idx is not None:
                order.append(v_idx)
            order.extend(p[0] for p in picked)
            switch_cost = 0
            cur = sel0
            for idx in order:
                switch_cost += (idx - cur) % n_sw
                cur = idx
            cost = axis_cost + spr_cost + switch_cost
            if best is None or cost < best[0]:
                best = (cost, hy, vx, picked, h_idx, v_idx)
    if best is None:
        return None
    cost, hy, vx, picked, h_idx, v_idx = best
    print(f"  [R3 双轴分解] ✓ 全覆盖! h_y→{hy} v_x→{vx} cost={cost} picks={picked}")
    return _dual_generate_actions(h_axis, v_axis, hy, vx, picked, h_idx, v_idx,
                                  perc, n_sw, sel0)

def _dual_generate_actions(h_axis, v_axis, hy, vx, picked, h_idx, v_idx,
                           perc, n_sw, sel0):
    actions = []
    cur = sel0
    for idx, target_y, sprite in ((h_idx, hy, h_axis), (v_idx, vx, v_axis)):
        if idx is None:
            continue
        for _ in range((idx - cur) % n_sw):
            actions.append(5)
        cur = idx
        if sprite is h_axis:
            dy = target_y - h_axis["y"]
            actions.extend([2] * dy if dy > 0 else [1] * (-dy))
        else:
            dx = vx - v_axis["x"]
            actions.extend([4] * dx if dx > 0 else [3] * (-dx))
    by_idx = {s["idx"]: s for s in perc["movable"]}
    for s_idx, ox, oy in picked:
        for _ in range((s_idx - cur) % n_sw):
            actions.append(5)
        cur = s_idx
        s = by_idx[s_idx]
        dx, dy = ox - s["x"], oy - s["y"]
        actions.extend([4] * dx if dx > 0 else [3] * (-dx))
        actions.extend([2] * dy if dy > 0 else [1] * (-dy))
    return nums_to_names(actions)          # §3.2：出口只给 abstract name

# ═══════════════════════════════════════════════════════════
# R3-B: 状态空间搜索
# ═══════════════════════════════════════════════════════════

def r3_state_search(env, t_limit=60, max_nodes=200000):
    g = env._game
    if is_won(g):
        return []
    li0 = int(g._current_level_index)
    snap0 = _snapshot(g)
    try:
        g.next_level = lambda: None
    except:
        pass
    try:
        seq = 0
        heap = [(int(_heuristic(g)), 0, 0, seq, snap0, [])]
        seen = {_state_key(g)}
        t0 = time.time()
        while heap:
            if time.time() - t0 > t_limit or seq > max_nodes:
                return None
            h, _, _, _, snap, path = heapq.heappop(heap)
            mx = int(g.lelsvjlwneo.ilqnjlrnkk) - 1
            if len(path) >= mx:
                continue
            for a in AR25_ACTION_NAMES:      # §3.2：搜索树里的路径全程是 name 列表
                _restore(g, snap)
                try:
                    g.perform_action(act_input(a), raw=True)
                except:
                    continue
                if int(g._current_level_index) > li0 or is_won(g):
                    return path + [a]
                if g._state == GameState.GAME_OVER:
                    continue
                k = _state_key(g)
                if k in seen:
                    continue
                seen.add(k)
                seq += 1
                heapq.heappush(heap, (int(_heuristic(g)) * 10 + len(path) + 1, len(path) + 1, 0, seq, _snapshot(g), path + [a]))
        return None
    finally:
        try:
            del g.next_level
        except:
            pass
        _restore(g, snap0)

# ═══════════════════════════════════════════════════════════
# R3-C: Beam Search
# ═══════════════════════════════════════════════════════════

def r3_beam_search(env, t_limit=300, max_nodes=3000000, beam_width=8):
    g = env._game
    if is_won(g):
        return []
    li0 = int(g._current_level_index)
    snap0 = _snapshot(g)
    try:
        g.next_level = lambda: None
    except:
        pass
    try:
        max_steps = int(g.lelsvjlwneo.ilqnjlrnkk) - 1
        seq = 0
        beam = [(int(_heuristic(g)) * 10, 0, seq, snap0, [])]
        seen = {_state_key(g)}
        nodes = 0
        best_sol = None
        best_len = 999
        t0 = time.time()
        while beam and time.time() - t0 < t_limit and nodes < max_nodes:
            next_beam = []
            for _ in range(min(beam_width, len(beam))):
                if not beam:
                    break
                f, depth, _, snap, path = heapq.heappop(beam)
                if depth >= max_steps or depth >= best_len:
                    continue
                for a in AR25_ACTION_NAMES:      # §3.2：beam 的路径同样是 name 列表
                    _restore(g, snap)
                    try:
                        g.perform_action(act_input(a), raw=True)
                    except:
                        continue
                    if int(g._current_level_index) > li0 or is_won(g):
                        result = path + [a]
                        if len(result) < best_len:
                            best_len = len(result)
                            best_sol = result
                            print(f"    [beam] 🎯 {best_len}步 ({time.time()-t0:.0f}s, {nodes}节点)")
                        continue
                    if g._state == GameState.GAME_OVER:
                        continue
                    k = _state_key(g)
                    if k in seen:
                        continue
                    seen.add(k)
                    h_val = int(_heuristic(g))
                    seq += 1
                    nodes += 1
                    next_beam.append((h_val * 10 + depth + 1, depth + 1, seq, _snapshot(g), path + [a]))
            for item in next_beam:
                heapq.heappush(beam, item)
            if len(beam) > beam_width * 4:
                beam = heapq.nsmallest(beam_width * 2, beam)
        return best_sol
    finally:
        try:
            del g.next_level
        except:
            pass
        _restore(g, snap0)

# ═══════════════════════════════════════════════════════════
# R4: 剪枝贪心步进 (动作效果反馈 + 覆盖进度 + 反循环)
# ═══════════════════════════════════════════════════════════

class R4Scorer:
    def __init__(self):
        self.stats = {a: {"uses": 0, "eff": 0, "ineff": 0} for a in AR25_ACTION_NAMES}
        self.visited = set()
        self.recent = []

    def score(self, action, perc, prev_perc):
        s = self.stats[action]
        total = s["eff"] + s["ineff"]
        success_rate = s["eff"] / total if total > 0 else 0.5
        cov_gain = 0.0
        if prev_perc:
            cov_gain = (perc["covered"] - prev_perc["covered"]) * 5.0
        goal_prox = 0.0
        if perc["movable"] and perc["covered"] < perc["total_targets"]:
            sel = perc["sel_idx"]
            movable = perc["movable"]
            if 0 <= sel < len(movable) and perc["targets"]:
                sx, sy = movable[sel]["x"], movable[sel]["y"]
                targets = list(perc["targets"])
                cx = sum(t[0] for t in targets) / len(targets)
                cy = sum(t[1] for t in targets) / len(targets)
                dx, dy = ACTION_DELTA[action]   # §3.2：动作语义按 name 查表，不再比裸编号
                old_d = abs(sx - cx) + abs(sy - cy)
                new_d = abs(sx + dx - cx) + abs(sy + dy - cy)
                goal_prox = (old_d - new_d) * 0.3
        loop = self.recent[-5:].count(action) * 0.8 if len(self.recent) >= 5 else 0
        explore = 1.0 / (1 + s["uses"])
        return 2.0 * success_rate + cov_gain + goal_prox - loop + 0.3 * explore

    def choose(self, perc, prev_perc):
        cands = [(a, self.score(a, perc, prev_perc)) for a in AR25_ACTION_NAMES]
        cands.sort(key=lambda x: -x[1])
        return cands[0][0]

    def update(self, action, prev_perc, new_perc):
        eff = False
        if prev_perc and new_perc:
            if new_perc["covered"] > prev_perc["covered"]:
                eff = True
            if prev_perc["state_key"] != new_perc["state_key"]:
                eff = True
        self.stats[action]["eff" if eff else "ineff"] += 1
        self.stats[action]["uses"] += 1
        self.recent.append(action)
        self.visited.add(new_perc["state_key"])

def r4_greedy_play(env, t_limit=60, max_steps=200):
    """R4 贪心步进: 逐动作评分选择, 直到通关或超时。失败时恢复原状态。

    成功退出时引擎**保留**这段步进（与 beam/状态搜索相反），所以整段要 commit 进 recording；
    回滚那条路径 abort，一行都不留。
    """
    g = env._game
    if is_won(g):
        return []
    li0 = int(g._current_level_index)
    snap0 = _snapshot(g)
    scorer = R4Scorer()
    prev = r2_perceive(g)
    path = []
    t0 = time.time()
    with _attempt() as at:
        while time.time() - t0 < t_limit and len(path) < max_steps:
            perc = r2_perceive(g)
            if perc["won"] or int(g._current_level_index) > li0:
                at.commit()
                return path
            action = scorer.choose(perc, prev)
            try:
                at.step(g, action)
            except:
                continue
            path.append(action)
            new_perc = r2_perceive(g)
            scorer.update(action, prev, new_perc)
            prev = new_perc
            if new_perc["won"] or int(g._current_level_index) > li0:
                at.commit()
                return path
            if g._state == GameState.GAME_OVER:
                break
        _restore(g, snap0)
    return None

# ═══════════════════════════════════════════════════════════
# R0: 已知解法优先 (arc_shadow.KNOWN_SOLUTIONS)
# ═══════════════════════════════════════════════════════════

def try_known_solution(g, level_idx):
    """尝试 arc_shadow 中已验证的解法, 秒级返回。返回 abstract name 列表（§3.2）。

    回放成功时这段动作留在引擎上 → commit 进 recording；验证失败会 `_restore` → abort。
    """
    if level_idx not in KNOWN_SOLUTIONS:
        return None
    sol = nums_to_names(KNOWN_SOLUTIONS[level_idx])   # 解法表存的是编号，出口转成 name
    if not sol:
        return None
    snap = _snapshot(g)
    li0 = int(g._current_level_index)
    with _attempt() as at:
        try:
            run_names(g, sol, at=at)
            if int(g._current_level_index) > li0 or is_won(g):
                at.commit()
                print(f"  [R0-已知解法] ✓ {len(sol)}步直接通关")
                return sol
            print(f"  [R0-已知解法] 验证失败({len(sol)}步未通关, 可能关卡状态不匹配)")
        except Exception as e:
            print(f"  [R0-已知解法] 执行异常: {e}")
        _restore(g, snap)
    return None

# ═══════════════════════════════════════════════════════════
# 验证辅助: 执行动作序列并检查通关
# ═══════════════════════════════════════════════════════════

def _verify_and_return(g, path, method_name):
    """执行 abstract name 序列, 验证通关后返回, 失败则恢复原状态。

    与 `try_known_solution` 同一套证据口径：通关成立 → 整段 commit；随后 `_restore` 回滚 →
    整段 abort，不留"搜索尝试过"的帧。
    """
    snap = _snapshot(g)
    li0 = int(g._current_level_index)
    with _attempt() as at:
        try:
            run_names(g, path, at=at)
            if int(g._current_level_index) > li0 or is_won(g):
                at.commit()
                return path, method_name
            print(f"  [{method_name}] 验证失败({len(path)}步未通关)")
        except Exception as e:
            print(f"  [{method_name}] 执行异常: {e}")
        _restore(g, snap)
    return None, None

# ═══════════════════════════════════════════════════════════
# R2+R3+R4 组合入口 (v5: 增强预算 + 已知解法优先)
# ═══════════════════════════════════════════════════════════

def r234_solve(env, t_limit=900, level_idx=0):
    """R2 感知 → R3 分层搜索 → R4 贪心。返回 `(path, method, plan)`。

    - `path`: §3.2 的 abstract action name 列表；失败时为 `None`。
    - `method`: 历史分层串（写进 report.json 的 `method`），② 之后它只是"哪一层出的解"的标签；
      planner 名由 `AR25_METHOD_TO_PLANNER` 映射，plan 里用 `ar25_method` 同时留住两者。
    - `plan`: 该层产出、并已当场过 `validate_plan` 的九字段 plan；失败时为 `None`（决策失败
      就记没有 plan，不编一个 `validity=none` 的壳出来冒充出口）。

    `plan.input_state_hash` 取的是本函数进门、还没执行任何动作时的 `perc`。搜索失败的那些层
    会把引擎 `_restore` 回这一帧，所以后面成功的层仍然对得上这份哈希。
    """
    g = env._game
    li0 = int(g._current_level_index)

    # R2 感知（同时是 plan 的 input_state_hash / expected_goal / subgoals 的来源）
    perc = r2_perceive(g)
    r2_ar25.enrich(g, perc)            # 正式哈希要读引擎；不 enrich 则 state_hash 直接抛错
    if _TRAIL is not None:
        # 本关入口态必须就是上一条证据的终态。对不上不在这里抛异常——那属于记账缺陷，
        # 该以 `audit.chain_breaks` 的形式进 report.json 并把 verdict 判成 BLOCKED，
        # 而不是让一次真跑通因为记账中断（抛在这里等于把 FAIL 变成"没跑完"）。
        _TRAIL.check_continuity(ar25_state_hash(perc))
    is_hard = level_idx >= 5
    print(f"  [R2] L{level_idx+1} 轴={perc['n_axes']}({perc['atype']}) "
          f"拼块={len(perc['movable'])} 目标={perc['total_targets']} "
          f"覆盖={perc['covered']}/{perc['total_targets']} "
          f"预算={perc['budget']} {'[困难关卡]' if is_hard else ''}")

    # 0. 已知解法优先 (秒级)
    known = try_known_solution(g, level_idx)
    if known is not None:
        method = "R0-已知解法"
        # 罐头回放不做搜索：界 = 重放 len(path) 步、不展开任何分支节点
        budget = {"max_nodes": 1, "max_depth": _pos_int(len(known))}
        return known, method, ar25_plan(perc, method, known, budget)

    # 1a. R3 目标分解 (单轴关卡)
    if perc["n_axes"] == 1 and perc["atype"] != '?':
        path = solve_goal_decomp(g, perc)
        if path is not None:
            result, method = _verify_and_return(g, path, "R3-目标分解")
            if result is not None:
                # max_nodes 是「每个轴位下」的组合上限，max_depth 是枚举的轴位数 (0..20)
                budget = {"time_ms": _pos_int(t_limit * 1000),
                          "max_nodes": GOAL_DECOMB_COMBO_CAP, "max_depth": BOARD}
                return result, method, ar25_plan(perc, method, result, budget)

    # 1b. R3 双轴目标分解 (双轴关卡)
    if perc["n_axes"] == 2:
        path = solve_goal_decomp_dual(g, perc)
        if path is not None:
            result, method = _verify_and_return(g, path, "R3-双轴分解")
            if result is not None:
                # 外层枚举 (h_y, v_x) 共 BOARD² 组，每组内拼块组合上限 DUAL_COMBO_CAP
                budget = {"time_ms": _pos_int(t_limit * 1000),
                          "max_nodes": DUAL_COMBO_CAP, "max_depth": BOARD * BOARD}
                return result, method, ar25_plan(perc, method, result, budget)

    # 2. R3 beam search (L6+: 更宽beam更长预算)
    beam_w = 16 if is_hard else 8
    beam_t = min(300 if is_hard else 60, t_limit)
    beam_nodes = 3000000 if is_hard else 1000000
    print(f"  [R3-beam] 开始 (t_limit={beam_t}s, beam={beam_w}, max_nodes={beam_nodes})")
    path = r3_beam_search(env, t_limit=beam_t, beam_width=beam_w, max_nodes=beam_nodes)
    if path is not None:
        method = "R3-beam"
        budget = {"time_ms": _pos_int(beam_t * 1000), "max_nodes": beam_nodes,
                  "max_depth": _pos_int(perc["steps_left"]), "beam_width": _pos_int(beam_w)}
        return path, method, ar25_plan(perc, method, path, budget)

    # 3. R3 arc_shadow (L6+: 600s/2M节点)
    if is_hard:
        import arc_shadow
        arc_shadow._env = env
        arc_shadow._game = g
        arc_shadow._valid = True
        # arc_shadow 是老 API：它的 _num2act/_act2num/_ACTS 都以 int 为键、内部按编号搜索，
        # 所以这里仍然交回 int 表，只在它**返回**的那一刻转成 abstract name（§3.2 边界）。
        arc_shadow._num2act = ACT_MAP
        arc_shadow._act2num = {v: k for k, v in ACT_MAP.items()}
        arc_shadow._ActionInput = ActionInput
        arc_shadow._GameState = GameState
        arc_shadow._ACTS = [ACT_MAP[n] for n in [1, 2, 3, 4, 5]]
        shadow_t = min(600, t_limit)
        shadow_nodes = 2000000
        print(f"  [R3-arc_shadow] L{level_idx+1} (t_limit={shadow_t}s, max_nodes={shadow_nodes})")
        int_path = arc_shadow.solve_level(t_limit=shadow_t, max_nodes=shadow_nodes)
        if int_path is not None:
            method = "R3-arc_shadow"
            path = nums_to_names(int_path)      # int 出口 → §3.2 abstract name
            budget = {"time_ms": _pos_int(shadow_t * 1000), "max_nodes": shadow_nodes}
            return path, method, ar25_plan(perc, method, path, budget)

    # 4. R3 状态搜索 (加长预算)
    search_t = min(120 if is_hard else 30, t_limit)
    search_nodes = 500000 if is_hard else 200000
    print(f"  [R3-搜索] 开始 (t_limit={search_t}s, max_nodes={search_nodes})")
    path = r3_state_search(env, t_limit=search_t, max_nodes=search_nodes)
    if path is not None:
        method = "R3-搜索"
        budget = {"time_ms": _pos_int(search_t * 1000), "max_nodes": search_nodes}
        return path, method, ar25_plan(perc, method, path, budget)

    # 5. R4 贪心步进 (最终回退, 加长预算)
    r4_t = min(120 if is_hard else 30, t_limit)
    r4_steps = 300 if is_hard else 200
    print(f"  [R4-贪心] 开始 (t_limit={r4_t}s, max_steps={r4_steps})")
    path = r4_greedy_play(env, t_limit=r4_t, max_steps=r4_steps)
    if path is not None:
        method = "R4-贪心"
        budget = {"time_ms": _pos_int(r4_t * 1000), "max_nodes": r4_steps,
                  "max_depth": r4_steps}
        return path, method, ar25_plan(perc, method, path, budget)

    return None, "失败", None

# ═══════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════

def main():
    """跑关 + 出 §5 的三份产物：`manifest.json` / `recording.jsonl` / `report.json`。

    证据写入器在建好后、第一个动作之前就必须写下基线行（§5.2 第 1 行是 RESET 态）；
    跑测结束时用 `evidence_compat.replay_recording` **回读自己写的 recording** 来定判据，
    判据不手填 True（同 `run_ls20_r2r3.py:522` 的自证段）。
    """
    global _TRAIL, _RUN_ID, _EPISODE_ID
    MAX_LEVELS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    START_LEVEL = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    arcade = Arcade(environments_dir=environments_dir("ar25"),
                    operation_mode=OperationMode.OFFLINE)
    gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ar25")][0]
    print(f"game_id = {gid}")
    env = arcade.make(gid)
    env.reset()
    g = env._game

    # 证据目录与 run 身份必须在第一个动作之前就位；plan/观测都要引用同一个 run_id。
    commit, branch = git_output("rev-parse", "HEAD"), git_output("branch", "--show-current")
    _RUN_ID = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}-ar25-r234"
    _EPISODE_ID = gid
    run_dir = pathlib.Path(str(state_dir() / f"ar25_r234_{time.strftime('%Y%m%d_%H%M%S_')}{os.getpid()}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    recording_path = run_dir / "recording.jsonl"
    _TRAIL = TickTrail(recording_path, run_id=_RUN_ID, episode_id=_EPISODE_ID,
                       legal_actions=list(AR25_ACTION_NAMES))
    _TRAIL.baseline(_observe(g, tick=0), _engine_frame(g))

    print(f"\nAR25 R2+R3+R4 组合闯关 (r234 flow v5)")
    import arc_shadow as _as
    print(f"  arc_shadow 来源: {_as.__file__}")
    print(f"  搜索预算: L1-L5=60s / L6+=900s (beam 300s + arc_shadow 600s)")
    print(f"  总预算: 500 steps / 900.0s")
    print(f"  R3 出口: {pc.PLAN_SCHEMA}（每关一份九字段 plan，validity 分 candidate/verified_offline）")
    print(f"  R2 观测: {r2_ar25.OBSERVATION_SCHEMA}（input_state_hash = 语义态 sha256[:16]，"
          f"字段表见 r2_ar25.AR25_STATE_FIELDS）")
    print(f"  逐 tick 证据: {_rel_to_repo(recording_path)}（回滚的搜索帧不入证）")
    print(f"  关 方法             步数   预算     搜索耗时      节点 结果            plan")
    print(f"-----------------------------------------------------------------")

    total_steps = 0
    total_search = 0.0
    levels_info = []
    won_all = True
    levels_solved = 0            # 出过 plan 的关数：plan_hash_linked 的分母

    for level in range(MAX_LEVELS):
        if int(g._current_level_index) < level:
            print(f"{level+1:3d} 未能进入")
            won_all = False
            _flush_level(None)
            break
        if level < START_LEVEL:
            continue

        is_hard = level >= 5
        t_limit = 900 if is_hard else 60
        t0 = time.time()
        result, method, plan = r234_solve(env, t_limit=t_limit, level_idx=level)
        elapsed = time.time() - t0

        if result is None:
            budget = int(g.lelsvjlwneo.ilqnjlrnkk)
            print(f"{level+1:3d} {method:<18} {'-':>4} {budget:>4}   {elapsed:7.1f}s         ✗未通关")
            won_all = False
            # 失败就没有 plan：不拿 validity=none 的空壳冒充"出口已套契约"（§7.1 R3 门槛）
            levels_info.append({"level": level+1, "method": method, "steps": None,
                                "ok": False, "elapsed": elapsed, "plan": None})
            _flush_level(None)
            break

        # 目标分解 / 已知解法这两条分支在 _verify_and_return 里就已经把动作打在真引擎上了，
        # 所以这里一般是空转；beam 与 状态搜索 的 finally 会 _restore 回入口帧，
        # R4 贪心成功时不回滚——即"引擎停在当前关"的补放分支主要服务 beam/搜索这两层。
        if not (int(g._current_level_index) > level or is_won(g)):
            print(f"  [校验] 引擎仍停在 L{level+1}，补放 {len(result)} 步解法")
            with _attempt() as at:
                try:
                    run_names(g, result, at=at)
                except Exception as e:
                    print(f"  [校验] 补放中断: {e}")
                finally:
                    # 补放没有回滚点：已经执行掉的动作就是真发生在引擎上，必须留在证据里
                    at.commit()

        ok = int(g._current_level_index) > level or is_won(g)
        total_steps += len(result)
        total_search += elapsed
        levels_solved += 1
        budget = int(g.lelsvjlwneo.ilqnjlrnkk)
        plan_tag = f"{plan['planner']}/{plan['validity']}" if plan else "无plan"
        print(f"{level+1:3d} {method:<18} {len(result):>4} {budget:>4}   {elapsed:7.1f}s         {'✓通关' if ok else '✗未通关'} {len(result)}步  plan={plan_tag}")
        levels_info.append({"level": level+1, "method": method, "steps": len(result),
                            "ok": ok, "elapsed": round(elapsed, 1),
                            "planner": plan["planner"], "plan_id": plan["plan_id"],
                            "plan_validity": plan["validity"], "plan": plan})
        _flush_level(plan)
        if not ok:
            won_all = False
            break

    final_level = int(g._current_level_index)
    print(f"\n=================================================================")
    print(f"总计: {total_steps}步, {total_search:.1f}s搜索")
    print(f"结果: L{final_level+1}, state={g._state}, won={won_all}")

    trail_summary = _TRAIL.close()
    audit = trail_summary["hash_audit"]
    recorded = trail_summary["actions"]

    # ── 自证：回读刚写的 recording，判据由协议层给出，不手填 ──────────────
    replay_info, replay_ok = {}, False
    try:
        res = ev.replay_recording(recording_path, legal_actions=list(AR25_ACTION_NAMES))
        replay_ok = len(res.transitions) == recorded
        replay_info = {"transitions": len(res.transitions), "reset_count": res.reset_count,
                       "final_state": res.final_state,
                       "levels_completed": res.levels_completed}
    except ev.EvidenceValidationError as exc:
        replay_info = {"error": str(exc)}

    schema_ok, actions_legal, scan_err = True, True, ""
    with recording_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            try:
                row = ev.validate_tick(json.loads(line)["data"])
            except (ev.EvidenceValidationError, KeyError, json.JSONDecodeError) as exc:
                schema_ok, scan_err = False, f"line {line_number}: {exc}"
                break
            name = (row.get("requested_action") or {}).get("name")
            if name != "RESET" and name not in AR25_ACTION_NAMES:
                actions_legal, scan_err = False, f"line {line_number}: 非法动作 {name}"

    link = audit["plan_link"]
    criteria = {
        "schema_valid": schema_ok,
        "actions_legal": actions_legal,
        "replay_complete": replay_ok,
        # 每份 plan 的 input_state_hash 都得是本关首行的入口态，且**每个成功关**都出过 plan
        "plan_hash_linked": _TRAIL.plan_hash_linked and link["plans"] == levels_solved,
        # 引擎当前位置 == 最后一条证据的终态：有动作没记进 recording 就在这里现形
        "chain_continuous": audit["chain_breaks"] == 0,
        # 反向分歧 >0 = 正式哈希漏收了搜索键能区分的状态，那它不配当状态身份（§3.2 字段③）
        "state_hash_not_coarser_than_search_key": audit["hash_split_engine_key_only"] == 0,
        "terminal_verified": won_all or str(g._state) == str(GameState.GAME_OVER),
    }
    # 除 terminal_verified 之外的六条都是"这份证据自不自洽"的判据；全过才允许保留 PASS/FAIL。
    evidence_ok = all([schema_ok, actions_legal, replay_ok,
                       criteria["plan_hash_linked"], criteria["chain_continuous"],
                       criteria["state_hash_not_coarser_than_search_key"]])
    verdict = "PASS" if won_all else "FAIL"
    if not evidence_ok:
        verdict = "BLOCKED"        # §10：证据不自洽就当没通过，不做替换
    print(f"verdict={verdict} 记录 {trail_summary['ticks']} 行 / {recorded} 动作 "
          f"(丢弃回滚帧 {trail_summary['discarded_rows']} 行) "
          f"chain_breaks={audit['chain_breaks']} plan_link={link['linked']}/{link['plans']}")

    planner_tally, validity_tally = {}, {}
    for x in levels_info:
        if x.get("planner"):
            planner_tally[x["planner"]] = planner_tally.get(x["planner"], 0) + 1
            validity_tally[x["plan_validity"]] = validity_tally.get(x["plan_validity"], 0) + 1

    unproven_fields = [k for k, v in r2_ar25.AR25_STATE_FIELDS.items() if "未证实" in v]
    limitations = [
        "offline 引擎, 非 live Scorecard",
        "plan 是「每关一份、整条路径」的粒度：完整 plan 只盖在本关第一行，其余行只带 plan_id"
        "（join 键）；能证目标分解与预算，不能当逐帧决策链证据",
        "步数条不进 state_hash（同 r2_ls20 的 HUD 教训），逐 tick 记在 game_specific.steps_left",
        f"beam/状态搜索/arc_shadow 的回滚帧不入证：本轮丢弃 {trail_summary['discarded_rows']} 行",
        "R2 字段表里这些引擎字段语义未证实、但按 arc_shadow._snapshot 的可变性收进哈希: "
        + ", ".join(unproven_fields),
        f"哈希双向审计：语义变/搜索键没变 {audit['hash_split_semantic_only']} 次（新哈希更细），"
        f"反向 {audit['hash_split_engine_key_only']} 次；只扣一步不动几何 {audit['steps_only']} 次",
    ] + ([] if schema_ok else [f"tick schema: {scan_err}"]) \
        + ([] if actions_legal else [f"动作合法性: {scan_err}"]) \
        + ([] if replay_ok else [f"replay 不自洽: {replay_info}"]) \
        + ([] if criteria["chain_continuous"]
           else [f"{audit['chain_breaks']} 处引擎位置与最后一条证据的终态不符：有动作没进 "
                 "recording，样本见 report.game_specific.hash_audit.examples"]) \
        + ([] if criteria["state_hash_not_coarser_than_search_key"]
           else [f"{audit['hash_split_engine_key_only']} 次「搜索键变了、语义哈希没变」："
                 "state_hash 欠收了一类可变状态，样本见 hash_audit.examples"]) \
        + ([] if criteria["plan_hash_linked"]
           else [f"{link['plans'] - link['linked']} 份 plan 的 input_state_hash 对不上本关入口态，"
                 "样本见 report.game_specific.plan_link.examples"])
    report = ev.build_verification_report(
        run_id=_RUN_ID, tier="offline_engine", verdict=verdict, criteria=criteria,
        metrics={"actions": total_steps, "recorded_actions": recorded,
                 "recorded_ticks": trail_summary["ticks"],
                 "levels_completed": final_level,
                 # 引擎的 _current_level_index 是「当前关卡下标」，最后一关通关时不再 +1
                 "levels_cleared": final_level + 1 if won_all else final_level,
                 "total_search_seconds": round(total_search, 1),
                 "plans_by_planner": planner_tally, "plans_by_validity": validity_tally,
                 "steps_by_level": {x["level"]: x.get("steps") for x in levels_info}},
        evidence_refs=["recording.jsonl", "manifest.json", "result.json"],
        limitations=limitations,
        game_specific={"r2_schema": r2_ar25.OBSERVATION_SCHEMA, "plan_schema": pc.PLAN_SCHEMA,
                       "protocol_backend": ev.BACKEND, "replay": replay_info,
                       "hash_audit": {k: v for k, v in audit.items() if k != "plan_link"},
                       "plan_link": link,
                       "final_state_hash": trail_summary["final_state_hash"]},
    )
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    manifest = ev.build_manifest(
        run_id=_RUN_ID, game_id=gid, branch=branch, commit=commit,
        module_versions={"r2": r2_ar25.OBSERVATION_SCHEMA,
                         "r3": "r234-v5+plan_contract", "r4": "r234-v5",
                         "evidence": ev.SCHEMA_VERSION},
        evidence_tier="offline_engine", mode="execute", seed=None,
        limits={"max_steps": 500, "timeout_s": 900, "max_levels": MAX_LEVELS},
        source_recording=None,
        # 证据文件会被提交，里面只存相对仓库根的路径，不存机器绝对路径
        game_specific={"environments_dir": _rel_to_repo(environments_dir("ar25")),
                       "protocol_backend": ev.BACKEND,
                       # 正式 input_state_hash 覆盖哪些字段、每个字段是什么语义（§3.2 字段③
                       # 要能被第三方审计：只给一个 16 位哈希等于让人猜）。
                       "state_hash_fields": list(r2_ar25.SEMANTIC_STATE_FIELDS),
                       "state_hash_requires": list(r2_ar25.ENRICHED_KEYS),
                       "engine_field_semantics": r2_ar25.AR25_STATE_FIELDS,
                       "start_level": START_LEVEL},
    )
    manifest["artifacts"] = ["recording.jsonl", "report.json", "result.json", "manifest.json"]
    manifest["per_tick_recording"] = {"path": "recording.jsonl",
                                       "ticks": trail_summary["ticks"],
                                       "format": trail_summary["format"],
                                       "backend": trail_summary["backend"]}
    manifest["status"] = {"PASS": "completed", "FAIL": "completed",
                          "BLOCKED": "blocked"}[verdict]
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    # 保存结果到 JSON
    result_data = {
        # 证据基线（§5.1）：缺 branch/commit/evidence_tier 的跑测无法复现，也无法定级
        "schema_version": ev.SCHEMA_VERSION,
        "run_id": _RUN_ID,
        "agent_build": f"git:{commit}",
        "project_branch": branch,
        "module_versions": manifest["module_versions"],
        # 本地 arc_agi OFFLINE 引擎 = offline_engine，不等于 live Scorecard
        "evidence_tier": "offline_engine",
        "mode": "execute",
        "limits": {"max_steps": 500, "timeout_s": 900},
        "status": {"PASS": "pass", "FAIL": "fail", "BLOCKED": "blocked"}[verdict],
        "verdict": verdict,
        "verification_criteria": criteria,
        "environments_dir": _rel_to_repo(environments_dir("ar25")),
        # ③ 落地：逐 tick 帧真的写出来了，判据来自协议层回读，不是"未证"
        "per_tick_recording": {"path": "recording.jsonl", "ticks": trail_summary["ticks"],
                               "actions": recorded, "format": trail_summary["format"],
                               "backend": trail_summary["backend"],
                               "final_state_hash": trail_summary["final_state_hash"],
                               "hash_audit": audit},
        "limitations": limitations,
        # §3.2 R3 出口：每关一份 plan（levels[i]["plan"]），字段齐九个，落盘前已过 validate_plan
        "plan_schema": pc.PLAN_SCHEMA,
        "plans_emitted": sum(1 for x in levels_info if x.get("plan")),
        "planner_distribution": planner_tally,
        # 兼容既有读取方
        "game": gid,
        "mode_legacy": "OFFLINE",
        "flow": "r234-v5",
        "levels_completed": final_level,
        "won": won_all,
        "state": str(g._state),
        "total_steps": total_steps,
        "total_search_seconds": round(total_search, 1),
        "levels": levels_info,
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"RESULT_FILE={result_path}")

if __name__ == '__main__':
    main()