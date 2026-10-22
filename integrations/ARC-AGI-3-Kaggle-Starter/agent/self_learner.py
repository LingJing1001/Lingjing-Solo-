"""SelfLearner v6 — 参考冠军(Daniel Franzen)方法：观测目标 + BFS + 死亡/无效守卫。

冠军方法解码（来自 arc-agi-3-champion-solution.ipynb）：
1) 结构化观测（ARCAID3 词法视图）读出**目标**与动作规则；addendum 明确写：
   "when the game is about making an agent navigate to a target, write an explicit
   search algorithm such as BFS" —— 即「目标已知 -> 写 BFS」。
2) **Known-Death 守卫**：从某个完全相同棋盘状态发过会致死的动作为止；
   再次遇到直接拒绝（KnownDeathActionError），不浪费尝试。
3) **No-op 守卫**：重复无效动作被拒绝。
4) 真正裁决是环境（levels_completed 增加 = 过关）。

本实现（无 LLM，纯算法复刻其方法）：
- 归纳阶段：对每个方向键从干净初始帧独立观测 3 次，学出动作->每步位移（位置感知）；
  同时**提取静止目标掩码**（差分中从不动的非墙格）= 关卡目标，和可移动精灵颜色。
- 规划阶段：在运动学模拟器上 BFS，目标函数 = 把可移动精灵搬到目标掩码上
  （覆盖度 + 距离），直接搜「最短动作序列」；候选按该目标排序。
- 执行阶段：用真实环境裁决；**Known-Death / No-op 守卫**按 (棋盘状态, 动作) 记忆，
  跳过已知致死/无效动作，并随尝试累积（跨 reset/重放持久）。
- 关卡进度：已通关方案缓存，死亡 reset 后重放回快进；同关 BFS 搜索按起始画面缓存续跑。
"""
from __future__ import annotations

import os
import sys
import json
import time
import heapq
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np

_FILE = Path(__file__).resolve()
for cand in (_FILE.parents[3], _FILE.parents[2], Path.cwd()):
    if (cand / "lingjing_solo" / "arcaid3" / "effect_rules.py").is_file():
        sys.path.insert(0, str(cand))
        break
for cand in (_FILE.parents[2], _FILE.parents[3], Path.cwd()):
    if (cand / "arc_adaptor" / "ar25_solver" / "solver.py").is_file():
        # append（而非 insert(0)）：避免与框架 vendor/.../agents 包名冲突
        sys.path.append(str(cand / "arc_adaptor"))
        break

from agents.agent import Agent  # noqa: E402
from arcengine import FrameData, GameAction, GameState  # noqa: E402
from lingjing_solo.arcaid3.effect_rules import (  # noqa: E402
    EffectRuleLearner,
    background_color,
    find_entities,
    match_entities,
)

# 结构化求解（基于真实物理的 A*/宏搜索）—— 冠军方法"结构化观测 + 规则 + BFS"的落地：
# 直接从运行引擎抽取关卡结构（sprite/axes/targets/tags），用正确物理模型在线搜索，
# 不含任何逐关硬编码脚本。导入失败则退化为像素归纳+离线 BFS。
try:  # noqa: E402
    from ar25_solver.solver import solve as _ar25_solve, verify as _ar25_verify
    from ar25_solver.simulator import AR25Simulator as _AR25Simulator
    _HAVE_AR25 = True
except Exception:  # pragma: no cover
    _HAVE_AR25 = False

# 通用层（移植自冠军 ARCAID3 lexicon view）：只读像素网格的结构化观测 + 规则归纳。
# 与 ar25_solver 相反，它不依赖任何引擎内部属性，因此对 25 个盘都可用来做
# "观测 -> 归纳 when <action> [if/unless <rel>] then <verb>(<who>,<how>)"，
# 再用学到的规则做前瞻规划。专用求解器不可用时接管。
try:  # noqa: E402
    from lexicon import Perception as _Perception
    from lexicon.planner import RulePlanner as _RulePlanner
    from lexicon.llm_planner import LLMPlanner as _LLMPlanner
    from lexicon.llm_planner import selflearn_llm_enabled as _selflearn_llm_enabled
    from lexicon.graphsim import StateGraph as _StateGraph
    from lexicon.goalsim import (GoalModel as _GoalModel, GoalPolicy as _GoalPolicy,
                                 decompose as _decompose, Piece as _Piece,
                                 goal_text as _goal_text, keys_text as _keys_text,
                                 keys_kind as _keys_kind, diagnose as _diagnose)
    # 只读视图：把上面各层的结论翻译成模型能读的几行。它不返回动作、不参与决策，
    # 所以任何一层出错都只是少一行上下文，不会改变行为。
    from lexicon.view import build as _build_view
    _HAVE_LEXICON = True
except Exception:  # pragma: no cover
    _HAVE_LEXICON = False

    def _selflearn_llm_enabled() -> bool:
        """Fallback when the lexicon package is missing (same semantics)."""
        sl = str(os.environ.get("SELFLEARN_LLM", "") or "").strip().lower()
        if sl in ("1", "on", "true", "yes"):
            return True
        if sl in ("0", "off", "false", "no", "none"):
            return False
        return str(os.environ.get("LINGJING_LLM", "off")).strip().lower() in {
            "1", "on", "true", "yes",
        }

COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
KEYS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7")
MAX_ENTITY_CELLS = 600
WALL_COLOR = 10  # ar25 中央分隔墙；目标掩码排除它


def _is_simple_action(name: str) -> bool:
    """True = 不需要坐标的普通按键。点击类（ACTION6@r,c）不适合当候选——
    瞎试会点到画面正中，纯浪费一步。"""
    try:
        return not GameAction.from_id(
            int(str(name).replace("ACTION", ""))).is_complex()
    except Exception:
        return False


def _bbox_gap(p, q) -> int:
    """两个组件 bbox 的切比雪夫间隙；<=0 相交/相贴，1 = 差一格拼缝。"""
    gy = max(p.at[0], q.at[0]) - min(p.at[2], q.at[2])
    gx = max(p.at[1], q.at[1]) - min(p.at[3], q.at[3])
    return max(gy, gx)


def _grid(fr: FrameData):
    layers = getattr(fr, "frame", None) or []
    arrs = [np.asarray(x) for x in layers if np.asarray(x).size]
    if not arrs:
        return np.zeros((0, 0), dtype=int)
    return arrs[-1]


def _act_id(a) -> int:
    """'ACTION3' / '3' / 3 / GameAction -> 3. Unknown -> 0."""
    if isinstance(a, int):
        return a
    s = str(getattr(a, "name", a)).strip()
    if s.upper().startswith("ACTION"):
        s = s[6:]
    try:
        return int(s)
    except ValueError:
        return 0


def _changed(a, b) -> bool:
    if a.shape != b.shape:
        return True
    return bool(np.any(a != b))


def _find(grid):
    return find_entities(grid, background_color(grid), max_cells=MAX_ENTITY_CELLS)


def _side_key(grid, e):
    h, w = grid.shape[0], grid.shape[1]
    ec = e.c + e.w / 2.0
    er = e.r + e.h / 2.0
    sx = 1 if ec >= w / 2 else -1
    sy = 1 if er >= h / 2 else -1
    return (sx, sy)


def _canon(grid):
    if grid is None or not grid.size:
        return ()
    ents = _find(grid)
    return tuple(sorted((e.color, e.r, e.c, e.w, e.h) for e in ents))


def _score_target(grid, target_mask, movable_colors):
    """目标函数：把可移动精灵搬到 target 掩码上。
    返回越高越好：覆盖 target 的精灵格数 - 精灵质心到最近 target 的总距离。"""
    if grid is None or not grid.size:
        return 0
    if not target_mask:
        return 0
    ents = _find(grid)
    if not ents:
        return 0
    tset = target_mask
    s = 0.0
    for e in ents:
        if e.color not in movable_colors:
            continue
        cells = list(e.cells())
        if not cells:
            continue
        overlap = sum(1 for (r, c) in cells if (r, c) in tset)
        s += overlap * 2.0
        cy = e.r + e.h / 2.0
        cx = e.c + e.w / 2.0
        # 到最近 target 格的距离（用预计算的 target 质心近似，省去逐格距离）
        dmin = min(
            abs(cx - tc) + abs(cy - tr) for (tr, tc) in _TARGET_CENTROIDS
        ) if _TARGET_CENTROIDS else 0.0
        s -= dmin
    return s


# 全局目标质心缓存（同一次归纳内）
_TARGET_CENTROIDS: list = []


def _update_target_centroids(target_mask):
    global _TARGET_CENTROIDS
    if not target_mask:
        _TARGET_CENTROIDS = []
        return
    rs = [r for r, c in target_mask]
    cs = [c for r, c in target_mask]
    n = len(target_mask)
    _TARGET_CENTROIDS = [(sum(rs) / n, sum(cs) / n)]


# ----------------------------------------------------------------- 运动学模型
class KinModel:
    def __init__(self):
        self.kin = {}
        self.noop = set()

    def observe(self, prev_grid, action, cur_grid):
        bg = background_color(prev_grid)
        ea = _find(prev_grid)
        eb = _find(cur_grid)
        if not ea and not eb:
            return
        pairs = match_entities(ea, eb)
        table = self.kin.setdefault(action, {})
        found = False
        for i, j, dr, dc in pairs:
            if dr == 0 and dc == 0:
                continue
            e = ea[i]
            key = (e.signature, _side_key(prev_grid, e))
            table[key] = (dr, dc)
            found = True
        if not found:
            self.noop.add(action)

    def productive(self, keys):
        return [a for a in keys if self.kin.get(a)]

    @staticmethod
    def _apply(grid, action, kin, bg):
        ents = _find(grid)
        out = [[bg] * len(grid[0]) for _ in range(len(grid))]
        moved = []
        for e in ents:
            key = (e.signature, _side_key(grid, e))
            d = kin.get(key)
            moved.append((e, d if d else (0, 0)))
        for e, d in sorted(moved, key=lambda x: x[1] != (0, 0)):
            for r, c in e.cells():
                nr, nc = r + d[0], c + d[1]
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                    out[nr][nc] = e.color
        return out

    def predict(self, grid, plan, max_steps=64):
        g = [list(r) for r in grid]
        bg = background_color(np.asarray(g))
        for act, cnt in plan:
            kin = self.kin.get(act, {})
            if not kin:
                continue
            steps = cnt if cnt is not None else max_steps
            for _ in range(steps):
                g2 = self._apply(np.asarray(g), act, kin, bg)
                if g2 == g:
                    break
                g = g2
        return g


def _runs(path):
    runs = []
    for a in path:
        if runs and runs[-1][0] == a and runs[-1][1] is not None:
            runs[-1] = (a, runs[-1][1] + 1)
        else:
            runs.append((a, 1))
    return runs


# ═══════════════════════════════════════════════════════
# 跨运行记忆（自学的持久化层）
# ═══════════════════════════════════════════════════════

class _Memory:
    """把"验证过的方案"与"守卫"落盘，下次运行直接复用 —— 这是 agent 的记忆。

    存什么:
      plans  : (关卡, 结构化初始状态指纹) -> 已通关验证过的动作序列
      guards : (关卡, 像素状态指纹)       -> 已知致死 / 无效动作集合
      stats  : 累计命中次数 / 落盘次数

    刻意不存什么: 任何"关卡号 -> 脚本"的硬编码。记忆的键是**观测到的状态**，
    不是关卡编号，所以换盘、改布局、换初始位置都会自动未命中并退回求解，
    不存在背答案的风险。

    环境变量:
      SELFLEARN_MEMORY=0        关闭记忆
      SELFLEARN_MEMORY_PATH=... 指定记忆文件（默认与本文件同目录 .selflearn_memory.json）
    """

    MAX_PLANS = 500
    MAX_GUARDS = 4000

    def __init__(self, game_id: str, path: Optional[str] = None,
                 enabled: bool = True):
        self.game_id = str(game_id or "unknown")
        self.enabled = bool(enabled)
        self.path = Path(path) if path else (_FILE.parent / ".selflearn_memory.json")
        self.data: dict = {"version": 1, "games": {}}
        self.hits = 0
        self.saves = 0
        self._dirty = False
        if self.enabled:
            self._load()

    # ---- 读写 ----
    def _load(self):
        try:
            if self.path.is_file():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                self.data.setdefault("games", {})
                g = self.data["games"].get(self.game_id, {})
                print(f"[mem] loaded {self.path.name}: game={self.game_id} "
                      f"plans={len(g.get('plans', {}))} guards={len(g.get('guards', {}))} "
                      f"notes={len(g.get('notes', {}))}", flush=True)
                # 开局先把上次的复盘念一遍：这是 agent 唯一的"上次我学到什么"
                if g.get("notes"):
                    print("[mem] 历史复盘:\n" + self.notes_text(), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[mem] load failed ({type(e).__name__}: {e}) - start empty", flush=True)
            self.data = {"version": 1, "games": {}}

    def flush(self):
        if not self.enabled or not self._dirty:
            return
        try:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)  # 原子替换，避免写一半被打断
            self._dirty = False
        except Exception as e:  # noqa: BLE001
            print(f"[mem] flush failed: {type(e).__name__}: {e}", flush=True)

    def _game(self) -> dict:
        g = self.data["games"].setdefault(self.game_id, {})
        g.setdefault("plans", {})
        g.setdefault("guards", {})
        return g

    @staticmethod
    def _key(level, state_key) -> str:
        return f"{level}|{state_key!r}"

    # ---- 方案记忆 ----
    def lookup_plan(self, level, state_key):
        if not self.enabled:
            return None
        rec = self._game()["plans"].get(self._key(level, state_key))
        if rec and rec.get("actions"):
            self.hits += 1
            return list(rec["actions"])
        return None

    def save_plan(self, level, state_key, actions, strategy):
        """只在**真实引擎确认过关后**调用，保证记忆里全是被验证过的方案。"""
        if not self.enabled or not actions:
            return
        plans = self._game()["plans"]
        if len(plans) >= self.MAX_PLANS:  # 简单容量控制：丢最旧的
            for k in sorted(plans, key=lambda k: plans[k].get("ts", 0))[:50]:
                plans.pop(k, None)
        plans[self._key(level, state_key)] = {
            "actions": [int(a) for a in actions],
            "strategy": strategy,
            "steps": len(actions),
            "ts": time.time(),
        }
        self.saves += 1
        self._dirty = True
        self.flush()
        print(f"[mem] saved L{level} {len(actions)}步 ({strategy}) "
              f"total_plans={len(plans)}", flush=True)

    # ---- 守卫记忆（Known-Death / No-op）----
    def guards_for(self, level, canon):
        if not self.enabled:
            return None
        return self._game()["guards"].get(f"{level}|{canon}")

    def save_guards(self, level, canon, death, noop):
        if not self.enabled or (not death and not noop):
            return
        guards = self._game()["guards"]
        if len(guards) >= self.MAX_GUARDS:
            for k in list(guards)[:200]:
                guards.pop(k, None)
        key = f"{level}|{canon}"
        cur = guards.setdefault(key, {"death": [], "noop": []})
        cur["death"] = sorted(set(cur.get("death", [])) | set(death))
        cur["noop"] = sorted(set(cur.get("noop", [])) | set(noop))
        self._dirty = True

    # ---- 机制知识（导航层）：可走颜色/功能键/机关色/贫瘠目标 ----
    # 与 plans/guards 的区别：这里的键不是"状态指纹"而是"这个游戏的机制"，
    # 所以跨关、跨运行都该复用（冠军方案里对应"已通关关卡规则钉进 prompt"）。
    def rules_for(self):
        if not self.enabled:
            return None
        return self._game().get("rules")

    def save_rules(self, rules: dict):
        if not self.enabled or not rules:
            return
        cur = self._game().setdefault("rules", {})
        for k, v in (rules or {}).items():
            if isinstance(v, (set, list, tuple)):
                cur[k] = sorted({str(x) for x in v})
            elif v:
                cur[k] = v
        cur["ts"] = time.time()
        self._dirty = True

    # ---- 关卡复盘：给人看，也给下一关当先验 ----
    def save_note(self, level, note):
        """每关一条。同一关重跑以最后一次为准（失败也要留，失败是数据）。

        note 用位置参数传整个字典：里面本来就有 "level" 字段，若走 **kwargs
        会与形参撞名（实测 TypeError: got multiple values for 'level'）。
        """
        if not self.enabled:
            return
        notes = self._game().setdefault("notes", {})
        rec = notes.setdefault(str(int(level)), {})
        rec.update(note or {})
        rec["ts"] = time.time()
        self._dirty = True

    def note(self, level):
        if not self.enabled:
            return None
        return self._game().get("notes", {}).get(str(int(level)))

    def latest_kind(self, exclude_level=None):
        """最近一条带键语义的复盘。跨关先验用：同一个游戏里键不会变。"""
        if not self.enabled:
            return None
        skip = None if exclude_level is None else str(int(exclude_level))
        best = None
        for k, rec in self._game().get("notes", {}).items():
            if k == skip or not rec.get("kind"):
                continue
            if best is None or rec.get("ts", 0) > best.get("ts", 0):
                best = rec
        return (best or {}).get("kind")

    def notes_text(self) -> str:
        notes = self._game().get("notes", {})
        if not notes:
            return "(还没有复盘)"
        out = []
        for k in sorted(notes, key=lambda x: int(x)):
            r = notes[k]
            out.append(f"  L{k} [{r.get('result', '?')}] {r.get('steps', '?')}步 | "
                       f"{r.get('goal', '?')} | 按键 {r.get('keys', '?')} | "
                       f"{r.get('block', '?')}")
        return "\n".join(out)

    def summary(self) -> str:
        g = self.data["games"].get(self.game_id, {})
        return (f"memory: plans={len(g.get('plans', {}))} "
                f"guards={len(g.get('guards', {}))} hits={self.hits} saves={self.saves}")


class MyAgent(Agent):
    MAX_ACTIONS: int = int(os.environ.get("SELFLEARN_MAX_ACTIONS", "12000"))
    # 每局墙钟上限（秒）。0 = 不限。runner 可设 agent.time_deadline 传绝对
    # 截止时刻；也可用环境变量 ARC_GAME_TIME_LIMIT_MIN 传分钟数。
    TIME_LIMIT_SEC: float = 0.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 每局墙钟上限：runner 设绝对截止 time_deadline；否则看环境变量
        # ARC_GAME_TIME_LIMIT_MIN（分钟，0/缺省 = 不限）。
        self.time_deadline: Optional[float] = None
        self._deadline_hit: bool = False
        self._deadline_logged: bool = False
        try:
            self.TIME_LIMIT_SEC = (
                float(os.environ.get("ARC_GAME_TIME_LIMIT_MIN", "0") or 0) * 60.0
            )
        except (TypeError, ValueError):
            self.TIME_LIMIT_SEC = 0.0
        self._level = -1
        self._phase = "INDUCE"
        self._last_grid = None
        self._start_grid = None
        self._kin = KinModel()
        self._erl = EffectRuleLearner(
            cell_size=1, max_entity_cells=MAX_ENTITY_CELLS,
            min_support=2, min_confidence=0.5, min_lift=1.0,
            use_side_condition=True, max_dist=8.0,
        )
        self._danger: dict[str, int] = {}
        self._death_count = 0
        self._attempt = 0

        self._plan: list[tuple[str, Optional[int]]] = []
        self._pi = 0
        self._run = 0
        self._flat = 0
        self._dirty = False

        self._ind: list[str] = []
        self._ind_i = 0
        self._ind_phase = 0
        self._ind_obs = 0
        self._ind_prev = None

        # 已知致死/无效动作记忆：(canon_state) -> set(action)
        self._death: dict = {}
        self._noop: dict = {}

        self._target_mask: set = set()
        self._movable_colors: set = set()

        self._level_search: dict = {}
        self._proven_cache: dict = {}
        self._brute: list = []
        self._brute_idx = 0

        # ---- 在线 BFS/回溯搜索状态（真实引擎做转移，ACTION7 undo 回溯）----
        self._online = bool(int(os.environ.get("SELFLEARN_ONLINE", "1")))
        self._search_root: object = None      # 当前关卡根状态的 canon
        self._applied: list = []              # 从根到当前真实状态的"有效动作"路径(不含 undo)
        self._tried: dict = {}               # canon_state -> set(action) 已作为子节点尝试过
        self._pending_undo: int = 0          # 待发出的 undo 次数(回溯)
        self._prev_canon: object = None       # 上一步 canon(用于检测 no-op/转移)
        self._last_action: str = ""
        self._visited: set = set()           # 当前路径上已访问的 canon(破环)
        self._search_keys: list = []         # 候选动作顺序(启发式)

        # ---- 结构化求解（闭环模型预测控制）----
        self._use_struct = _HAVE_AR25 and os.environ.get("SELFLEARN_STRUCT", "1") == "1"
        # 记录初始值：失败次数在关卡内累计，换关后恢复（见关卡切换处）
        self._use_struct_base = self._use_struct
        self._struct_fail = 0
        self._struct_cache_key = None
        self._struct_plan: list = []
        # ---- 通用层：结构化观测 + 规则归纳（不依赖引擎内部，任何盘可用）----
        self._use_generic = _HAVE_LEXICON and os.environ.get("SELFLEARN_GENERIC", "1") == "1"
        # 消融开关：目标层与状态图各自可关，用于判断"过关是谁的功劳"。
        # SELFLEARN_GOAL=0 只关目标层（goalsim），SELFLEARN_GRAPH=0 只关状态图。
        self._use_goal = self._use_generic and os.environ.get("SELFLEARN_GOAL", "1") == "1"
        self._use_graph = self._use_generic and os.environ.get("SELFLEARN_GRAPH", "1") == "1"
        self._perc = _Perception() if _HAVE_LEXICON else None
        self._planner = _RulePlanner(self._perc) if _HAVE_LEXICON else None
        self._graph = None             # 观测到的状态转移图（真实边，不靠规则预测）
        self._graph_prev = None        # 上一拍通用决策的 (指纹, 动作)
        # 目标层：从画面上找出"要达成的东西"，再按"离目标更近"决策。
        # 纯覆盖式探索到不了终点（ar25 160 拍 76 局面 0 关已证），因为它没有终点。
        self._goal = None              # GoalModel：目标掩码 + 覆盖分
        self._gpol = None               # GoalPolicy：键语义 + 位移规划
        self._gplan: list = []         # 待执行的按键序列
        self._gvec = None              # 该计划对应的需要的位移
        self._gwarm: list = []         # 还没学过语义的键，先各按一次
        self._gsampled: set = set()    # 这一关已经按过、用于学语义的键
        self._gblocked: set = set()    # 当前位移向量下被挡住的方向
        self._gcov_prev: float = 0.0   # 上一拍的覆盖分，用来区分“挡住”与“正在被吸收”
        self._goal_log_at = 0
        # ---- 控制权仲裁：目标层必须证明自己在靠近目标，否则让位给导航层 ----
        # g50t 实测：按 A5 后出现的回放幽灵与玩家同形同位移，goalsim 的
        # "same-shape-as-mover" 把它认成目标 → 目标层带着玩家追影子，5 次 GAME_OVER，
        # 而导航层（已正确找到开关）一次都没拿到控制权。两个通用判据：
        #   影子   = 目标锚点与玩家锚点每拍同向同量平移 → 结构上追不到，本关永久退出
        #   无进展 = 连续 60 步覆盖分没涨 → 让位 40 步，之后重新给一次机会
        self._goal_acted = False       # 上一拍的动作是不是目标层发的（致死记账用）
        self._goal_anchor_prev = None  # 上一拍目标锚点 (ty, tx)，用于判影子
        self._goal_shadow = False      # 本关已判定目标为影子
        self._goal_best_cov = -1.0     # 本关见过的最高覆盖分
        self._goal_prog_at = 0         # 上次覆盖分上涨的步数
        self._goal_yield_until = -1    # 目标层让位到的步数
        self._goal_deaths = 0          # 本关目标层导致的 GAME_OVER 次数
        self._level_t0: int = 0        # 本关起始步，用来算“这关花了多少步”
        self._note_at: int = 0         # 上次写复盘快照的步数
        self._generic_last = None      # 上一拍由通用层发出的动作
        self._generic_fp = None        # 上一拍通用层的状态指纹
        self._generic_tried: dict = {}  # 指纹 -> 已试过的动作（规则不足时的有信息试探）
        self._generic_log_at = 0

        # ---- 通用导航层：把"会响应方向键的块"平移到目标（无任何游戏常量）----
        # avatar = 位移随按键而变的组件（计时条/巡逻 NPC 对每个键都一个样，据此排除）；
        # 位移语义 = 同签名组件按键前后的锚点差；障碍 = 非背景格 + 学到的过不去的格；
        # 规划 = 以位移为边的 BFS，每拍用真实帧重算。
        self._use_nav = self._use_goal and os.environ.get("SELFLEARN_NAV", "1") == "1"
        self._nav_prev = None          # 上一次通用决策时的棋盘
        self._nav_skip = False         # 死亡/reset 后的第一拍不做转移观测
        self._nav_votes: dict = {}     # sig -> {action: {delta: count}}
        self._nav_sig = None           # avatar 组件签名 (colour, size)
        self._nav_shape = None         # avatar 形状（相对 bbox 左上角的偏移集）
        self._nav_anchor = None        # avatar 当前锚点 (y, x)
        self._nav_walls: set = set()   # 学到的"按了键也站不上去"的格子（撞 2 次）
        self._nav_wall_hits: dict = {} # 格子 -> 撞了几次（1 次只是疑似）
        self._nav_floor: set = set()   # 学到的"可走颜色"（avatar 站过/周边就是地板）
        self._nav_movers: set = set()  # 观测到移动过的 (sig, 锚点)：它们不是目标
        self._nav_spawn = None         # 本关出生锚点（出生点上的东西不是目标）
        self._nav_tried: set = set()   # 已到达但没过关的目标锚点
        self._nav_pressed: set = set() # 已经按过一次用来学语义的键
        self._nav_gaveup = False       # avatar 识别失败，本层永久让位
        self._nav_target = None        # 当前 BFS 的目标锚点
        self._nav_ftemp: set = set()   # 顺路探过的中转锚点（BFS 到不了终点时用）
        self._nav_frontier: set = set()  # 已经当"最近可达点"走过一遍的锚点
        self._nav_front_tgt = None     # 当前正在挤过去的前沿锚点
        self._nav_switch: set = set()  # 踩上去会改变地图的格子（按钮/开关）
        self._nav_fn_tried: dict = {}  # 开关处 -> 已试过的非方向键
        self._nav_a5_tries = 0         # 无路时试过几次"功能键"
        self._nav_a5_at = -999         # 上次试功能键的时刻
        self._nav_log_at = 0
        # 复合实体：多色 sprite 会被同色连通切成好几块，用"同步同向平移"把碎块
        # 粘回一个整体（ls20 的玩家就是 5x5 里 12+9 两色拼的）
        self._nav_composite = os.environ.get("SELFLEARN_COMPOSITE", "1") == "1"
        self._nav_pat = None           # frozenset((dy, dx, colour)) 相对左上角
        self._nav_dir_votes: dict = {} # act -> {delta: count}（不区分签名，全局一份）
        self._nav_dir_at: dict = {}    # act -> 上次观测到该键位移的步号
        # ---- 跨关保留的"机制知识" ----
        # 几何（墙/锚点/出生点）每关都不同，必须清；机制不变，必须留：
        #   floor     可走颜色（从 avatar 足迹归纳，跨关累计）
        #   fn_map    按下去能让地图大变的功能键（g50t 的 A5 回放就是这类）
        #   xf_colors 踩上去会改造自身外观的机关色（ls20 的变形机）
        #   barren    到达也不过关的目标外观（"到了也白到"）
        # 全部从交互归纳，不含任何游戏常量。
        self._nav_rules = {
            "floor": set(), "fn_map": set(), "xf_colors": set(), "barren": set(),
            # movable_colors: 撞上去"像是墙"、后来自己挪走了的颜色 = 可推方块。
            # 规划时按"推得动"处理，不按"不可通行"处理——这是 g50t 那种
            # "你走一步它滚一格"的机关盘的关键区别（静态墙和滚动块长得一样）。
            "movable_colors": set(),
        }
        self._nav_mov_hits: dict = {}  # 颜色 -> 观察到"挪走过"的次数
        self._nav_pat_prev = None      # 上一拍自身外观（检测"被机关改造"）
        self._nav_xf_tried: set = set()  # 本关已踩过的机关锚点
        self._nav_arrive: dict = {}    # 目标锚点 -> 到达次数（判"贫瘠"要证据）
        self._nav_open_until: int = -1  # 机关刚把路打开：这段窗口内先走去拿路
        self._nav_gate_until: int = -1  # 闸门窗口：这段内墙表整体失效（盲冲）
        self._nav_blind: bool = False   # 当前是否处于闸门窗口
        self._nav_gate_wins: list = []  # 每次闸门窗口实测的有效拍数（跨关复用）
        self._nav_gate_start: int = -1  # 本次窗口起始步（用来实测窗口有多长）
        self._nav_gate_dead: set = set() # 本次窗口内撞过的轴（v/h），撞了就换轴
        self._nav_gate_axis: str = ""   # 本次窗口内正在走的轴（走廊里要一直走下去）
        self._nav_gate_cells: set = set()  # 上次闸门事件里"在动"的格子（开门区）
        self._nav_push: set = set()      # 撞上去但"可推"的方块格（本关）
        self._nav_chg_hist: list = []   # 每拍画面变化格数（算基线用）
        # 撞墙记不记墙（消融开关）：闸门类盘面撞一下就把整条走廊拉黑，
        # 而闸门是会被打开的，过早拉黑反而封死唯一的路。
        self._nav_wall_learn = os.environ.get("SELFLEARN_NAVWALL", "1") == "1"
        self._nav_death: dict = {}     # 锚点 -> 已知致死的动作（先验式拦截）
        self._nav_noop: dict = {}      # (锚点, 动作) -> [次数, 步号] 已知空转
        self._nav_acted = False        # 上一拍是不是导航层发的动作（致死记账用）
        self._nav_tgt_pos = None       # 目标位置粘性：外观会变，位置不会
        self._nav_target_pat = None    # 当前目标的外观（到了就用它记贫瘠）

        # ---- LLM 决策层（冠军的第二半：128K 模型读 code/rules 视图做决策）----
        # 关掉时完全不影响规则前瞻；开启后每 N 拍问一次模型，拿一小段动作队列。
        # SELFLEARN_LLM 是权威开关：=1 挂大模型，=0 退回纯算法。
        self._llm = (_LLMPlanner() if (_HAVE_LEXICON and _selflearn_llm_enabled())
                     else None)
        self._llm_queue: list = []
        self._llm_last_outcome = None
        self._llm_recent: list = []    # ["ACTION2->noop", ...] 给模型的短期记忆
        if self._llm is not None:
            print(f"[learn] llm layer on: backend={self._llm.backend} chunk={self._llm.chunk} "
                  f"every={self._llm.every} cap={self._llm.max_chars}", flush=True)
            if self._llm.backend == "openai":
                _ep = self._llm._endpoint()
                _think = os.environ.get("SELFLEARN_LLM_THINK", "0").lower() in (
                    "1", "true", "yes", "on")
                print(f"[learn] llm endpoint: {_ep['url']} model={_ep['model']} "
                      f"think={int(_think)}", flush=True)

        # ---- 跨运行记忆 ----
        self._mem = _Memory(
            game_id=getattr(self, "game_id", "unknown"),
            path=os.environ.get("SELFLEARN_MEMORY_PATH") or None,
            enabled=os.environ.get("SELFLEARN_MEMORY", "1") == "1",
        )
        # 待落盘：本次求解的方案，等真实引擎确认过关后才写入记忆
        self._pending_mem: Optional[tuple] = None
        self._mem_saved_levels: set = set()
        # 上一运行留下来的机制知识：这是"上次我摸清了这个游戏的什么脾气"
        _r = self._mem.rules_for()
        if _r:
            for _k in self._nav_rules:
                _v = _r.get(_k)
                if isinstance(_v, (list, tuple)):
                    # 颜色类集合必须还原成 int：落盘时统一转成了字符串，
                    # 直接塞回去会让 `int(cv[y, x]) in floor` 永远为 False
                    # （等于忘了上次学的一切，白跑一局）。
                    if _k in ("floor", "xf_colors", "movable_colors"):
                        _v = {int(x) for x in _v if str(x).lstrip("-").isdigit()}
                    self._nav_rules[_k] = set(_v)
            if any(self._nav_rules[k] for k in self._nav_rules):
                print(f"[learn] nav 复用上次运行的机制知识: "
                      f"floor={sorted(self._nav_rules['floor'])} "
                      f"fn={sorted(self._nav_rules['fn_map'])} "
                      f"xf={sorted(self._nav_rules['xf_colors'])} "
                      f"barren={len(self._nav_rules['barren'])} "
                      f"mov={sorted(self._nav_rules['movable_colors'])}", flush=True)
        print(f"[learn] layers struct={int(self._use_struct)} generic={int(self._use_generic)} "
              f"goal={int(self._use_goal)} graph={int(self._use_graph)} "
              f"online={int(self._online)}", flush=True)

    # ---------- 工具 ----------
    def _names(self, latest) -> list[str]:
        # latest 可能是 FrameData，也可能是 numpy 帧（离线回退路径）——做兜底
        aa = getattr(latest, "available_actions", None) if latest is not None else None
        if not aa:
            aa = getattr(self, "_last_available_actions", None)
        ids = [_act_id(a) for a in (aa or [])]
        ids = [i for i in ids if i]
        if not ids:
            ids = [1, 2, 3, 4]
        out = []
        for i in ids:
            try:
                out.append(GameAction.from_id(i).name)
            except Exception:
                out.append(f"ACTION{i}")
        return out

    def _keys(self, latest: FrameData) -> list[str]:
        return [n for n in self._names(latest) if n in KEYS]

    def _act(self, name: str) -> GameAction:
        # 防呆：容忍 int / "3" / "ACTION3" 三种写法。之前 `getattr(GameAction,"1")`
        # 会静默退回 ACTION1，导致通用层 30 步全在撞墙而毫无报错。
        key = str(name)
        if key.isdigit():
            key = f"ACTION{key}"
        # "ACTION6@row,col" —— 点击类动作要带格子坐标，否则引擎按 (32,32) 处理
        coords = None
        if "@" in key:
            key, _, pos = key.partition("@")
            try:
                rr, cc = pos.split(",")
                coords = (int(float(cc)), int(float(rr)))  # 引擎用 (x=col, y=row)
            except Exception:
                coords = None
        action = getattr(GameAction, key, None)
        if action is None:
            print(f"[learn] WARN unknown action {name!r} -> ACTION1", flush=True)
            action = GameAction.ACTION1
        if action.is_complex():
            action.set_data({"x": coords[0], "y": coords[1]} if coords else {"x": 32, "y": 32})
        action.reasoning = {"text": f"selflearner:{name}"}
        return action

    def _reset_action(self, why: str) -> GameAction:
        action = GameAction.RESET
        action.reasoning = {"text": f"selflearner:reset:{why}"}
        return action

    def _guarded(self, act: str, cur) -> bool:
        """True 表示该 (状态,动作) 已知致死或无效 -> 应跳过。"""
        if hasattr(cur, "frame"):  # 可能是 FrameData
            cur = _grid(cur)
        c = _canon(cur)
        if act in self._death.get(c, set()) or act in self._noop.get(c, set()):
            return True
        return False

    # ---------- 目标提取 ----------
    def _extract_target(self, frames_grids):
        """从归纳过程中采集的若干帧中，找出静止(从不动)的非墙非背景格 = 目标掩码；
        同时记录出现过失位(可移动)的精灵颜色。frames_grids: list[(prev,cur)] 差分序列。"""
        if not frames_grids:
            return
        base = frames_grids[0][0]
        ever = np.zeros_like(base, dtype=bool)
        mov_colors = set()
        for prev, cur in frames_grids:
            if prev.shape != base.shape:
                continue
            d = prev != cur
            ever |= d
            for y, x in zip(*np.nonzero(d)):
                mov_colors.add(int(prev[y, x]))
                mov_colors.add(int(cur[y, x]))
        # 静止非背景格
        stat = (base != 0) & (~ever)
        tmask = set()
        for y, x in zip(*np.nonzero(stat)):
            col = int(base[y, x])
            if col == WALL_COLOR or col == background_color(base):
                continue
            tmask.add((int(y), int(x)))
        self._target_mask = tmask
        # 可移动精灵颜色：去掉背景、墙、以及完全静止的目标色
        bg = background_color(base)
        self._movable_colors = {c for c in mov_colors if c not in (bg, WALL_COLOR)}
        _update_target_centroids(self._target_mask)
        print(
            f"[learn] target mask cells={len(self._target_mask)} "
            f"movable_colors={sorted(self._movable_colors)}",
            flush=True,
        )

    # ---------- 结构化观测 + 物理正确求解器（冠军方法落地）----------
    _IMMOVABLE = "0056icpryeujyf"  # 不可移动实体（从切换列表中排除）

    def _extract_level(self):
        """从运行中的真实引擎抽取结构化关卡数据（sprite/axes/targets/tags），
        并构造与引擎 ayyvxqrhnzw 完全一致的切换顺序 switch_order 与当前选中索引
        sel_idx。返回 level_data dict；失败返回 None。"""
        g = getattr(self, "arc_env", None)
        g = getattr(g, "_game", None) if g is not None else None
        if g is None:
            return None
        if not any(hasattr(g, a) for a in ("jtkyjqznbnp", "ouurgkpbbjj", "fswikrcrdmx")):
            # 不是 ar25 系的引擎：结构化抽取无意义，立刻让通用层（规则归纳）接管，
            # 不必等 3 次失败。
            self._use_struct = False
            return None
        try:
            def _sdict(s):
                px = np.asarray(s.pixels)
                h, w = int(px.shape[0]), int(px.shape[1])
                flat = px.reshape(-1).astype(int)
                # 引擎用 -1 作透明哨兵；若整块实心（没有 -1）则全部格子都算实体，
                # 绝不能退回"众数/中位数当透明"，否则实心精灵会被整体丢掉。
                has_sentinel = bool((flat == -1).any())
                if has_sentinel:
                    cells = [[int(c), int(r)] for r in range(h) for c in range(w)
                             if int(px[r, c]) != -1]
                else:  # 实心块：全部格子都是实体
                    cells = [[int(c), int(r)] for r in range(h) for c in range(w)]
                tags = [str(t) for t in (getattr(s, "tags", []) or [])]
                # 模拟器按 list 读 tags；同时保留 tag 字符串兼容旧调用方
                return {"x": int(s.x), "y": int(s.y), "cells": cells,
                        "tags": tags, "tag": "|".join(tags)}

            # 反射轴即使带 IMMOVABLE 也照样反射，必须保留（只有选中列表才排除它）
            axes_raw = list(getattr(g, "jtkyjqznbnp", []) or [])
            axis_ids = {id(a) for a in axes_raw}
            # 可移动精灵放前面，保证 switch_order 的下标与 sprites 前缀对齐；
            # 不可移动的静态精灵追加在尾部（只作为覆盖物参与胜利判定）
            movable_raw = [s for s in (getattr(g, "ouurgkpbbjj", []) or [])
                           if id(s) not in axis_ids
                           and self._IMMOVABLE not in getattr(s, "tags", [])]
            scenery_raw = [s for s in (getattr(g, "ouurgkpbbjj", []) or [])
                           if id(s) not in axis_ids
                           and self._IMMOVABLE in getattr(s, "tags", [])]
            sprites_raw = movable_raw + scenery_raw
            axes = [_sdict(a) for a in axes_raw]
            sprites = [_sdict(s) for s in sprites_raw]
            axis_idx = {id(a): i for i, a in enumerate(axes_raw)}
            sprite_idx = {id(s): i for i, s in enumerate(sprites_raw)}
            all_ent = [e for e in (getattr(g, "ayyvxqrhnzw", []) or [])
                       if self._IMMOVABLE not in getattr(e, "tags", [])]
            switch_order = []
            for e in all_ent:
                if id(e) in axis_idx:
                    switch_order.append(("axis", axis_idx[id(e)]))
                elif id(e) in sprite_idx:
                    switch_order.append(("sprite", sprite_idx[id(e)]))
            sel = getattr(g, "yvifanjrcyu", None)
            sel_idx = 0
            if sel is not None:
                try:
                    sel_idx = all_ent.index(sel)
                except ValueError:
                    sel_idx = 0
            targets = [{"x": int(t.x), "y": int(t.y)}
                       for t in (getattr(g, "fswikrcrdmx", []) or [])]
            bs = int(getattr(g, "dqwpuqcubca", 21))
            bud = int(getattr(getattr(g, "lelsvjlwneo", None), "ilqnjlrnkk", 64))
            lvl = int(getattr(g, "_current_level_index", self._level))
            ld = {"board_size": bs, "budget": bud, "level": lvl,
                  "sprites": sprites, "axes": axes, "targets": targets,
                  "_switch_order": switch_order, "_sel_idx": sel_idx}
            _dump = os.environ.get("SELFLEARN_DUMP_LEVEL", "")
            if _dump:
                try:
                    import json as _json
                    with open(f"{_dump}.L{lvl}.json", "w", encoding="utf-8") as _f:
                        _json.dump(ld, _f)
                    print(f"[learn] dumped level L{lvl} -> {_dump}.L{lvl}.json", flush=True)
                except Exception:  # noqa: BLE001
                    pass
            return ld
        except Exception as e:  # noqa: BLE001
            print(f"[learn] extract failed: {type(e).__name__}: {e}", flush=True)
            return None

    def _struct_state_key(self, ld, sel_idx, switch_order):
        sk = tuple(sorted((s["x"], s["y"]) for s in ld["sprites"]))
        ak = tuple(sorted((a["x"], a["y"]) for a in ld["axes"]))
        return (sel_idx, sk, ak)

    def _engine_state_key(self, ld, sel_idx):
        """真实引擎当前状态指纹：精灵位 + 轴位 + 选中下标。"""
        return (tuple((s["x"], s["y"]) for s in ld["sprites"]),
                tuple((a["x"], a["y"]) for a in ld["axes"]),
                sel_idx)

    def _structured_next_action(self, latest_frame):
        """闭环模型预测控制（MPC）。

        每拍从运行中的真实引擎抽取结构化状态；若与内部模拟器的预测一致，
        直接复用已求解计划的首动作（不再重解，速度快）；一旦出现偏差（切换
        顺序漂移、选中态不同、被墙挡住等），立即用真实物理模型重新求解并自愈。
        求解策略：A*（短时间预算）失败则回退 macro 宏搜索。
        返回 GameAction 或 None（退化为像素归纳方案）。
        """
        if not _HAVE_AR25:
            print("[learn] struct UNAVAILABLE (_HAVE_AR25=False)", flush=True)
            return None
        ld = self._extract_level()
        if ld is None:
            print("[learn] struct extract=None", flush=True)
            return None
        so = ld.pop("_switch_order", None)
        si = ld.pop("_sel_idx", 0)
        ekey = self._engine_state_key(ld, si)
        lv = ld.get("level", self._level)

        # ---- 记忆命中：直接复用上次通关验证过的方案（秒过，无需再求解）----
        cached = self._mem.lookup_plan(lv, ekey)
        if cached:
            try:
                self._struct_sim = _AR25Simulator(ld, sel_idx=si, switch_order=so)
            except Exception:  # noqa: BLE001
                self._struct_sim = None
            self._struct_plan = list(cached)
            a = self._struct_plan.pop(0)
            if self._struct_sim is not None:
                self._struct_sim.step(int(a))
            print(f"[learn] memory hit L{lv} {len(cached)+1}步 (hits={self._mem.hits})",
                  flush=True)
            return self._act(f"ACTION{a}")

        # ---- 预测一致：直接执行计划中的下一个动作，并推进内部模拟器 ----
        sim = getattr(self, "_struct_sim", None)
        if sim is not None and self._struct_plan:
            if sim.state_key() == ekey:
                a = self._struct_plan.pop(0)
                sim.step(int(a))
                return self._act(f"ACTION{a}")
            self._struct_drift = getattr(self, "_struct_drift", 0) + 1

        # ---- 偏差或首次：重新求解 ----
        # 单次求解预算（秒）。25 秒足够 L0-L5，但 L6/L7 这类 40+ 目标的双轴关卡
        # 的轨道搜索要 30~40 秒才能撞到解；25 秒会让它整关判死、退回像素学习。
        # 实测 50 秒可把 ar25 从 6/8 关推到 8/8 全通，而单局总耗时仍不到 3 分钟。
        tl = int(os.environ.get("SELFLEARN_SOLVE_TLIMIT", "50"))
        acts = None
        used = None
        # macro 通常极快（0.1~0.4s）先试；失败再用轨道覆盖搜索（多轴大关卡），
        # 最后才是逐步 A*（部分关卡需十几秒）
        for strat, budget in (("macro", max(tl, 10)), ("orbit", tl), ("astar", tl)):
            try:
                acts = _ar25_solve(ld, strategy=strat, t_limit=budget,
                                   verbose=False, sel_idx=si, switch_order=so)
            except Exception as e:  # noqa: BLE001
                print(f"[learn] solve({strat}) err: {type(e).__name__}: {e}", flush=True)
                acts = None
            if acts:
                used = strat
                break
        if not acts:
            self._struct_sim = None
            self._struct_fail = getattr(self, "_struct_fail", 0) + 1
            print(f"[learn] struct no-solution L{ld.get('level')} sprites={len(ld['sprites'])} "
                  f"axes={len(ld['axes'])} targets={len(ld['targets'])} sel={si} so={so} "
                  f"fail#{self._struct_fail}", flush=True)
            # 连续多次求解失败 -> 本关放弃结构化，退回像素自学习（避免死循环重试）
            if self._struct_fail >= 3:
                self._use_struct = False
                print("[learn] struct disabled after 3 failures -> pixel learner", flush=True)
            return None
        try:
            self._struct_sim = _AR25Simulator(ld, sel_idx=si, switch_order=so)
        except Exception:  # noqa: BLE001
            self._struct_sim = None
        # 记录待落盘：等真实引擎确认过关后才写进记忆，保证只记被验证过的方案
        if self._pending_mem is None and lv not in self._mem_saved_levels:
            self._pending_mem = (lv, ekey, list(acts), used)
        self._struct_plan = list(acts)
        a = self._struct_plan.pop(0)
        if self._struct_sim is not None:
            self._struct_sim.step(int(a))
        print(f"[learn] struct L{ld.get('level')} strat={used} "
              f"plan={len(acts)+1}步->{acts} sel={si}", flush=True)
        return self._act(f"ACTION{a}")

    # ---------- 通用层：结构化观测 + 规则归纳 + 规则前瞻 ----------
    def _generic_observe(self, cur_grid):
        """每拍读一次当前帧，并把上一拍通用层动作的结果喂给规则学习器。

        这是"归纳"的来源：两次观测之间发生了什么（哪个对象移动/变色/消失），
        配上当时执行的动作，就是规则语法的一条证据。

        动作必须**只用一次**：这一拍消费掉 `_generic_last` 后立即清空，
        否则同一动作会被配到后续多帧上，规则会退化成 `when any`（实测踩过）。
        """
        act, self._generic_last = self._generic_last, None
        self._generic_acted = act
        if not self._use_generic or self._perc is None:
            return
        try:
            arr = np.asarray(cur_grid)
            if arr.size == 0:
                return
            grid = [[int(v) for v in row] for row in arr]
            ev = self._perc.observe(grid, action=act)
            # 目标层也每拍看一次：首帧定目标掩码，之后累积"哪块在动"的证据，
            # 并用上一拍的动作反推"这个键把小块平移了多少"。
            if self._goal is not None:
                self._goal.observe(grid, act)
                if self._gpol is not None:
                    self._gpol.observe(grid, act)
            # 给 LLM 的短期记忆：这一步做了什么、看到几处变化。它比规则文本更早
            # 暴露"规则错了"，因为变化数与预测不符。
            if act:
                self._llm_recent.append(f"{act}->{len(ev or [])}chg")
                del self._llm_recent[:-12]
                self._llm_last_outcome = None
            if os.environ.get("SELFLEARN_DEBUG") == "1" and self.action_counter % 10 == 1:
                print(f"[learn] gdbg steps={self._perc.step_count} "
                      f"learner_steps={len(self._perc.learner.steps)} "
                      f"cands={len(self._perc.learner.cands)} "
                      f"acted={act} events={len(ev or [])} "
                      f"scale={self._perc.last_code['scale'] if self._perc.last_code else None}",
                      flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[learn] generic observe err: {type(e).__name__}: {e}", flush=True)

    def _generic_feedback(self, outcome):
        """把真实结果告诉规划器：'death' / 'nothing' / 'progress'。"""
        act = getattr(self, "_generic_acted", None)
        if self._planner is None or self._generic_fp is None or not act:
            return
        self._planner.record_outcome(self._generic_fp, act, outcome)
        # 让模型下一轮知道上一次尝试的结局（死亡/空转是它最容易忽略的信号）
        self._llm_last_outcome = outcome
        if outcome in ("death", "noop") and self._llm_recent:
            self._llm_recent[-1] = f"{self._llm_recent[-1]} [{outcome}]"

    def _click_targets(self, grid):
        """Where a click could plausibly matter: the centre of every object.

        A blind `ACTION6` always lands on (32,32), which is usually empty background
        — so click-type mechanics look like "this action does nothing" and the rules
        never learn them. Enumerating object centres turns the 64x64 coordinate space
        into a handful of meaningful candidates.
        """
        try:
            code = self._perc.encode_current(grid)
            if not code:
                return []
            out, seen = [], set()
            for p in code.get("placed") or []:
                cells = p.get("cells") or []
                if not cells:
                    continue
                r = sum(int(c[0]) for c in cells) // len(cells)
                c = sum(int(c[1]) for c in cells) // len(cells)
                if (r, c) in seen:
                    continue
                seen.add((r, c))
                out.append((r, c))
            return out
        except Exception:
            return []

    def _generic_probe(self, fp, grid=None):
        """规则还没学到时的有信息试探：优先没试过的、没被判死/无效的动作。"""
        keys = list(getattr(self, "_last_available_actions", None) or KEYS)
        tried = self._generic_tried.get(fp, set())
        cands = []
        for a in keys:
            if a == "ACTION6" and grid is not None:
                for r, c in self._click_targets(grid):
                    cands.append(f"ACTION6@{r},{c}")
            else:
                cands.append(a)
        for a in cands:
            if a in tried:
                continue
            if (fp, a.split("@")[0]) in self._planner.dead \
                    or (fp, a.split("@")[0]) in self._planner.noop:
                continue
            return a
        for a in keys:
            if (fp, a) not in self._planner.dead:
                return a
        return None

    def _llm_next_action(self, grid, fp):
        """LLM 决策层：先消费动作队列，队列空且到点就问模型。

        这是冠军解法里"模型读结构化视图做决策"的那一半，接在我们自己归纳出的
        `code` / `rules` / `predict` 三个视图上。返回 None = 未接管，交给规则前瞻。
        任何异常都吞掉并回退——模型是增强项，不是依赖项。
        """
        if self._llm is None:
            return None
        try:
            avail = tuple(a for a in (getattr(self, "_last_available_actions", None) or _ACTIONS)
                          if a in _ACTIONS)
            while self._llm_queue:
                a = self._llm_queue.pop(0)
                if avail and a.split("@")[0] not in avail:
                    continue
                self._generic_fp = fp
                self._generic_last = a
                self._generic_tried.setdefault(fp, set()).add(a)
                print(f"[learn] llm act {a} q={len(self._llm_queue)} why={self._llm.last_why}",
                      flush=True)
                return self._act(a)
            if not self._llm.ready(self.action_counter):
                return None
            # 各层只"汇报"，不"决定"：模型拿到这几行后既可以采用，也可以推翻。
            # 视图出错只会浪费几行 token，不会让 agent 走错路——这是它与
            # `_goal_next_action`（直接返回动作）的根本区别。
            inferred = _build_view(
                goal_model=self._goal, goal_policy=self._gpol,
                graph=self._graph, grid=grid, fp=fp,
                code=getattr(self._perc, "last_code", None))
            acts = self._llm.decide(
                self._perc, grid, self.action_counter, getattr(self._perc, "level", None),
                allowed=avail or _ACTIONS, recent=self._llm_recent,
                outcome=self._llm_last_outcome, planner=self._planner, fp=fp,
                inferred=inferred)
            if not acts:
                return None
            self._llm_queue = list(acts[1:])
            a = acts[0]
            self._generic_fp = fp
            self._generic_last = a
            self._generic_tried.setdefault(fp, set()).add(a)
            print(f"[learn] llm plan {acts} why={self._llm.last_why}", flush=True)
            return self._act(a)
        except Exception as e:  # noqa: BLE001
            print(f"[learn] llm next err: {type(e).__name__}: {e}", flush=True)
            return None

    def _generic_next_action(self, cur_grid):
        """用学到的规则做一步前瞻：预测每个动作会改变什么，优先没站过的局面。

        通用层的定位是"没有专用求解器时的地板"——它不假设任何游戏的物理，
        只用从这局自己的交互里归纳出来的规则。返回 None 表示交给像素路径。
        """
        if not self._use_generic or self._perc is None or self._planner is None:
            return None
        self._nav_acted = False
        self._goal_acted = False
        try:
            arr = np.asarray(cur_grid)
            if arr.size == 0:
                return None
            grid = [[int(v) for v in row] for row in arr]
            # 只考虑这一局真正可用的动作（否则会去试不存在的 ACTION5/6/7）
            avail = getattr(self, "_last_available_actions", None)
            if avail:
                allowed = tuple(a for a in avail if a in _ACTIONS)
                if allowed:
                    self._planner.actions = allowed
            fp = self._planner.fingerprint_of(grid)
            self._planner.note_visit(grid)

            # ---- 目标优先：先问"要达成什么"，再问"哪儿没去过" ----
            # 图与规则的打分都是"局面变新"，而变新不是终点。目标层给出一个真正
            # 的目的地（目标掩码）和一条位移计划，能算出来就直接执行。
            ga = self._goal_next_action(grid)
            if ga is not None:
                self._remember_generic(fp, ga)
                return self._act(ga)

            # ---- 导航层：目标层算不出计划时的备胎 ----
            # 曾排在目标层前面，理由是"BFS 绕墙比贪心分解强"。实测反了：ar25 上
            # 它把 88 格 3 色的复合体认成 avatar（真正可动的是 5 格单色的块），
            # 然后"全图探遍无路"，连试 ACTION5，整关 120 步 0 进展。认错对象时
            # 它不会返回 None —— 它会自信地走错，这比不干活更糟。
            # 导航层每次都要观测（学 avatar/墙/开关），但"要不要它来发动作"分两种
            # 时序：SELFLEARN_NAVLAST=1 时它退到状态图之后，只在前面几层都给不出
            # 计划时才接管——这才是"备胎"的本意。默认仍是抢先接管（旧行为）。
            _nav_last = os.environ.get("SELFLEARN_NAVLAST", "0") == "1"
            if self._use_nav:
                self._nav_observe(grid)
                if os.environ.get("SELFLEARN_DEBUG") == "1" and self._nav_sig is not None:
                    print(f"[navdbg] t={self.action_counter} act={getattr(self, '_generic_acted', None)} "
                          f"anchor={self._nav_anchor} walls={len(self._nav_walls)}", flush=True)
                if not _nav_last:
                    na = self._nav_next_action(grid)
                    if na is not None:
                        self._nav_acted = True
                        self._remember_generic(fp, na)
                        return self._act(na)

            # ---- 状态图：只走真实观测到的边 ----
            # 规则模型会外推，也就会在规则稀疏处自信地走错；图不会。代价是图只认识
            # 走过的局面，靠 frontier 策略（回到"还有键没试过"的局面）来长出新边。
            if self._graph is not None:
                self._graph.note_visit(fp)
                if self._graph_prev is not None:
                    src, pact = self._graph_prev
                    self._graph.observe(src, pact, fp)
                    del src, pact
                path = self._graph.plan(fp)
                if path:
                    if self.action_counter % 40 == 0:
                        print(f"[learn] {self._graph.stats()} fp={len(fp)} "
                              f"rules={len(self._perc.rules())}", flush=True)
                    a = self._materialise(path[0], grid, fp)
                    if a is not None:
                        self._remember_generic(fp, a)
                        return self._act(a)

            # 模型优先：它读的是同一份 code/rules 视图，但能做多步因果推理。
            if _nav_last and self._use_nav:
                na = self._nav_next_action(grid)
                if na is not None:
                    self._nav_acted = True
                    self._remember_generic(fp, na)
                    return self._act(na)
            acted = self._llm_next_action(grid, fp)
            if acted is not None:
                return acted
            a = self._planner.best_action(grid)
            if a is None:
                a = self._generic_probe(fp, grid)
            if a is None:
                return None
            self._remember_generic(fp, a)
            if self.action_counter - self._generic_log_at >= 25:
                self._generic_log_at = self.action_counter
                info = self._graph.stats() if self._graph is not None else ""
                print(f"[learn] generic rules [{info}]\n" + self._perc.rules_text(),
                      flush=True)
            return self._act(a)
        except Exception as e:  # noqa: BLE001
            print(f"[learn] generic next err: {type(e).__name__}: {e}", flush=True)
            return None

    def _goal_next_action(self, grid):
        """朝目标走：先学"每个键把小块平移多少"，再把需要的位移拆成按键序列。

        为什么不是"选覆盖上升的动作"：覆盖是阶跃信号。ar25 上走到目标旁边的
        11 步里覆盖恒为 0，最后两步才跳到 36%，所以按覆盖做贪心根本起不来；
        按"离目标最近"做贪心又会在"刚好碰到"（距离 0、覆盖 18%）处停住，因为
        再没有单个动作能提高覆盖。真正的信号是**需要的位移向量**：
        目标就是可动块的平移副本，答案 = 目标锚点 - 小块锚点。

        三个相位：
          warm-up  每个键按一次，学出它的平移（全局语义，跨局面复用）
          plan     需要的位移按学到的平移分解成按键序列，逐个执行
          re-plan  每拍都用真实帧重算向量，被挡住就换一条路（MPC）

        返回 None 表示"目标层没有可执行的计划"，交给状态图/规则/像素 BFS。
        """
        if not self._use_goal:
            return None
        if self._goal is None or self._gpol is None or not self._goal:
            return None
        # ---- 控制权仲裁（不硬编码任何游戏，只看"它能不能证明自己在靠近目标"）----
        if self._goal_shadow:
            return None
        if self.action_counter < self._goal_yield_until:
            return None
        acts = [a for a in (self._last_available_actions or [])
                if a in _ACTIONS and a not in ("ACTION6", "ACTION7")]
        if not acts:
            acts = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

        # ---- warm-up：语义未知的键必须先各按一次，否则无从分解位移 ----
        unknown = [a for a in acts
                   if a not in self._gpol.disp and a not in self._gsampled]
        if unknown:
            a = unknown[0]
            self._gsampled.add(a)
            return a

        cursors = self._gpol.cursors
        goal_cells = self._goal.goal_mask
        if not cursors or not goal_cells:
            return None

        # ---- 该动哪一块、动多少 ----
        # 目标通常是可动块的"幽灵"：形状相同。锚点用 bbox 左上角，形状一致时
        # 对齐锚点就是形状完全重合。多块时取已经压在目标上最多的那块。
        piece = max(cursors, key=lambda p: len(p.cells & goal_cells))
        ty = min(c[0] for c in goal_cells)
        tx = min(c[1] for c in goal_cells)
        # ---- 影子目标：目标块与玩家同步、同向、同量平移 → 它跟着你走，追不到 ----
        _pg = self._goal_anchor_prev
        self._goal_anchor_prev = (ty, tx)
        if _pg is not None:
            _gd = (ty - _pg[0], tx - _pg[1])
            _ld = self._gpol.last_delta
            _pd = _ld[1] if _ld else (0, 0)
            if _gd != (0, 0) and _gd == _pd:
                self._goal_shadow = True
                print(f"[learn] goal 目标与玩家同步平移 {_gd} → 判定为影子，"
                      f"目标层本关退出", flush=True)
                return None
        vec = (ty - piece.at[0], tx - piece.at[1])
        if vec == (0, 0):
            # 已经对齐：覆盖完成，剩下的交给引擎判定或别的层
            self._gplan, self._gvec = [], None
            return None

        if self._gvec != vec:
            self._gvec = vec
            self._gplan = []
            self._gblocked = set()

        # 上一拍按的键若没让小块移动，说明这个方向在当前局面被挡住。
        # 但"没移动"不等于"没用"：ar25 上把形状压到幽灵上时，小块是逐像素被
        # 吸收的（45→44→43 px），bbox 不动而覆盖在涨——那正是一条正确的路。
        # 所以只有**既没移动、覆盖也没涨**才判为挡住，而且一移动就立刻解除。
        ld = self._gpol.last_delta
        cov_now = self._goal.score(grid)
        if ld:
            a, d = ld
            if d != (0, 0):
                self._gblocked.discard(a)
            elif cov_now <= getattr(self, "_gcov_prev", 0.0) + 1e-9:
                self._gblocked.add(a)
        self._gcov_prev = cov_now

        # ---- 无进展就让位：覆盖分 60 步没涨，说明这条路走不通（g50t 追影子、
        # ar25 卡在"刚好碰到"都会表现成这样）。让位 40 步给导航/状态图，之后
        # 重新给目标层一次机会——不是永久禁用，目标层在 ar25 上仍是最好的层。
        if cov_now > self._goal_best_cov + 1e-9:
            self._goal_best_cov = cov_now
            self._goal_prog_at = self.action_counter
        elif not self._gplan and self.action_counter - self._goal_prog_at > 60:
            self._goal_yield_until = self.action_counter + 40
            print(f"[learn] goal {self.action_counter - self._goal_prog_at} 步覆盖无进展 "
                  f"→ 让位给导航层 40 步", flush=True)
            return None

        if not self._gplan:
            disp = {k: v for k, v in self._gpol.disp.items()
                    if k not in self._gblocked and k in acts}
            self._gplan = _decompose(vec, disp)
            if os.environ.get("SELFLEARN_DEBUG") == "1":
                print(f"[gdbg] vec={vec} blocked={sorted(self._gblocked)} "
                      f"disp={disp} plan={self._gplan[:6]} "
                      f"piece={piece} goal=({ty},{tx}) cov={self._goal.score(grid):.2f}",
                      flush=True)
            if not self._gplan:
                return None
            if self.action_counter - self._goal_log_at >= 20:
                self._goal_log_at = self.action_counter
                print(f"[learn] goal plan vec={vec} n={len(self._gplan)} "
                      f"| {self._goal.text()} | {self._gpol.text()}", flush=True)
        self._goal_acted = True
        return self._gplan.pop(0)

    def _write_note(self, result="running", steps=None):
        """把这一关的经过写成一句话存盘。

        存两样东西，用途不同：
          goal / keys / block —— 给人看，也是下一关开局的第一句话
          kind               —— 给程序读，新关直接复用，省掉一轮瞎按

        失败的关也写。之前只有"过关才存方案"，所以走了 830 步 0 关的时候
        盘上一个字都没有 —— 最该留下的恰恰是那种。
        """
        if not (self._use_goal and self._goal is not None):
            return
        try:
            gp = self._gpol
            disp = dict(gp.disp) if gp is not None else {}
            blocked = set(gp.blocked) if gp is not None else set()
            cursors = list(getattr(gp, "cursors", []) or [])
            cov = 0.0
            if self._goal.goal_mask and self._last_grid is not None:
                cov = float(self._goal.score(self._last_grid))
            used = int(steps if steps is not None else
                       (self.action_counter - getattr(self, "_level_t0", 0)))
            rec = {
                "level": int(self._level),
                "result": str(result),
                "steps": used,
                "goal": _goal_text(self._goal.goal),
                "keys": _keys_text(disp, blocked),
                "block": ("已过关" if result == "cleared" else
                          _diagnose(self._goal.goal, cursors, blocked, cov,
                                    bool(getattr(gp, "linked_now", False)))),
                "cov": round(cov, 3),
                "kind": _keys_kind(disp),
            }
            self._mem.save_note(self._level, rec)
            self._mem.flush()
            print(f"[note] L{self._level} {result} {used}步 | {rec['goal']} | "
                  f"{rec['keys']} | 卡点: {rec['block']}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[note] write failed: {type(e).__name__}: {e}", flush=True)

    # ---------- 通用导航层 ----------
    # 「把会响应方向键的那块平移到目标上」的通用解法。不含任何游戏常量：
    # 按键含义、步长、障碍颜色、目标偏移全部从这局自己的反馈里学。
    #   avatar   = 位移随按键而变的组件。计时条对每个键都平移同一向量、巡逻 NPC
                 # 的位移与按键无关且不集中，用这两条把它们排除。
    #   位移语义 = 同签名组件按键前后的锚点差（多块同签名时位移必须一致才采信）。
    #   障碍     = 当前帧里所有非背景格 + 学到的"按了键也站不上去"的格子。
    #   规划     = 以位移为边的 BFS（最短路），每拍用真实帧重算（MPC）。

    def _nav_pieces(self, cv, min_size=1, max_size=800):
        """导航层自带的组件 finder：与 GoalPolicy.pieces 的差别是允许 1 格小件
        （g50t 这类"环+芯"的 avatar，芯是独立颜色的小组件，得跟着一起算空位）。"""
        out = []
        h, w = cv.shape
        try:
            bg = int(background_color(cv))
        except Exception:
            bg = 0
        seen = np.zeros((h, w), dtype=bool)
        for y0 in range(h):
            for x0 in range(w):
                if seen[y0, x0]:
                    continue
                col = int(cv[y0, x0])
                seen[y0, x0] = True
                if col == bg:
                    continue
                stack = [(y0, x0)]
                cells = []
                while stack:
                    cy, cx = stack.pop()
                    cells.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx]
                                and int(cv[ny, nx]) == col):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                if min_size <= len(cells) <= max_size:
                    out.append(_Piece(col, cells))
        return out

    def _nav_index(self, cv):
        out = {}
        for p in self._nav_pieces(cv):
            out.setdefault((p.colour, p.size), []).append(p.at[:2])
        return out

    def _nav_piece(self, cv, sig, near=None):
        best, best_d = None, None
        for p in self._nav_pieces(cv):
            if (p.colour, p.size) != sig:
                continue
            if near is None:
                return p
            d = abs(p.at[0] - near[0]) + abs(p.at[1] - near[1])
            if best_d is None or d < best_d:
                best, best_d = p, d
        return best

    def _nav_mode(self, sig, act):
        per = self._nav_votes.get(sig, {}).get(act)
        if not per:
            return None
        tot = sum(per.values())
        d, n = max(per.items(), key=lambda kv: kv[1])
        return d if n >= 0.7 * tot else None

    def _nav_dir(self, act):
        """某个键的位移语义（全局一份，不分签名）。多色 sprite 换形状/转朝向时
        签名会变，按签名投票就学不到稳定位移，所以这里只按"按键"记。"""
        per = self._nav_dir_votes.get(act)
        if not per:
            return None
        tot = sum(per.values())
        d, n = max(per.items(), key=lambda kv: kv[1])
        return d if n >= 0.7 * tot else None

    def _nav_locate_composite(self, pv, cv, act, old=None):
        """用「同步同向平移」把多色 sprite 拼回一个整体。

        同一色连通会把*多色* sprite 切碎（ls20 的玩家是 5x5 里两种颜色拼的，
        被切成 10 格 + 15 格两块）。这些碎块每一拍的平移完全一致、bbox 又紧挨，
        所以按位移分组、取碎块最多的那组做 union，就是一个整体。
        单色 avatar（g50t）在这一步只是"组里只有一个碎块"，结果不变。

        但"同位移"不等于"同一实体"：回放幽灵和玩家节拍同步、位移相同，
        直接 union 会把两者拼成一个横跨画面的"大 avatar"，锚点跳到角落
        （g50t 按 A5 后锚点变 (1,1)/(63,55) 就是这么来的）。所以组内还要
        按空间连通度再分堆，连成一片的才可能是同一个 sprite；然后在多堆里
        按"离预测位置近 -> 外观对得上 -> 大"挑选。
        """
        prevp = self._nav_pieces(pv)
        curp = self._nav_pieces(cv)
        by = {}
        for p in prevp:
            by.setdefault((p.colour, p.size), []).append(p)
        moved = []
        for p in curp:
            cand = by.get((p.colour, p.size))
            if not cand:
                continue
            best = min(cand, key=lambda q: abs(q.at[0] - p.at[0])
                       + abs(q.at[1] - p.at[1]))
            d = (p.at[0] - best.at[0], p.at[1] - best.at[1])
            if d != (0, 0):
                moved.append((best, p, d))
        if not moved:
            return None
        groups = {}
        for m in moved:
            groups.setdefault(m[2], []).append(m)
        # 预测位置：上一帧锚点 + 这个键的众数位移（撞墙没动时上面已提前返回）
        pred = None
        if old is not None and act in KEYS:
            dm = self._nav_dir(act) or (
                self._nav_mode(self._nav_sig, act) if self._nav_sig else None)
            if dm:
                pred = (old[0] + dm[0], old[1] + dm[1])
        best_cells = None
        best_key = None
        best_d = (0, 0)
        for d, grp in groups.items():
            # 组内按连通度分堆：同一堆的碎块在空间上连成一片，才可能是
            # 同一个多色 sprite；幽灵/别的实体各归各堆。
            for cl in self._nav_conn_clusters(grp):
                cells = set()
                for _o, new, _dd in cl:
                    cells |= set(new.cells)
                y0 = min(c[0] for c in cells)
                x0 = min(c[1] for c in cells)
                dist = (abs(y0 - pred[0]) + abs(x0 - pred[1])) if pred else None
                if dist is not None and dist > 24:
                    continue  # 离预测位置太远：是幽灵/别的实体，不是玩家
                if dist is not None:
                    key = (0, dist, -len(cells))
                elif self._nav_pat is not None:
                    pat = frozenset((y - y0, x - x0, int(cv[y, x]))
                                    for y, x in cells)
                    key = (1, 0 if self._nav_pat_same(pat, self._nav_pat) else 1,
                           -len(cells))
                else:
                    key = (2, 0, -len(cells))
                if best_key is None or key < best_key:
                    best_key = key
                    best_cells = cells
                    best_d = d
        if not best_cells:
            return None
        y0 = min(c[0] for c in best_cells)
        x0 = min(c[1] for c in best_cells)
        pat = frozenset((y - y0, x - x0, int(cv[y, x])) for y, x in best_cells)
        shape = frozenset((y - y0, x - x0) for y, x in best_cells)
        if act in KEYS:
            per = self._nav_dir_votes.setdefault(act, {})
            per[best_d] = per.get(best_d, 0) + 1
            self._nav_dir_at[act] = self.action_counter
        return (y0, x0), shape, pat

    @staticmethod
    def _nav_conn_clusters(grp):
        """把同位移的碎块按空间连通度分堆（允许 1 格拼缝）。

        同一个 sprite 的异色碎块彼此紧挨；回放幽灵和玩家即使位移相同，
        位置也相隔整个画面，绝对不会连成一片。
        """
        clusters = []
        for m in sorted(grp, key=lambda m: (m[1].at[0], m[1].at[1])):
            a = m[1]
            hit = None
            for cl in clusters:
                if any(_bbox_gap(a, q[1]) <= 1 for q in cl):
                    hit = cl
                    break
            if hit is None:
                clusters.append([m])
            else:
                hit.append(m)
        return clusters

    def _nav_pick_avatar(self):
        best, best_key = None, None
        for sig, per in self._nav_votes.items():
            if len(per) < 2:
                continue  # 至少两个键有反应
            modes = {}
            for a, c in per.items():
                tot = sum(c.values())
                d, n = max(c.items(), key=lambda kv: kv[1])
                if n < 0.7 * tot:
                    modes = None  # 同一个键位移忽左忽右 = 巡逻物，不是玩家
                    break
                modes[a] = d
            if not modes:
                continue
            uniq = {d for d in modes.values() if d != (0, 0)}
            if len(uniq) < 2:
                continue  # 永远同一个方向 = 计时条这类装饰物
            key = (sig[1], len(uniq))
            if best_key is None or key > best_key:
                best, best_key = sig, key
        return best

    def _nav_observe(self, cur_grid):
        """把上一拍「按键 -> 组件平移」的观测记进投票箱，维护 avatar 锚点/形状。"""
        prev = self._nav_prev
        act = getattr(self, "_generic_acted", None)
        self._nav_prev = cur_grid
        if self._nav_skip or prev is None or act is None:
            self._nav_skip = False
            return
        try:
            pv, cv = np.asarray(prev), np.asarray(cur_grid)
            # 注意不能因为"这一拍画面没变"就提前退出：按了方向键却被引擎拒绝
            # （撞墙、出界、通道关闭）恰恰就是"画面毫无变化"。那种拍子正是学
            # 障碍的唯一机会，必须放进去走下面的记墙分支。
            if pv.shape != cv.shape:
                return
            pi, ci = {}, self._nav_index(cv)
            pp = self._nav_pieces(pv)          # 上一帧的组件（判"结构变化"要用整块）
            for p in pp:
                pi.setdefault((p.colour, p.size), []).append(p.at[:2])
            for sig, before in pi.items():
                after = ci.get(sig)
                if not after or len(after) != len(before):
                    continue
                # 最近邻配对（同签名多块时排序配对会张冠李戴，幽灵块就是受害者）
                rem = sorted(after)
                for bb in sorted(before):
                    j = min(range(len(rem)),
                            key=lambda i: (abs(rem[i][0] - bb[0])
                                           + abs(rem[i][1] - bb[1])))
                    d = (rem[j][0] - bb[0], rem[j][1] - bb[1])
                    if d != (0, 0) and act in KEYS:
                        per = self._nav_votes.setdefault(sig, {}).setdefault(act, {})
                        per[d] = per.get(d, 0) + 1
                        # 移动过的具体实例（NPC/计时条/回放幽灵的出发点）不能当目标
                        self._nav_movers.add((sig, tuple(bb)))
                    rem.pop(j)
            old = self._nav_anchor
            comp = (self._nav_locate_composite(pv, cv, act, old)
                    if self._nav_composite else None)
            if comp is not None:
                anchor, shape, pat = comp
                if self._nav_pat is None:
                    print(f"[learn] nav 复合 avatar={anchor} pat={len(pat)}格 "
                          f"({len(set(c for _, _, c in pat))}色)", flush=True)
                self._nav_anchor = anchor
                self._nav_shape = shape
                # —— 自身外观被改造：刚踩着的非地板格就是"机关" ----
                # ls20 这类"变形走迷宫"：地板上的机器会改掉你携带的方块。判据纯从
                # 像素来：外观变了 + 落脚点有块既不是地板也不是自己的颜色。
                if (self._nav_pat is not None and self._nav_pat != pat
                        and act in KEYS and old is not None):
                    try:
                        bgc = int(background_color(cv))
                    except Exception:
                        bgc = -1
                    own_cols = {c for _, _, c in pat}
                    hh, ww = cv.shape
                    for (oy, ox) in shape:
                        yy, xx = anchor[0] + oy, anchor[1] + ox
                        if not (0 <= yy < hh and 0 <= xx < ww):
                            continue
                        v = int(cv[yy, xx])
                        if (v not in self._nav_floor and v != bgc
                                and v not in own_cols
                                and v not in self._nav_rules["xf_colors"]):
                            self._nav_rules["xf_colors"].add(v)
                            print(f"[learn] nav 机关色 {v}: 踩上去自身外观变了 "
                                  f"at={anchor}", flush=True)
                self._nav_pat_prev = self._nav_pat
                self._nav_pat = pat
                # 换关后 avatar 可能变大变小（携带的方块不同）：签名跟着刷，
                # 否则按旧尺寸做的排除/回退会一直错位
                if (self._nav_sig is None
                        or (isinstance(self._nav_sig, tuple)
                            and self._nav_sig[0] == "composite"
                            and self._nav_sig[1] != len(pat))):
                    self._nav_sig = ("composite", len(pat))
                if self._nav_pat_prev is not None and self._nav_pat_prev != pat:
                    # 外观变了 = 刚被机关改造。这一拍的跟踪不可信（碎块的
                    # 颜色/尺寸签名换了，拼出来的"整体"可能少一块），别拿它
                    # 做撞墙/撤墙的判断；下一拍图案稳定了再正常跟踪。
                    return
            elif self._nav_composite and self._nav_pat is not None and old is not None:
                # 复合模式，但这一拍没有任何东西平移 = 按键被拒（撞墙/边界）。
                # 位置原地不动，交给下面"按了键却没动 -> 记墙"的分支处理。
                # 这里绝不能走按签名的 fallback：多色 avatar 的签名根本对不上，
                # 一 return 就永远学不到墙，只能在原地震。
                self._nav_anchor = old
            else:
                if self._nav_sig is None:
                    self._nav_sig = self._nav_pick_avatar()
                    if self._nav_sig is not None:
                        print(f"[learn] nav avatar={self._nav_sig} "
                              f"(votes={sum(len(v) for v in self._nav_votes.values())})",
                              flush=True)
                if self._nav_sig is None:
                    return
                # 撞墙没动时退回按签名最近邻跟踪（此时没有位移可用来拼整体）
                if old is None:
                    piece = self._nav_piece(cv, self._nav_sig)
                else:
                    pred = old
                    if act in KEYS:
                        dm = self._nav_mode(self._nav_sig, act)
                        if dm:
                            pred = (old[0] + dm[0], old[1] + dm[1])
                    piece = self._nav_piece(cv, self._nav_sig, pred)
                if piece is None:
                    return
                self._nav_anchor = piece.at[:2]
                self._nav_shape = frozenset(
                    (y - piece.at[0], x - piece.at[1]) for y, x in piece.cells)
            if act not in KEYS or old is None:
                return
            d = (self._nav_anchor[0] - old[0], self._nav_anchor[1] - old[1])
            # —— 执行前预测的对账：学了位移就该能预测这一拍落在哪 ----
            # 落空且原地没动 = 这个 (锚点, 动作) 是空转，记两次以上规划时就不再提议
            # （dfranzen 的 KnownNoOpActionError 的无模型版）。会过期，避免被
            # NPC 临时挡路这种一次性因素永久屏蔽。
            if d == (0, 0):
                k = (tuple(old), act)
                rec = self._nav_noop.get(k)
                if rec is None or self.action_counter - rec[1] > 40:
                    self._nav_noop[k] = (1, self.action_counter)
                else:
                    self._nav_noop[k] = (rec[0] + 1, self.action_counter)
            # —— 地图是会变的：颜色变过的格子不再是墙 ----
            # 闸门开合/机关伸缩/踏板落盖/回卷都属这类。只靠一次性墙表会把"已经打开
            # 的路"永久挡住，所以每拍都把上一拍是墙、这一拍颜色变了的格子从墙名单
            # 里放出来（真还是墙的话，下次撞到会重新记进去）。
            # 注意这不限于"走通了"：功能键（回放/传送）也能让地图大变，那种拍子
            # avatar 往往被搬走，同样要走这段逻辑。
            # —— 闸门事件识别：看"画面变化的规模"而不是"墙有没有被冲掉"——
            # 玩家自己走一步只变 own 那么多格（g50t 里还有一块同步移动的回放幽灵，
            # 所以是 2×own）。某拍突然变出好几倍，说明有别的东西在动：闸门开合、
            # 机关伸缩、镜头回卷。用"历史中位数"当基线，不写死任何常量。
            try:
                _chg = int((np.asarray(pv) != np.asarray(cv)).sum())
            except Exception:
                _chg = 0
            self._nav_chg_hist.append(_chg)
            if len(self._nav_chg_hist) > 40:
                self._nav_chg_hist.pop(0)
            _base = 2 * max(1, len(self._nav_shape or ())) * 2  # 玩家 + 一个同步影子
            _med = sorted(self._nav_chg_hist)[len(self._nav_chg_hist) // 2]
            _outlier = _med > 0 and _chg > 1.8 * _med and _chg > _base
            if _outlier:
                _cy, _cx = np.nonzero(np.asarray(pv) != np.asarray(cv))
                _cells = {(int(y), int(x)) for y, x in zip(_cy.tolist(), _cx.tolist())}
                # 去掉玩家自己占的位置：那不是"世界在动"
                _own_now = {(self._nav_anchor[0] + oy, self._nav_anchor[1] + ox)
                            for (oy, ox) in (self._nav_shape or ())}
                _world = _cells - _own_now
                self._nav_gate_cells = _world
                _win2 = self._nav_gate_win_len()
                self._nav_gate_start = self.action_counter
                self._nav_gate_dead = set()
                self._nav_gate_axis = ""
                self._nav_gate_until = self.action_counter + _win2
                self._nav_open_until = self._nav_gate_until
                if _world:
                    _ys = [c[0] for c in _world]
                    _xs = [c[1] for c in _world]
                    print(f"[learn] nav 闸门事件 t={self.action_counter}: "
                          f"{_chg} 格在动(基线 {_med}) 开门区 "
                          f"y{min(_ys)}-{max(_ys)} x{min(_xs)}-{max(_xs)} "
                          f"-> 抢 {_win2} 拍", flush=True)
            dropped = 0
            if os.environ.get("SELFLEARN_DEBUG") == "1" and self.action_counter < self._nav_gate_until:
                # 量闸门时序：窗口内每拍画面变了多少格。变了 = 门又动了
                # （开/关），这条数据用来实测"窗口能撑几拍"，不写死常量。
                try:
                    _ch = int((np.asarray(pv) != np.asarray(cv)).sum())
                except Exception:
                    _ch = -1
                print(f"[gatedbg] t={self.action_counter} changed={_ch} "
                      f"left={self._nav_gate_until - self.action_counter} "
                      f"act={act} anchor={self._nav_anchor} "
                      f"dead={sorted(self._nav_gate_dead)} axis={self._nav_gate_axis}",
                      flush=True)
            if self._nav_walls:
                cy2, cx2 = np.nonzero(pv != cv)
                for yy, xx in zip(cy2.tolist(), cx2.tolist()):
                    if (yy, xx) in self._nav_walls:
                        self._nav_walls.discard((yy, xx))
                        dropped += 1
                        # 这格刚才是墙、现在不是了 = 它自己挪走了。按颜色记账，
                        # 攒够 2 次就把这个颜色归成"可动障碍"（跨关有效）。
                        _mc = int(np.asarray(pv)[yy, xx])
                        self._nav_mov_hits[_mc] = self._nav_mov_hits.get(_mc, 0) + 1
                        if self._nav_mov_hits[_mc] >= 2 and \
                                _mc not in self._nav_rules["movable_colors"]:
                            self._nav_rules["movable_colors"].add(_mc)
                            print(f"[learn] nav 可动障碍: 颜色 {_mc} 自己挪走过 "
                                  f"{self._nav_mov_hits[_mc]} 次 -> 规划时按'可推'"
                                  f"处理，不再当墙", flush=True)
            if d == (0, 0):
                # 按了方向键但 avatar 没动：把"新进入的非地板格"标成墙。
                # 整块标会误伤脚下地板；拦路的只能是闸门/实体这类非地板格。
                dm = self._nav_dir(act) or self._nav_mode(self._nav_sig, act)
                if dm and self._nav_shape is not None \
                        and self._nav_wall_learn:
                    h2, w2 = cv.shape
                    cur = {(self._nav_anchor[0] + oy, self._nav_anchor[1] + ox)
                           for (oy, ox) in self._nav_shape}
                    for (oy, ox) in self._nav_shape:
                        yy = self._nav_anchor[0] + dm[0] + oy
                        xx = self._nav_anchor[1] + dm[1] + ox
                        if ((yy, xx) in cur
                                or not (0 <= yy < h2 and 0 <= xx < w2)):
                            continue
                        # 撞不动就是挡路：记一次"疑似墙"。但闸门是会被打开的，
                        # 一次撞不动的证据不够硬（也可能是回放幽灵临时挡道、
                        # 或者闸门正要开）。同一个格子撞够 2 次才拉黑，给它一次
                        # "下次再来试试"的机会——这条对"闸门只开几拍"的关卡是
                        # 决定性的（g50t 的走廊被判死一次就再也走不过去）。
                        # 可动障碍不当墙：它下次可能就在别处了（甚至这步就能推动）
                        if int(cv[yy, xx]) in self._nav_rules["movable_colors"]:
                            self._nav_push.add((yy, xx))
                            continue
                        if self._nav_blind:
                            # 窗口内撞墙：这个轴这次窗口内别走了，换另一个轴
                            self._nav_gate_dead.add("v" if dm[0] else "h")
                            self._nav_gate_axis = ""
                        self._nav_wall_hits[(yy, xx)] = \
                            self._nav_wall_hits.get((yy, xx), 0) + 1
                        if self._nav_wall_hits[(yy, xx)] >= 2:
                            self._nav_walls.add((yy, xx))
                        if int(cv[yy, xx]) in self._nav_floor:
                            self._nav_floor.discard(int(cv[yy, xx]))
            elif self._nav_shape is not None:
                # 走通了：沿途都是可走的，从墙名单里除名（清掉 NPC 临时挡路）
                y0 = self._nav_anchor[0] - d[0]
                x0 = self._nav_anchor[1] - d[1]
                sy = (d[0] > 0) - (d[0] < 0)
                sx = (d[1] > 0) - (d[1] < 0)
                for i in range(max(abs(d[0]), abs(d[1])) + 1):
                    yy, xx = y0 + sy * i, x0 + sx * i
                    for (oy, ox) in self._nav_shape:
                        self._nav_walls.discard((yy + oy, xx + ox))
            if dropped >= 3:
                # 地图结构大变 = 刚触发了什么。三类知识：
                # 0) 闸门窗口：不管玩家是不是被搬走了（回卷类机关会把玩家丢回
                #    出生点，脚下反而是地板，sw 判据抓不到），只要"地图刚大变"，
                #    就说明某扇门可能开过——接下来几拍墙表一律不作数，先冲。
                #    窗口长度用实测值：上一次窗口从开到再变一共撑了几拍。
                _win = 16
                if self._nav_gate_wins:
                    _ws = sorted(self._nav_gate_wins[-5:])
                    _win = max(8, min(40, _ws[len(_ws) // 2]))
                if self._nav_gate_start >= 0:
                    self._nav_gate_wins.append(
                        self.action_counter - self._nav_gate_start)
                    if len(self._nav_gate_wins) > 20:
                        self._nav_gate_wins.pop(0)
                self._nav_gate_start = self.action_counter
                self._nav_gate_dead = set()
                self._nav_gate_axis = ""
                self._nav_gate_until = self.action_counter + _win
                self._nav_open_until = self._nav_gate_until
                print(f"[learn] nav 闸门窗口 ({_win} 拍, 地图变了 {dropped} 格)"
                      f": 这几拍忽略墙表直冲终点", flush=True)
                # 1) 刚按的是功能键 -> 这个键能让地图大变，跨关记住（无路时优先试它）
                if act is not None and act not in KEYS:
                    if act not in self._nav_rules["fn_map"]:
                        self._nav_rules["fn_map"].add(act)
                        print(f"[learn] nav 功能键 {act}: 按下去地图大变 "
                              f"(动了 {dropped} 个墙格)", flush=True)
                # 2) avatar 新占的"非地板格"就是触发它的开关，记下来待会儿在它上面试键
                hh, ww = pv.shape
                sw = set()
                for (oy, ox) in self._nav_shape:
                    yy = self._nav_anchor[0] + oy
                    xx = self._nav_anchor[1] + ox
                    if (0 <= yy < hh and 0 <= xx < ww
                            and int(pv[yy, xx]) not in self._nav_floor):
                        sw.add((yy, xx))
                if sw:
                    # 只有"脚下确实压着非地板格"才算开关；没有就只是画面在动，
                    # 别把探路进度也一起清掉（清了会原地打转）
                    self._nav_switch |= sw
                    # 机关刚把路打开：闸门开合往往只管几拍（g50t 的回放幽灵
                    # 就是"替我压住闸门"的临时工）。所以这段时间不要回头去
                    # 踩机关（状态狩猎/试键），先顺着新开的路往终点冲。
                    self._nav_ftemp.clear()
                    self._nav_frontier.clear()
                    self._nav_front_tgt = None
                    if self.action_counter - self._nav_log_at >= 5:
                        self._nav_log_at = self.action_counter
                        print(f"[learn] nav 开关: at={self._nav_anchor} "
                              f"踩上去地图变了 (动了 {dropped} 个墙格, "
                              f"开关格 {len(sw)})", flush=True)
        except Exception as e:  # noqa: BLE001
            if os.environ.get("SELFLEARN_DEBUG") == "1":
                print(f"[learn] nav observe err: {type(e).__name__}: {e}", flush=True)

    # ---- 外观（图案）比较：多色 sprite 的形状+颜色+相对排布 ----
    @staticmethod
    def _nav_pat_of(cells, cv):
        """把一块格子压成相对外观：{(dy, dx, colour)}，平移无关。"""
        if not cells:
            return None
        y0 = min(c[0] for c in cells)
        x0 = min(c[1] for c in cells)
        return frozenset((y - y0, x - x0, int(cv[y, x])) for y, x in cells)

    @staticmethod
    def _nav_flips(pat):
        """外观的八种朝向（四组镜像 + 四组旋转）：目标图案可能有任意朝向，
        变形类游戏里"旋转"本身就是要把状态调对的维度之一。"""
        if not pat:
            return []
        out = []
        for fy in (1, -1):
            for fx in (1, -1):
                out.append(frozenset((fy * dy, fx * dx, c) for dy, dx, c in pat))
                out.append(frozenset((fx * dx, fy * dy, c) for dy, dx, c in pat))
        return out

    @staticmethod
    def _nav_pat_key(pat):
        return repr(sorted(pat))

    def _nav_pat_same(self, a, b):
        """两块外观是否一致（允许镜像/半转）。"""
        if not a or not b:
            return False
        if a == b:
            return True
        bb = set(self._nav_pat_key(f) for f in self._nav_flips(b))
        return any(self._nav_pat_key(f) in bb for f in self._nav_flips(a))

    def _nav_pat_barren(self, pat):
        """这个外观是不是"到了也白到"（贫瘠目标）？"""
        if not pat or not self._nav_rules["barren"]:
            return False
        barren = self._nav_rules["barren"]
        return any(self._nav_pat_key(f) in barren for f in self._nav_flips(pat))

    def _nav_can_stand(self, cv, shape, p, own, forbidden, goal_set, allow_goal):
        """空地判定（从这局自己的成功/失败里学出来的近似引擎规则）：
        - 足迹至少一格踩在"地板色"上（悬空 = 引擎拒绝）；
        - 不压学到的墙 / 会动的东西（NPC、计时条）；
        - 其余颜色（开关、轨道、装饰、虚空）乐观放行——走过去试一下，
          被引擎拒绝就把"新进入的非地板格"记成墙，下次绕开。"""
        h, w = cv.shape
        touch = 0
        for (oy, ox) in shape:
            y, x = p[0] + oy, p[1] + ox
            if not (0 <= y < h and 0 <= x < w):
                return False
            # 闸门窗口内墙表整体失效：机关刚把路打开，旧墙表里那条走廊已经
            # 不作数了（闸门往往只开几拍，先冲过去再说）。
            if (y, x) in forbidden or (
                    (y, x) in self._nav_walls and not self._nav_blind
                    and int(cv[y, x]) not in self._nav_rules["movable_colors"]):
                return False
            v = int(cv[y, x])
            if v in self._nav_floor or (y, x) in own:
                touch += 1
                continue
            if allow_goal and (y, x) in goal_set:
                touch += 1
                continue
        if not self._nav_floor:
            # 这局学不出"地板色"（墙和空地同色，比如 ls20 的墙阵）：只能纯靠
            # 撞出来的墙名单硬绕，除墙之外一律乐观放行
            return True
        return touch > 0

    def _nav_find_target(self, cv, piece=None, pat=None):
        """目标 = 长得最像 avatar 的那块静止的东西。

        优先级：形状同/镜像 > 同色 > 尺寸相近。会动的（NPC/计时条）和 avatar
        自己除外——它们不是"要到达的地方"。目标层（GoalModel）经常被 HUD、
        计分板这类静止小块骗走，所以导航层自己认目标，目标层只做兜底。

        两条额外判据（都是"到了也白到"的教训）：
        - 位置粘性：目标不会动，认过一次就按位置找，别因为自己变形（携带的
          方块被机关改掉）就突然认不出终点；
        - 贫瘠排除：到达也不过关的外观不再当目标，否则会一遍遍走到同一个点。
        """
        if piece is not None:
            base = frozenset((y - piece.at[0], x - piece.at[1])
                             for y, x in piece.cells)
            cols = {piece.colour}
            my_size = piece.size
            self_cells = set(piece.cells)
        elif pat:
            base = frozenset((dy, dx) for dy, dx, _ in pat)
            cols = {c for _, _, c in pat}
            my_size = len(base)
            self_cells = {(self._nav_anchor[0] + dy, self._nav_anchor[1] + dx)
                          for dy, dx, _ in pat} if self._nav_anchor else set()
        else:
            return None
        # 自己的当前外观（多色 sprite 也可能正带着被改造过的方块）
        my_pat = self._nav_pat_of(piece.cells, cv) if piece is not None else pat
        shapes = set()
        for fy in (1, -1):
            for fx in (1, -1):
                shapes.add(frozenset((fy * y, fx * x) for y, x in base))
        bw = max(p[1] for p in base) - min(p[1] for p in base) + 1
        bh = max(p[0] for p in base) - min(p[0] for p in base) + 1
        best, best_key = None, None
        h, w = cv.shape
        for p in self._nav_pieces(cv):
            if set(p.cells) & self_cells:
                continue  # 与 avatar 重叠：那是它自己的一部分
            # 贴边的块多半是 HUD：状态预览、步数条、计分板都钉在画面边缘。
            # 它们往往和 avatar 同色同尺寸，光靠"长得像"会被骗走。
            # 判据要留出包住它的那圈边框，所以阈值放宽到 4 格。
            if (p.at[0] <= 4 or p.at[1] <= 4
                    or p.at[2] >= h - 5 or p.at[3] >= w - 5):
                continue
            if self._nav_spawn is not None and p.at[:2] == self._nav_spawn:
                continue  # 出生点上的是"出生标记/回放幽灵"，不是要去的地方
            if (p.colour, p.size, p.at[:2]) in self._nav_movers:
                continue  # 移动过的实例不是目标
            if self._nav_switch and (set(p.cells) & self._nav_switch):
                continue  # 已经认出来是开关的东西是"工具"，不是终点
            if self._nav_pat_barren(self._nav_pat_of(p.cells, cv)):
                # 到了也白到的外观：除非自身外观已经和它对上（改造成功了，
                # 值得再试一次），否则不再把它当终点
                if not (my_pat and self._nav_pat_same(my_pat,
                                                      self._nav_pat_of(p.cells, cv))):
                    continue
            same_shape = frozenset(
                (y - p.at[0], x - p.at[1]) for y, x in p.cells) in shapes
            same_colour = p.colour in cols
            same_box = (p.at[2] - p.at[0] + 1 == bh and p.at[3] - p.at[1] + 1 == bw)
            if not (same_shape or same_colour or same_box):
                continue
            # 位置粘性：认过一次的目标不会长脚，靠近记忆位置的就该加分——
            # 否则自己一变形（携带的方块被机关改掉）就会突然认不出终点
            sticky = 0
            if self._nav_tgt_pos is not None:
                sticky = 1 if (abs(p.at[0] - self._nav_tgt_pos[0])
                               + abs(p.at[1] - self._nav_tgt_pos[1]) <= 3) else 0
            key = (2 if same_shape else 0, 2 if same_box else 0,
                   1 if same_colour else 0, sticky,
                   -abs(p.size - my_size), p.size)
            if best_key is None or key > best_key:
                best, best_key = p, key
        if best is not None:
            # 认定了就记住位置：下一步即使自己变形也还认得它
            self._nav_tgt_pos = best.at[:2]
        return best

    def _nav_gate_win_len(self) -> int:
        """闸门窗口该给几拍：默认 16，之后用实测的窗口长度中位数自适应。"""
        if not self._nav_gate_wins:
            return 16
        ws = sorted(self._nav_gate_wins[-5:])
        return max(8, min(40, ws[len(ws) // 2]))

    def _nav_dbg(self, msg):
        """导航层的限频诊断（SELFLEARN_DEBUG=1 才输出）。"""
        if os.environ.get("SELFLEARN_DEBUG") != "1":
            return
        if self.action_counter - self._nav_log_at < 20:
            return
        self._nav_log_at = self.action_counter
        print(f"[learn] nav {msg}", flush=True)

    def _nav_next_action(self, grid):
        """BFS 最短路把 avatar 平移到目标锚点。返回 None = 本层没有可执行计划。"""
        if self._nav_gaveup:
            return None
        cv = np.asarray(grid)
        if cv.ndim != 2 or cv.size == 0:
            return None
        avail = self._last_available_actions or list(KEYS)
        # 闸门窗口：门刚被机关打开，旧墙表里那条走廊已经不作数了
        self._nav_blind = self.action_counter < self._nav_gate_until
        if self._nav_blind and self.action_counter - self._nav_log_at >= 20:
            self._nav_log_at = self.action_counter
            print(f"[learn] nav 闸门窗口内 (剩 {self._nav_gate_until - self.action_counter} 拍)"
                  f": 忽略墙表直冲终点", flush=True)

        # ---- avatar 未知：把每个方向键各按一次，攒位移语义 ----
        if self._nav_sig is None:
            for a in KEYS:
                if a in avail and a not in self._nav_pressed:
                    self._nav_pressed.add(a)
                    return a
            # 证据不够就先让位，下一拍观测攒够了再选（识别是渐进的，别一次判死）
            self._nav_sig = self._nav_pick_avatar()
            if self._nav_sig is None:
                return None

        try:
            bg = int(background_color(cv))
        except Exception:
            bg = 0
        # 跟踪用预测锚点（防同签名幽灵块抢目标）
        pred = self._nav_anchor
        if pred is not None and getattr(self, "_generic_acted", None) in KEYS:
            dm = self._nav_mode(self._nav_sig, self._generic_acted)
            if dm:
                pred = (pred[0] + dm[0], pred[1] + dm[1])
        piece = None
        if self._nav_composite and self._nav_pat is not None and self._nav_anchor:
            # 复合实体：observe 已经拼好整体，直接用
            anchor = self._nav_anchor
        else:
            piece = self._nav_piece(cv, self._nav_sig, pred)
            if piece is None or self._nav_shape is None:
                return None
            anchor = piece.at[:2]
        if self._nav_spawn is None:
            self._nav_spawn = anchor  # 本关第一次看到 avatar 的位置 = 出生点
        # 到过的候选锚点（没过关）：本局内不再作为目标
        if self._nav_target is not None and tuple(anchor) == tuple(self._nav_target):
            self._nav_tried.add(self._nav_target)
            # 走到了却没升级 = "到了也白到"。把目标外观记成贫瘠：之后不再把它
            # 当终点（否则会一遍遍走到同一个点），转而去改造自身外观。
            # 这是"执行前预测"学到的第一课——终点要的不只是位置。
            # 只到一次就判"贫瘠"太急：有些关卡第一次到终点确实不过关（要先把
            # 机关打开），第二次才成。攒够 2 次再下结论，避免一开局就把真终点拉黑。
            _tk = tuple(anchor)
            self._nav_arrive[_tk] = self._nav_arrive.get(_tk, 0) + 1
            if self._nav_target_pat is not None and self._nav_arrive[_tk] >= 2:
                for f in self._nav_flips(self._nav_target_pat):
                    self._nav_rules["barren"].add(self._nav_pat_key(f))
                print(f"[learn] nav 贫瘠目标 at={_tk}: 两次到达都不过关，"
                      f"转去改造自身 (barren={len(self._nav_rules['barren'])}种)",
                      flush=True)
                self._nav_target_pat = None
            self._nav_target = None
        self._nav_anchor = anchor
        # own = avatar 自己的格子 + bbox 里的非虚空格（环内的"芯"跟着一起走）
        h, w = cv.shape
        if piece is not None:
            self._nav_shape = frozenset(
                (y - piece.at[0], x - piece.at[1]) for y, x in piece.cells)
            y0, x0, y1, x1 = piece.at
            own = set(piece.cells)
        else:
            own = {(anchor[0] + dy, anchor[1] + dx) for dy, dx, _ in self._nav_pat}
            y0 = min(p[0] for p in own)
            x0 = min(p[1] for p in own)
            y1 = max(p[0] for p in own)
            x1 = max(p[1] for p in own)
        for y in range(max(0, y0), min(h, y1 + 1)):
            for x in range(max(0, x0), min(w, x1 + 1)):
                if int(cv[y, x]) != bg:
                    own.add((y, x))
        # 地板颜色：avatar 站过的地方就是可走的；出生时它嵌在地板里，所以
        # bbox 外扩 2 格的众数色（除去虚空与自身）也是地板。
        # 注意背景色不等于地板：很多盘的黑背景是"坑"，踩上去才是墙外。
        self._nav_floor.update(int(cv[y, x]) for (y, x) in own)
        body = piece.cells if piece is not None else own
        non_self = self._nav_floor - {int(cv[y, x]) for (y, x) in body if 0 <= y < h and 0 <= x < w}
        if not non_self:
            cnt = {}
            for y in range(max(0, y0 - 2), min(h, y1 + 3)):
                for x in range(max(0, x0 - 2), min(w, x1 + 3)):
                    v = int(cv[y, x])
                    if v != bg and (y, x) not in own:
                        cnt[v] = cnt.get(v, 0) + 1
            if cnt:
                self._nav_floor.add(max(cnt.items(), key=lambda kv: kv[1])[0])
            else:
                self._nav_floor.add(bg)
        # 可走颜色是"这个游戏"的机制，累计进跨关知识（换关时当先验播下去）
        self._nav_rules["floor"] |= self._nav_floor
        # 正踩着机关：记下这个锚点，别一遍遍踩同一块（记满就清，允许回访）
        if self._nav_rules["xf_colors"]:
            for (y, x) in own:
                if int(cv[y, x]) in self._nav_rules["xf_colors"]:
                    if len(self._nav_xf_tried) > 12:
                        self._nav_xf_tried.clear()
                    self._nav_xf_tried.add(tuple(anchor))
                    break

        # 位移边：每键的众数位移（非零）
        disp = {}
        for a in KEYS:
            if a not in avail:
                continue
            d = self._nav_dir(a) or self._nav_mode(self._nav_sig, a)
            if d and d != (0, 0):
                disp[a] = d
        if not disp:
            self._nav_dbg(f"no disp at={anchor} "
                          f"votes={self._nav_votes.get(self._nav_sig)}")
            return None

        # ---- 脚下是开关：先把非方向键试一遍 ----
        # 这类机关常常"踩住有效、一离开就复原"，而方向键每次都会离开；能替你把
        # 效果延续下去的往往只有功能键（回放/分身/传送/锁定）。所以一旦发现脚下
        # 是开关，优先试非方向键，而不是急着回去走迷宫。
        # 每个键整关只试一次：功能键常有"改写历史"的副作用（回卷），连着按会把
        # 刚录好的东西冲掉，反而更糟。
        foot = {(anchor[0] + oy, anchor[1] + ox) for (oy, ox) in self._nav_shape}
        if foot & self._nav_switch:
            for a in avail:
                if a in KEYS or a in self._nav_fn_tried:
                    continue
                if not _is_simple_action(a):
                    continue  # 点击类动作要带坐标，不能瞎试
                if a in self._nav_death.get(tuple(anchor), ()):
                    continue
                self._nav_fn_tried[a] = anchor
                print(f"[learn] nav 站在开关上 -> 试 {a} at={anchor}", flush=True)
                return a

        # 会动的东西（NPC/计时条等有位移记录的非 avatar 组件）当前位置不能压
        forbidden = set()
        for sig in self._nav_votes:
            if sig == self._nav_sig:
                continue
            for p2 in self._nav_pieces(cv):
                if (p2.colour, p2.size) == sig:
                    # 排除 avatar 自己（多色 sprite 的碎块也在移动名单里，
                    # 把它们标成禁区就没法动了）
                    forbidden.update(set(p2.cells) - own)

        # ---- 目标：先自己认（长得像 avatar 的静止块），认不出再用目标层 ----
        tgt = self._nav_find_target(cv, piece, self._nav_pat if piece is None else None)
        if tgt is not None:
            ty, tx = tgt.at[:2]
            goal_set = set(tgt.cells)
            tgt_pat = self._nav_pat_of(tgt.cells, cv)
        else:
            g = getattr(self._goal, "goal", None)
            if g is None or not g.cells:
                return None
            ty, tx = g.bbox[0], g.bbox[1]
            goal_set = set(g.cells)
            tgt_pat = None
        my_pat = self._nav_pat if piece is None else self._nav_pat_of(piece.cells, cv)

        # 目标锚点候选：目标 bbox 左上角 ±3（对齐偏移每局不同；位移是格点化的，
        # ±3 共 7 个连续偏移必覆盖任意步长的余数，否则永远差一格够不着）
        tset = {(ty + dy, tx + dx)
                for dy in (-3, -2, -1, 0, 1, 2, 3)
                for dx in (-3, -2, -1, 0, 1, 2, 3)
                if (ty + dy, tx + dx) not in self._nav_tried}
        h, w = cv.shape
        tset = {p for p in tset if 0 <= p[0] < h and 0 <= p[1] < w}
        if not tset:
            return None
        tset.discard(tuple(anchor))
        if not tset:
            return None

        # ---- BFS（最短路）；边按"已知致死/空转"过滤（先验式拦截）----
        start = tuple(anchor)

        # ---- 闸门窗口内：不信墙表，贪心直冲 ----
        # 门只开几拍，没时间绕路。按"离目标更远的那个轴先走"，撞了就换轴
        # （撞过的轴在本次窗口内不再试）。这是"先冲、撞了再学"的顺序，
        # 和平时"先学再走"相反——因为这次机会是有时限的。
        if self._nav_blind:
            # 优先抢"刚打开、而且比我们更靠近终点"的那块门：门开着只在这一侧
            # 有用，绕到远端去没意义。
            _dstart = min(abs(start[0] - g[0]) + abs(start[1] - g[1])
                          for g in tset)
            _doors = [c for c in self._nav_gate_cells
                      if min(abs(c[0] - g[0]) + abs(c[1] - g[1])
                             for g in tset) < _dstart]
            if _doors:
                _tgt0 = min(_doors, key=lambda c: (abs(c[0] - start[0])
                                                   + abs(c[1] - start[1]), c))
            else:
                _tgt0 = min(tset, key=lambda p: (abs(p[0] - start[0])
                                                 + abs(p[1] - start[1]), p))
            _dy = _tgt0[0] - start[0]
            _dx = _tgt0[1] - start[1]
            _axes = ["v", "h"] if abs(_dy) >= abs(_dx) else ["h", "v"]
            # 走廊里要一直走同一个轴：只要能走就继续，撞了才换。否则每拍都按
            # "哪边剩得多"重新选轴，会把一条直走廊走成锯齿（实测就卡在这）。
            if self._nav_gate_axis and self._nav_gate_axis not in self._nav_gate_dead:
                _axes = [self._nav_gate_axis] + [a for a in _axes
                                                 if a != self._nav_gate_axis]
            for _ax in _axes:
                if _ax in self._nav_gate_dead:
                    continue
                _best = None
                for _a, _d in disp.items():
                    if _ax == "v":
                        if _d[0] == 0 or (_d[0] > 0) != (_dy > 0):
                            continue
                    else:
                        if _d[1] == 0 or (_d[1] > 0) != (_dx > 0):
                            continue
                    _np = (start[0] + _d[0], start[1] + _d[1])
                    if not (0 <= _np[0] < h and 0 <= _np[1] < w):
                        continue
                    _best = _a
                    break
                if _best is not None:
                    self._nav_gate_axis = _ax
                    return _best
        prevmap, found = self._nav_bfs(cv, start, disp, own, forbidden,
                                       goal_set, tset, h, w)
        if found is not None and found in tset:
            self._nav_ftemp = set()  # 主路通了，中转点作废

        # ---- 执行前预测：光"走到"可能不够 ----
        # 走到目标却没过关（贫瘠），或者根本无路 + 自身外观和目标显示的不一致，
        # 都说明终点要求的不只是位置，还有"你当时长什么样"。判据纯像素：比较
        # 自己携带的图案和目标那块显示的图案。没有机关色可试就自然落到下面的
        # 前沿探索，不会比现在更差。
        # 但"预测"必须先有事实垫底：没真的到过终点（本关 _nav_tried 为空、
        # 也没有贫瘠记录）时，"图案不符"只是猜测，不能据此改道——否则一开局
        # 就绕开真终点，去踩机关踩到死。所以修完外观这条路要等"到过"再说。
        _hunt_ready = (bool(self._nav_tried) or bool(self._nav_rules["barren"])) \
            and self.action_counter >= self._nav_open_until
        if _hunt_ready and (found is None or self._nav_pat_barren(tgt_pat)) \
                and my_pat and tgt_pat and not self._nav_pat_same(my_pat, tgt_pat):
            xset, xcells = self._nav_xf_targets(cv, own, start)
            if xset:
                pm2, f2 = self._nav_bfs(cv, start, disp, own, forbidden,
                                        xcells, xset, h, w)
                if f2 is not None:
                    path = self._nav_backpath(pm2, f2)
                    if path:
                        print(f"[learn] nav 状态狩猎: 自身{len(my_pat)}格 vs "
                              f"目标{len(tgt_pat)}格图案不符 -> 先踩机关 {f2}",
                              flush=True)
                        return path[0]

        if found is None:
            # ---- 前沿探索：朝"能走到的、离目标最近的那一格"挤 ----
            # 不预设地图长什么样：BFS 把能站的锚点全枚举出来了，取离目标最近的
            # 一个当临时目标走过去；走到还没路就换下一个。这样即使真目标被机关
            # 挡着，agent 也会一路贴着墙往目标方向探，而不是原地打转。
            tgt_f = self._nav_front_tgt
            if tgt_f is not None and (tgt_f not in prevmap or tgt_f == start):
                self._nav_frontier.add(tgt_f)  # 走到了 / 已经走不到了，换一个
                tgt_f = None
            if tgt_f is None and prevmap:
                cands = [p for p in prevmap
                         if p not in self._nav_frontier and p != start]
                if cands:
                    tgt_f = min(cands, key=lambda p: (
                        min(abs(p[0] - t[0]) + abs(p[1] - t[1]) for t in tset), p))
            self._nav_front_tgt = tgt_f
            if tgt_f is not None:
                path = self._nav_backpath(prevmap, tgt_f)
                if path:
                    self._nav_dbg(f"探路 at={anchor} -> {tgt_f} n={len(path)} "
                                  f"walls={len(self._nav_walls)} "
                                  f"frontier={len(self._nav_frontier)}")
                    return path[0]
            # 可达区域全探遍了还是没路：试探"功能键"（这类盘常有机关/回放键）。
            # 连续按两次很关键：第一次往往是回卷（清历史），第二次才触发回放。
            # 跨关记住的"能让地图大变"的键排在最前面——它已经被验证有用。
            last_act = getattr(self, "_generic_acted", None)
            # 先真的把图探过一遍再动用功能键：刚开局可达锚点很少，"探遍"是假信号
            explored = len(self._nav_frontier) >= 4
            cands = [a for a in avail
                     if a not in KEYS and _is_simple_action(a)]
            cands.sort(key=lambda a: (0 if a in self._nav_rules["fn_map"] else 1, a))
            pick = None
            if explored and self._nav_a5_tries < 6 \
                    and self.action_counter >= self._nav_open_until:
                for a in cands:
                    if a in self._nav_death.get(start, ()):
                        continue
                    if last_act == a or self.action_counter - self._nav_a5_at >= 20:
                        pick = a
                        break
            if pick is not None:
                self._nav_a5_tries += 1
                self._nav_a5_at = self.action_counter
                print(f"[learn] nav: 全图探遍无路，试探 {pick} "
                      f"(#{self._nav_a5_tries})", flush=True)
                return pick
            self._nav_dbg(f"miss at={anchor} tgt=({ty},{tx}) "
                          f"disp={disp} walls={len(self._nav_walls)} "
                          f"tried={len(self._nav_tried)} floor={self._nav_floor}")
            return None
        path = self._nav_backpath(prevmap, found)
        if found in tset:
            self._nav_target = found
            self._nav_target_pat = tgt_pat
        elif len(path) <= 1:
            self._nav_ftemp.add(found)  # 中转点走到了也没用，拉黑
        if self.action_counter - self._nav_log_at >= 20:
            self._nav_log_at = self.action_counter
            print(f"[learn] nav sig={self._nav_sig} at={anchor} -> {found} "
                  f"n={len(path)} walls={len(self._nav_walls)} "
                  f"tried={len(self._nav_tried)} ftemp={len(self._nav_ftemp)}",
                  flush=True)
        return path[0]

    @staticmethod
    def _nav_backpath(prevmap, node):
        """把 BFS 的前驱表翻成动作序列。"""
        path = []
        p = node
        while prevmap.get(p) is not None:
            p, a = prevmap[p]
            path.append(a)
        path.reverse()
        return path

    def _nav_bfs(self, cv, start, disp, own, forbidden, goal_set, tset, h, w):
        """以"每键位移"为边的 BFS，返回 (前驱表, 命中点)。

        先验式拦截（冠军 sandbox 里 KnownNoOp/KnownDeath 的无模型版）：
        - 这个锚点上已知致死的动作，不走；
        - 这个 (锚点, 动作) 空转过两次以上且没过期，不走。
        墙名单、会动的组件照旧由 can_stand 管。
        """
        prevmap = {start: None}
        q = deque([start])
        found = None
        dead0 = self._nav_death.get(start, ())
        while q:
            p = q.popleft()
            dead = self._nav_death.get(p)
            for a, d in disp.items():
                if p is start and a in dead0:
                    continue
                if dead and a in dead:
                    continue
                rec = self._nav_noop.get((p, a))
                if rec and rec[0] >= 2 and self.action_counter - rec[1] <= 40:
                    continue
                np_ = (p[0] + d[0], p[1] + d[1])
                if np_ in prevmap or not (0 <= np_[0] < h and 0 <= np_[1] < w):
                    continue
                if not self._nav_can_stand(cv, self._nav_shape, np_, own,
                                           forbidden, goal_set,
                                           allow_goal=np_ in tset):
                    continue
                prevmap[np_] = (p, a)
                if np_ in tset:
                    found = np_
                    q.clear()
                    break
                q.append(np_)
            if found is not None:
                break
        return prevmap, found

    def _nav_xf_targets(self, cv, own, start, limit=800):
        """机关锚点集合：站上去能改造自身外观的位置。

        机关色是从"踩上去图案变了"归纳出来的（变形走迷宫类游戏地面的机器）。
        锚点 = avatar 形状覆盖到该色的某个偏移；已踩过的不再算。没有机关色
        就返回空集，调用方自然落到前沿探索。
        """
        xf = self._nav_rules["xf_colors"]
        if not xf or not self._nav_shape:
            return set(), set()
        h, w = cv.shape
        cells = set()
        for y in range(h):
            for x in range(w):
                if int(cv[y, x]) in xf:
                    cells.add((y, x))
        if not cells:
            return set(), set()
        anchors = set()
        for (cy, cx) in cells:
            for (oy, ox) in self._nav_shape:
                p = (cy - oy, cx - ox)
                if not (0 <= p[0] < h and 0 <= p[1] < w):
                    continue
                if p in self._nav_xf_tried:
                    continue
                anchors.add(p)
        if len(anchors) > limit:  # 太多了只留最近的，别把 BFS 撑爆
            anchors = set(sorted(anchors, key=lambda p: (
                abs(p[0] - start[0]) + abs(p[1] - start[1])))[:limit])
        return anchors, cells

    def _remember_generic(self, fp, a):
        """Bookkeeping for one decision: what we pressed, where, and how often."""
        self._generic_fp = fp
        self._generic_last = a
        self._generic_tried.setdefault(fp, set()).add(a)
        self._graph_prev = (fp, a)
        try:
            self._planner.note_try(fp, a)
        except Exception:
            pass

    def _materialise(self, action, grid, fp):
        """Turn a graph action into one the engine can run.

        The graph stores primitives without coordinates; a click still needs a target,
        so pick the first object centre this board has not been clicked on yet.
        """
        if action != "ACTION6":
            return action
        tried = self._generic_tried.get(fp, set())
        for r, c in self._click_targets(grid):
            cand = f"ACTION6@{r},{c}"
            if cand not in tried:
                return cand
        return None

    # ---------- BFS 搜索 ----------
    def _build_search(self, keys):
        print("[learn] _build_search start", flush=True)
        start = np.asarray(self._start_grid)
        prod = [a for a in keys if self._kin.kin.get(a)]
        if not prod:
            prod = list(keys)
        s0 = _score_target(start, self._target_mask, self._movable_colors)
        print(f"[learn] _build_search score0={s0:.2f} prod={prod}", flush=True)
        heap = [(-s0, 0, start.tolist(), [])]
        visited = {_canon(start)}
        expanded = 0
        results = []
        LIMIT = int(os.environ.get("SELFLEARN_BFS_NODES", "2000"))
        while heap and expanded < LIMIT:
            neg_s, depth, grid_list, path = heapq.heappop(heap)
            expanded += 1
            if expanded % 200 == 0:
                print(f"[learn] bfs expanded={expanded} heap={len(heap)}", flush=True)
            results.append((-neg_s, path))
            if depth >= 20:
                continue
            garr = np.asarray(grid_list)
            for a in prod:
                kin = self._kin.kin.get(a)
                if not kin:
                    continue
                g2 = KinModel._apply(garr, a, kin, background_color(garr))
                if g2 == grid_list:
                    continue
                cc = _canon(np.asarray(g2))
                if cc in visited:
                    continue
                visited.add(cc)
                sc = _score_target(np.asarray(g2), self._target_mask, self._movable_colors)
                heapq.heappush(heap, (-sc, depth + 1, g2, path + [a]))
        results.sort(key=lambda x: -x[0])
        seen = set()
        cands = []
        for sc, p in results:
            t = tuple(p)
            if t in seen:
                continue
            seen.add(t)
            cands.append(p)
        c = _canon(start)
        self._level_search[c] = [cands, 0, list(prod), self._kin]
        print(
            f"[learn] BFS done: {len(cands)} paths (best target-score "
            f"{results[0][0] if results else 0:.1f}) for {c}",
            flush=True,
        )

    def _start_next_candidate(self, cur):
        print(f"[learn] _start_next_candidate dirty={self._dirty} phase={self._phase} plan_set={bool(self._plan)}", flush=True)
        try:
            return self._start_next_candidate_inner(cur)
        except Exception as e:  # noqa: BLE001
            # 像素回退路径内部出错也不能让整轮崩掉
            print(f"[learn] candidate path err: {type(e).__name__}: {e}", flush=True)
            self._plan = []
            return self._act("ACTION5")

    def _start_next_candidate_inner(self, cur):
        cur_grid = _grid(cur) if hasattr(cur, "frame") else np.asarray(cur)
        c = _canon(np.asarray(self._start_grid)) if self._start_grid is not None else _canon(cur_grid)
        entry = self._level_search.get(c)
        keys = entry[2] if entry else self._keys(cur)
        if entry:
            cands, idx, _, _ = entry
            while idx < len(cands):
                path = cands[idx]
                entry[1] = idx + 1
                # 守卫：首动作已知致死/无效则跳过（Known-Death / No-op）
                if self._guarded(path[0], cur_grid):
                    idx = entry[1]
                    continue
                self._plan = _runs(path)
                if self._plan:
                    self._plan[-1] = (self._plan[-1][0], None)  # 末段推到底
                self._pi = 0
                self._run = 0
                self._flat = 0
                self._attempt += 1
                print(f"[learn] attempt#{self._attempt} BFS {self._plan}", flush=True)
                if self._dirty:
                    self._dirty = False
                    return self._reset_action("cand-reset")
                return self._act(self._plan[0][0])
            # 候选耗尽 -> 暴力兜底
        if not self._brute:
            self._brute = [[(a, None)] for a in keys]
            for a in keys:
                for b in keys:
                    if a == b:
                        continue
                    for k in COUNTS:
                        self._brute.append([(a, k), (b, None)])
            self._brute_idx = 0
        if self._brute_idx < len(self._brute):
            self._plan = list(self._brute[self._brute_idx])
            self._brute_idx += 1
            self._pi = 0
            self._run = 0
            self._flat = 0
            self._attempt += 1
            print(f"[learn] attempt#{self._attempt} BRUTE {self._plan}", flush=True)
            if self._dirty:
                self._dirty = False
                return self._reset_action("cand-reset")
            return self._act(self._plan[0][0])
        # 实在没招：纯 no-op 试探（极少走到）
        return self._act("ACTION5")

    def _next_segment(self, latest: FrameData):
        self._pi += 1
        self._run = 0
        self._flat = 0
        if self._pi < len(self._plan):
            return self._act(self._plan[self._pi][0])
        self._dirty = True
        return self._start_next_candidate(latest)

    # ---------- 归纳 ----------
    def _induction_step(self, cur):
        """从干净初始帧出发，对每个动作连续按 3 次、每次记一帧前后差分，学位移。
        同时累积差分序列用于提取目标掩码/可移动颜色。不发 RESET（否则 levels_completed 回退）。"""
        if not hasattr(self, "_ind_frames"):
            self._ind_frames = []
        if self._ind_i >= len(self._ind):
            return self._finish_induction(cur)
        A = self._ind[self._ind_i]
        if self._ind_phase == 0:
            self._ind_prev = cur
            self._ind_phase = 1
            return self._act(A)
        self._kin.observe(self._ind_prev, A, cur)
        self._erl.observe(self._ind_prev.tolist(), A, cur.tolist())
        self._ind_frames.append((self._ind_prev, cur))
        self._ind_obs += 1
        self._ind_prev = cur
        if self._ind_obs < 3:
            return self._act(A)
        self._ind_i += 1
        self._ind_phase = 0
        self._ind_obs = 0
        return self._induction_step(cur)

    def _finish_induction(self, cur):
        self._erl.fit()
        keys = self._ind
        prod = [a for a in keys if self._kin.kin.get(a)]
        noop = set(self._erl.stats.get("noop", {}))
        if not prod and noop:
            prod = [a for a in keys if a not in noop] or keys
        print("[learn] === champion EffectRuleLearner 学到的规则 ===", flush=True)
        print(self._erl.rules_prompt()[:1200], flush=True)
        self._extract_target(getattr(self, "_ind_frames", []))
        print(
            f"[learn] induction done: prod={prod} kin="
            f"{ {a: list(self._kin.kin[a].values()) for a in prod} }",
            flush=True,
        )
        self._build_search(keys)
        self._phase = "PLAN"
        self._dirty = True
        return self._start_next_candidate(cur)

    # ---------- 框架接口 ----------
    def _time_deadline(self) -> Optional[float]:
        """墙钟截止时刻；None = 本局不限时。"""
        d = getattr(self, "time_deadline", None)
        if d:
            return float(d)
        sec = float(getattr(self, "TIME_LIMIT_SEC", 0) or 0)
        if sec > 0:
            t0 = getattr(self, "timer", None)
            if t0:
                return float(t0) + sec
        return None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state == GameState.WIN:
            # 最后一关没有"升级"事件，方案在 WIN 时补存
            if self._pending_mem is not None:
                ml, mk, macts, mstrat = self._pending_mem
                self._mem.save_plan(ml, mk, macts, mstrat)
                self._pending_mem = None
            print(f"[learn] WIN - {self._mem.summary()}", flush=True)
            self._mem.flush()
            return True
        deadline = self._time_deadline()
        if deadline is not None and time.time() >= deadline:
            if not self._deadline_logged:
                self._deadline_logged = True
                self._deadline_hit = True
                print(f"[learn] TIME-LIMIT hit game={self.game_id} after "
                      f"{self.action_counter} actions", flush=True)
            return True
        return False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        cur = _grid(latest_frame)
        # 缓存可用动作集：离线像素回退路径拿不到 FrameData 时用它兜底
        _aa = getattr(latest_frame, "available_actions", None)
        if _aa:
            # 统一存 "ACTIONn" 字符串：规则语法的 trigger 与 _act() 都要这个名字
            self._last_available_actions = [f"ACTION{_act_id(a)}" for a in _aa if _act_id(a)]
        prev = self._last_grid
        self._last_grid = cur.copy() if cur.size else cur
        changed = prev is None or _changed(prev, cur)
        lv = latest_frame.levels_completed

        # ---- 通用层：结构化观测（关卡切换时先重置，避免跨关污染规则）----
        if self._use_generic and self._perc is not None:
            if lv != self._level:
                # 过关先记功（此时指纹还是上一关的），再重置到新关卡
                if self._generic_last is not None and lv > self._level:
                    self._generic_feedback("progress")
                # 复盘必须在重置之前写：目标掩码、键语义、卡点此刻都还属于上一关，
                # 一旦 reset_level 就全没了。
                if self._level >= 0 and lv > self._level:
                    self._write_note("cleared")
                self._perc.reset_level(lv)
                self._planner = _RulePlanner(self._perc)
                _acts = tuple(self._last_available_actions
                              or ("ACTION1", "ACTION2", "ACTION3", "ACTION4"))
                self._graph = _StateGraph(_acts) if self._use_graph else None
                self._graph_prev = None
                # 目标层跟着关卡重置：目标掩码必须取自"还没被覆盖"的第一帧，
                # 否则第一格已经被自己压住，覆盖分永远差一格。
                if self._use_goal:
                    self._goal = _GoalModel()
                    self._gpol = _GoalPolicy(self._goal)
                self._gplan, self._gvec, self._gwarm = [], None, []
                self._gsampled, self._gblocked = set(), set()
                self._gcov_prev = 0.0
                # 仲裁状态跟着关卡重置（死亡重开不算换关，所以"让位"会延续）
                self._goal_acted = False
                self._goal_anchor_prev = None
                self._goal_shadow = False
                self._goal_best_cov = -1.0
                self._goal_prog_at = self.action_counter
                self._goal_yield_until = -1
                self._goal_deaths = 0
                self._level_t0 = self.action_counter
                self._note_at = self.action_counter
                # 复用上一关学到的按键语义：同一个游戏里键不会变，没必要重学。
                # 只当先验、不锁死 —— 真按下去若位移不同，observe() 会覆盖它，
                # 所以即便先验是错的，代价也只是几步，远小于每关瞎按一轮。
                if self._gpol is not None:
                    _kind = self._mem.latest_kind(exclude_level=lv)
                    _pre = {}
                    for _k, _v in (_kind or {}).items():
                        if _k in _ACTIONS and len(_v) == 2 and (_v[0] or _v[1]):
                            _pre[_k] = (int(_v[0]), int(_v[1]))
                    if _pre:
                        self._gpol.disp.update(_pre)
                        self._gsampled |= set(_pre)
                        print(f"[note] L{lv} 复用上一关按键语义 "
                              f"{_keys_text(_pre, set())}", flush=True)
                self._generic_last = None
                self._generic_fp = None
                self._generic_tried = {}
                # 新关卡：上一关的动作队列与短期记忆全部作废
                self._llm_queue = []
                self._llm_recent = []
                self._llm_last_outcome = None
                # 导航层：几何知识（墙/到过的锚点）是"这一关"的，机制知识跨关保留。
                # _nav_pat/_nav_sig/_nav_votes/_nav_dir_votes 刻意不清：外观与按键
                # 语义属于"这个游戏"，不属于"这一关"——清了每关都要重学一遍。
                self._nav_walls = set()
                self._nav_wall_hits = {}
                self._nav_floor = set(self._nav_rules["floor"])  # 可走色跨关累计
                self._nav_movers = set()
                self._nav_spawn = None
                self._nav_ftemp = set()
                self._nav_frontier = set()
                self._nav_front_tgt = None
                self._nav_switch = set()
                self._nav_fn_tried = {}
                self._nav_a5_tries = 0
                self._nav_tried = set()
                self._nav_anchor = None
                self._nav_target = None
                self._nav_target_pat = None
                self._nav_tgt_pos = None
                self._nav_prev = None
                self._nav_pat_prev = None
                self._nav_skip = False
                self._nav_gaveup = False
                self._nav_xf_tried = set()
                self._nav_arrive = {}
                self._nav_open_until = -1
                self._nav_gate_until = -1
                self._nav_gate_start = -1
                self._nav_gate_dead = set()
                self._nav_gate_axis = ""
                self._nav_gate_cells = set()
                self._nav_push = set()
                self._nav_chg_hist = []
                self._nav_blind = False
                self._nav_death = {}
                self._nav_noop = {}
                _carry = []
                if self._nav_rules["fn_map"]:
                    _carry.append(f"功能键{sorted(self._nav_rules['fn_map'])}")
                if self._nav_rules["xf_colors"]:
                    _carry.append(f"机关色{sorted(self._nav_rules['xf_colors'])}")
                if self._nav_rules["floor"]:
                    _carry.append(f"地板色{sorted(self._nav_rules['floor'])}")
                if self._nav_rules["barren"]:
                    _carry.append(f"贫瘠目标{len(self._nav_rules['barren'])}种")
                if _carry:
                    print(f"[learn] nav L{lv} 跨关保留: {' '.join(_carry)}", flush=True)
                self._mem.save_rules({k: sorted(v) for k, v in
                                      self._nav_rules.items()})
                if self._llm is not None:
                    print(f"[learn] llm reset for level {lv} ({self._llm.stats()})", flush=True)
            self._generic_observe(cur)
            # 每 80 步留一次快照：就算这关最终过不去，盘上也要有"卡在哪"。
            if self._use_goal and self.action_counter - self._note_at >= 80:
                self._note_at = self.action_counter
                self._write_note("running")

        # ---- 死亡：记录 (状态,动作) 致死，换下一个候选 ----
        if latest_frame.state == GameState.GAME_OVER:
            self._generic_feedback("death")
            self._nav_skip = True  # reset 帧不是动作造成的转移，别进导航观测
            # 先验式拦截：记下"从这个锚点发这个键会死"。之后 BFS 走到这个
            # 锚点时就再也不会提议那个动作（dfranzen 的 KnownDeathActionError）。
            # 只记导航层自己发的动作——别的层发的，锚点未必对得上。
            _da = getattr(self, "_generic_acted", None)
            if (self._nav_acted and _da and self._nav_anchor is not None
                    and _da not in self._nav_death.get(tuple(self._nav_anchor), ())):
                self._nav_death.setdefault(tuple(self._nav_anchor), set()).add(_da)
                print(f"[learn] nav 致死记忆 at={tuple(self._nav_anchor)} "
                      f"act={_da} (之后规划绕开它)", flush=True)
            # 目标层致死：立刻让位一段时间，别让它带着玩家在同一关反复送死
            if not self._nav_acted and getattr(self, "_goal_acted", False):
                self._goal_deaths = getattr(self, "_goal_deaths", 0) + 1
                self._goal_yield_until = self.action_counter + 60
                self._goal_best_cov = -1.0
                print(f"[learn] goal 层致死 #{self._goal_deaths} → 让位 60 步",
                      flush=True)
            # 死了 = 关卡重开、录像带也清空：开关处的功能键值得重新试一次
            self._nav_fn_tried = {}
            self._nav_a5_tries = 0
            self._death_count += 1
            cur_act = self._plan[self._pi][0] if self._pi < len(self._plan) else None
            if cur_act and prev is not None:
                c = _canon(prev)
                self._death.setdefault(c, set()).add(cur_act)
                self._mem.save_guards(self._level, c, {cur_act}, set())
            self._plan = []
            self._dirty = True
            print(
                f"[learn] GAME_OVER death#{self._death_count} danger={self._danger} "
                f"death_memo={sum(len(v) for v in self._death.values())}",
                flush=True,
            )
            return self._reset_action("death")

        # ---- 关卡状态转移 ----
        if lv != self._level:
            if lv > self._level and self._level >= 0 and self._phase == "PLAN" \
                    and self._start_grid is not None:
                self._proven_cache[_canon(np.asarray(self._start_grid))] = list(self._plan)
            print(
                f"[learn] LEVEL UP -> L{lv} (step={self.action_counter}, "
                f"attempt#{self._attempt})",
                flush=True,
            )
            # ---- 过关确认：上一关的方案此刻才算被验证，写入跨运行记忆 ----
            if self._pending_mem is not None and lv > self._level:
                ml, mk, macts, mstrat = self._pending_mem
                self._mem.save_plan(ml, mk, macts, mstrat)
                self._mem_saved_levels.add(ml)
                self._pending_mem = None
            self._level = lv
            # ---- 取回本关的守卫记忆（Known-Death / No-op）----
            gc = _canon(cur)
            gg = self._mem.guards_for(lv, gc)
            if gg:
                self._death.setdefault(gc, set()).update(gg.get("death", []))
                self._noop.setdefault(gc, set()).update(gg.get("noop", []))
            # ---- 结构化求解模式：每关用真实物理模型 A* 在线求解 ----
            # 换关先恢复求解器：上一关解不出来（比如 targets 太多）只说明上一关难，
            # 不该让后面的关卡连试一次的机会都没有。之前这里是整局永久禁用。
            if self._struct_fail:
                print(f"[learn] struct reset for L{lv} (prev fails={self._struct_fail})",
                      flush=True)
            self._struct_fail = 0
            self._use_struct = self._use_struct_base
            if self._use_struct:
                self._struct_cache_key = None
                self._struct_plan = []
                self._struct_sim = None
                self._struct_drift = 0
                self._phase = "PLAN"
                a = self._structured_next_action(latest_frame)
                if a is not None:
                    return a
                # 求解失败：计数，连续多次失败则彻底退回像素自学习，
                # 否则本拍先落回离线路径（不再盲发 ACTION1）
                self._struct_fail = getattr(self, "_struct_fail", 0) + 1
                if self._struct_fail >= 3:
                    self._use_struct = False
                    print("[learn] struct disabled after 3 failures -> generic learner",
                          flush=True)
            # ---- 通用层接管：无专用求解器 / 求解失败时，用归纳出的规则前瞻 ----
            if not self._use_struct:
                ga = self._generic_next_action(cur)
                if ga is not None:
                    self._phase = "PLAN"
                    self._dirty = False
                    return ga
            c = _canon(cur)
            if c in self._proven_cache:
                self._plan = list(self._proven_cache[c])
                self._pi = 0
                self._run = 0
                self._flat = 0
                self._phase = "PLAN"
                self._dirty = False
                print(f"[learn] replay cached plan for L{lv}", flush=True)
                return self._act(self._plan[0][0]) if self._plan else self._act("ACTION1")
            entry = self._level_search.get(c)
            if entry is not None:
                self._start_grid = cur
                self._kin = entry[3]
                self._plan = []
                self._pi = self._run = self._flat = 0
                self._phase = "PLAN"
                self._dirty = False
                print(f"[learn] resume search for L{lv} (idx={entry[1]})", flush=True)
                return self._start_next_candidate(cur)
            self._start_grid = cur
            self._kin = KinModel()
            self._erl = EffectRuleLearner(
                cell_size=1, max_entity_cells=MAX_ENTITY_CELLS,
                min_support=2, min_confidence=0.5, min_lift=1.0,
                use_side_condition=True, max_dist=8.0,
            )
            self._danger.clear()
            self._plan = []
            self._pi = self._run = self._flat = 0
            self._attempt = 0
            self._brute = []
            self._brute_idx = 0
            self._ind = self._keys(latest_frame)
            self._ind_i = 0
            self._ind_phase = 0
            self._ind_obs = 0
            self._ind_prev = None
            self._ind_frames = []
            self._target_mask = set()
            self._movable_colors = set()
            self._phase = "INDUCE"
            return self._induction_step(cur)

        # ---- 无效动作记录（No-op 守卫）----
        if not changed:
            self._generic_feedback("nothing")
        if not changed and self._plan and self._pi < len(self._plan) and prev is not None:
            cur_act = self._plan[self._pi][0]
            c = _canon(prev)
            self._noop.setdefault(c, set()).add(cur_act)
            self._mem.save_guards(self._level, c, set(), {cur_act})

        # ---- 归纳阶段 ----
        if self._phase == "INDUCE":
            return self._induction_step(cur)

        # ---- 规划阶段 ----
        if self._phase == "PLAN" and self._use_struct:
            a = self._structured_next_action(latest_frame)
            if a is not None:
                return a
            # 结构化求解失败 -> 交给通用层
        if self._phase == "PLAN" and not self._use_struct and self._use_generic:
            ga = self._generic_next_action(cur)
            if ga is not None:
                return ga
        if not self._plan or self._pi >= len(self._plan):
            # 计划被清空或下标越界 -> 重新挑候选（而不是 IndexError 崩掉整轮）
            self._pi = 0
            return self._start_next_candidate(cur)

        act, cnt = self._plan[self._pi]
        self._run += 1
        if cnt is not None:
            if self._run >= cnt:
                return self._next_segment(latest_frame)
            return self._act(act)
        if changed:
            self._flat = 0
            return self._act(act)
        self._flat += 1
        if self._flat < 2:
            return self._act(act)
        return self._next_segment(latest_frame)
