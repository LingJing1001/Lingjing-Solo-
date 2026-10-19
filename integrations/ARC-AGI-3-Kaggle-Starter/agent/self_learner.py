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
from collections import defaultdict
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
    _HAVE_LEXICON = True
except Exception:  # pragma: no cover
    _HAVE_LEXICON = False

COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
KEYS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7")
MAX_ENTITY_CELLS = 600
WALL_COLOR = 10  # ar25 中央分隔墙；目标掩码排除它


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
                      f"plans={len(g.get('plans', {}))} guards={len(g.get('guards', {}))}",
                      flush=True)
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

    def summary(self) -> str:
        g = self.data["games"].get(self.game_id, {})
        return (f"memory: plans={len(g.get('plans', {}))} "
                f"guards={len(g.get('guards', {}))} hits={self.hits} saves={self.saves}")


class MyAgent(Agent):
    MAX_ACTIONS: int = int(os.environ.get("SELFLEARN_MAX_ACTIONS", "12000"))

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        self._struct_cache_key = None
        self._struct_plan: list = []
        # ---- 通用层：结构化观测 + 规则归纳（不依赖引擎内部，任何盘可用）----
        self._use_generic = _HAVE_LEXICON and os.environ.get("SELFLEARN_GENERIC", "1") == "1"
        self._perc = _Perception() if _HAVE_LEXICON else None
        self._planner = _RulePlanner(self._perc) if _HAVE_LEXICON else None
        self._generic_last = None      # 上一拍由通用层发出的动作
        self._generic_fp = None        # 上一拍通用层的状态指纹
        self._generic_tried: dict = {}  # 指纹 -> 已试过的动作（规则不足时的有信息试探）
        self._generic_log_at = 0

        # ---- LLM 决策层（冠军的第二半：128K 模型读 code/rules 视图做决策）----
        # 关掉时完全不影响规则前瞻；开启后每 N 拍问一次模型，拿一小段动作队列。
        self._llm = (_LLMPlanner() if (_HAVE_LEXICON and os.environ.get("SELFLEARN_LLM", "0") == "1")
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
        tl = int(os.environ.get("SELFLEARN_SOLVE_TLIMIT", "25"))
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

    def _generic_probe(self, fp):
        """规则还没学到时的有信息试探：优先没试过的、没被判死/无效的动作。"""
        keys = list(getattr(self, "_last_available_actions", None) or KEYS)
        tried = self._generic_tried.get(fp, set())
        for a in keys:
            if a in tried:
                continue
            if (fp, a) in self._planner.dead or (fp, a) in self._planner.noop:
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
            acts = self._llm.decide(
                self._perc, grid, self.action_counter, getattr(self._perc, "level", None),
                allowed=avail or _ACTIONS, recent=self._llm_recent,
                outcome=self._llm_last_outcome, planner=self._planner, fp=fp)
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
            # 模型优先：它读的是同一份 code/rules 视图，但能做多步因果推理。
            acted = self._llm_next_action(grid, fp)
            if acted is not None:
                return acted
            a = self._planner.best_action(grid)
            if a is None:
                a = self._generic_probe(fp)
            if a is None:
                return None
            self._generic_fp = fp
            self._generic_last = a
            self._generic_tried.setdefault(fp, set()).add(a)
            if self.action_counter - self._generic_log_at >= 25:
                self._generic_log_at = self.action_counter
                print("[learn] generic rules:\n" + self._perc.rules_text(), flush=True)
            return self._act(a)
        except Exception as e:  # noqa: BLE001
            print(f"[learn] generic next err: {type(e).__name__}: {e}", flush=True)
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
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state == GameState.WIN:
            # 最后一关没有"升级"事件，方案在 WIN 时补存
            if self._pending_mem is not None:
                ml, mk, macts, mstrat = self._pending_mem
                self._mem.save_plan(ml, mk, macts, mstrat)
                self._pending_mem = None
            print(f"[learn] WIN - {self._mem.summary()}", flush=True)
            self._mem.flush()
        return latest_frame.state == GameState.WIN

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
                self._perc.reset_level(lv)
                self._planner = _RulePlanner(self._perc)
                self._generic_last = None
                self._generic_fp = None
                self._generic_tried = {}
                # 新关卡：上一关的动作队列与短期记忆全部作废
                self._llm_queue = []
                self._llm_recent = []
                self._llm_last_outcome = None
                if self._llm is not None:
                    print(f"[learn] llm reset for level {lv} ({self._llm.stats()})", flush=True)
            self._generic_observe(cur)

        # ---- 死亡：记录 (状态,动作) 致死，换下一个候选 ----
        if latest_frame.state == GameState.GAME_OVER:
            self._generic_feedback("death")
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
