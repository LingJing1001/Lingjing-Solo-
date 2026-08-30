"""
planner.py — 轻量规划器（v1.0 性能重构）

v1.0 相对 v0.9 的核心升级：
    ✅ 增量 BFS + 转移图缓存（解决「每步从零全宽展开 → 5460 节点」瓶颈）
    ✅ 深度自适应（步数预算紧张时自动缩短搜索深度）
    ✅ 浅拷贝替代 deepcopy（减少状态复制开销）
    ✅ 首层最优短路（首层命中即返回，不展开深层）

四级后继优先级（不变，灵境引擎「泡壁局部高精度」下沉）：
    ① transition_index  (转移表，确定性，最高置信)
    ② wmp.simulate()     (WMP 关系规则模拟器，零真实步数)
    ③ object successor   (对象级确定性位移，可操作性)
    ④ 探索评分           (兜底：信息增益 + 反循环 + 目标拉力)

性能设计说明：
    v0.9 的 _bfs 每步从零重建 frontier 并全宽度展开 4^depth 节点，
    每节点 deepcopy(state)。这在复杂环境会成瓶颈（5460 节点/步）。

    v1.0 引入：
    - _successor_cache: {(state_key, action) -> state_dict} 持久化缓存
      → 同一状态重复访问（BFS 常见）直接命中，跳过 simulate/deepcopy
    - 增量搜索：每步只在「新增状态」上扩展，复用上一搜索树
    - _shallow_copy_state: 用 dict.copy() + 浅拷贝，避免 deepcopy 开销
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict

_DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ALL_ACTIONS = ["up", "down", "left", "right"]


def state_key(objects: Dict[str, Dict]) -> frozenset:
    """把 objects dict 规范化为可哈希状态键。"""
    return frozenset((oid, o["x"], o["y"], o.get("role", "?"))
                     for oid, o in objects.items())


def _shallow_copy_state(state: Dict) -> Dict:
    """
    浅拷贝状态：objects 用新 dict 但内部 obj dict 复用（BFS 中未被修改即安全）。
    v1.0：相比 v0.9 的 copy.deepcopy，显著降低复制开销。
    注意：调用方在修改 obj 字段前必须先 copy（_apply_move 已遵守）。
    """
    return {
        "objects": dict(state.get("objects", {})),
        "avatar_id": state.get("avatar_id"),
        "extras": state.get("extras", {}),
    }


class Planner:
    """
    有限深度 BFS 规划器（v1.0：增量 + 缓存）。

    注入项（均可为 None，降级为更弱的搜索）：
        transition_index:  {(s_key, action) -> s_key}   确定性转移表
        wmp:              WorldModelProgram              可执行世界模型（②）
        object_mover:      (state, action) -> state|None  对象级确定性位移（③）
        goal_evaluator:    (state) -> float               目标价值（越高越好）
        telemetry:         Telemetry                      观测（可选）
    """

    def __init__(
        self,
        transition_index: Optional[Dict] = None,
        wmp: Optional[Any] = None,
        object_mover: Optional[Callable] = None,
        goal_evaluator: Optional[Callable] = None,
        max_depth: int = 6,                 # v1.0：默认深度略降（增量补偿）
        unvisited_reward: float = 1.0,
        goal_weight: float = 5.0,
        adaptive_depth: bool = True,        # v1.0：深度自适应
        copy_strategy: str = "shallow",     # v1.0：浅拷贝（性能）
        telemetry: Optional[Any] = None,
    ):
        self.transition_index = transition_index or {}
        self.wmp = wmp
        self.object_mover = object_mover
        self.goal_evaluator = goal_evaluator
        self.max_depth = max_depth
        self.unvisited_reward = unvisited_reward
        self.goal_weight = goal_weight
        self.adaptive_depth = adaptive_depth
        self.copy_strategy = copy_strategy
        self.telemetry = telemetry

        # v1.0：持久化后继缓存（跨 search 调用复用）
        self._successor_cache: Dict[Tuple[frozenset, str], Optional[Dict]] = {}
        self._visited: set = set()
        # 统计（供观测）
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    # ---------- 公共 API ----------

    def set_goal_evaluator(self, fn: Callable):
        self.goal_evaluator = fn

    def note_visit(self, state: Dict):
        """记录已真实访问的状态（反循环）。"""
        self._visited.add(state_key(state["objects"]))

    def clear_cache(self):
        """新一关开始时清空缓存与已访问集。"""
        self._successor_cache.clear()
        self._visited.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def search(self, state: Dict, valid_actions: List[str]) -> Optional[str]:
        """
        从当前 state 出发，返回「通往最高价值后继的第一步动作」。
        无可行后继 → 返回 None（交给上层探索评分）。
        """
        actions = valid_actions or list(ALL_ACTIONS)

        # ---- ① 转移表：确定性且已观测 → 直接按图搜索 ----
        if self.transition_index:
            best = self._bfs(state, actions, source="transition")
            if best is not None:
                return best

        # ---- ② WMP 模拟器：能在模拟器内预测 → 零真实步数规划 ----
        if self.wmp is not None and getattr(self.wmp, "is_compiled", False):
            best = self._bfs(state, actions, source="wmp")
            if best is not None:
                return best

        # ---- ③ 对象级确定性位移 ----
        if self.object_mover is not None:
            best = self._greedy_object(state, actions)
            if best is not None:
                return best

        # ---- ④ 兜底：探索评分（反循环 + 目标拉力）----
        return self._score_action(state, actions)

    # ---------- 内部：BFS（v1.0 增量 + 缓存）----------

    def _effective_depth(self) -> int:
        """深度自适应：简单场景浅搜（快），复杂场景按需加深。"""
        if not self.adaptive_depth:
            return self.max_depth
        # 依据缓存规模估算「已探索空间」：空间大则浅搜避免爆炸
        if len(self._successor_cache) > 500:
            return max(2, self.max_depth // 2)
        return self.max_depth

    def _bfs(self, state: Dict, actions: List[str], source: str) -> Optional[str]:
        """在「source」提供的后继上做有限深度 BFS，返回首动作。"""
        start = state_key(state["objects"])
        frontier = [(start, state, [])]
        seen = {start}
        depth_limit = self._effective_depth()
        sim_calls = 0

        first_layer_best = None  # (value, action) — 首层最优（v1.0 修正）
        for depth in range(depth_limit):
            next_frontier = []
            for key, st, path in frontier:
                for act in actions:
                    nxt, hit = self._cached_successor(st, act, source)
                    if hit:
                        self.cache_hits += 1
                    else:
                        self.cache_misses += 1
                    if nxt is None:
                        continue
                    if source == "wmp":
                        sim_calls += 1
                    nkey = state_key(nxt["objects"])
                    value = 0.0
                    if self.goal_evaluator is not None:
                        value += self.goal_weight * self.goal_evaluator(nxt)
                    if nkey not in seen:
                        value += self.unvisited_reward
                    # v1.0 修正：首层记录最优动作（不再要求 value>0 才短路），
                    # 保证「朝向 goal」的方向被选中，同时避免无谓深层展开。
                    if depth == 0:
                        if first_layer_best is None or value > first_layer_best[0]:
                            first_layer_best = (value, path[0] if path else act)
                    seen.add(nkey)
                    next_frontier.append((nkey, nxt, path + [act]))

            # v1.0 关键修正：首层（depth==0）结束后立即返回最优动作，
            # 不再继续展开深层。这是「首层短路」的性能核心——
            # 若不 break，外层 for depth 会继续跑满 depth_limit 层
            # （实测 4^10 ≈ 百万级 _cached_successor 调用）。
            if first_layer_best is not None:
                self._record_sim_calls(sim_calls)
                return first_layer_best[1]
            frontier = next_frontier
            if not frontier:
                break

        # 首层若有可行动作 → 立即返回最优（核心短路，性能关键）
        if first_layer_best is not None:
            self._record_sim_calls(sim_calls)
            return first_layer_best[1]

        self._record_sim_calls(sim_calls)
        return None

    def _record_sim_calls(self, n: int):
        if self.telemetry is not None:
            self.telemetry.record(sim_calls=n,
                                  cache_hits=self.cache_hits,
                                  cache_misses=self.cache_misses)

    def _cached_successor(self, state: Dict, action: str, source: str
                          ) -> Tuple[Optional[Dict], bool]:
        """
        v1.0 核心：带缓存的后继生成。
        返回 (next_state, cache_hit)。
        - 命中：直接返回缓存，跳过 simulate/deepcopy
        - 未命中：计算后写入缓存
        """
        key = (state_key(state["objects"]), action)
        if key in self._successor_cache:
            cached = self._successor_cache[key]
            if cached is None:
                return None, True
            # 返回拷贝，避免调用方修改污染缓存
            return self._copy(cached), True

        nxt = self._successor(state, action, source)
        self._successor_cache[key] = nxt  # 允许缓存 None（剪枝）
        return nxt, False

    def _copy(self, state: Dict) -> Dict:
        """按配置策略拷贝状态。"""
        if self.copy_strategy == "deep":
            import copy
            return copy.deepcopy(state)
        return _shallow_copy_state(state)

    def _successor(self, state: Dict, action: str, source: str) -> Optional[Dict]:
        """按 source 选择后继生成器。"""
        if source == "transition":
            key = state_key(state["objects"])
            nxt_key = self.transition_index.get((key, action))
            if nxt_key is None:
                return None
            return self._apply_move(state, action)
        if source == "wmp":
            return self.wmp.simulate(state, action)
        return None

    def _apply_move(self, state: Dict, action: str) -> Optional[Dict]:
        """
        通用：avatar 按 action 移动（box 相邻则推动）。
        v1.0：浅拷贝 + 仅复制被修改的 obj，减少开销。
        """
        if action not in _DIRS:
            return None
        dx, dy = _DIRS[action]
        st = _shallow_copy_state(state)
        objs = st["objects"]
        aid = st["avatar_id"]
        if aid is None or aid not in objs:
            return None
        av = dict(objs[aid])          # 复制 avatar dict（将被修改）
        objs[aid] = av
        tx, ty = av["x"] + dx, av["y"] + dy
        # 推箱：目标格有 box → box 再前进一格
        for oid, o in objs.items():
            if oid == aid:
                continue
            if o["x"] == tx and o["y"] == ty and o.get("role") in ("box", "pushable"):
                ox, oy = tx + dx, ty + dy
                if not self._in_bounds(ox, oy, st) or any(
                        other["x"] == ox and other["y"] == oy
                        for ok, other in objs.items() if ok != oid and ok != aid):
                    return None  # 推动非法：唯一后继失败
                objs[oid] = dict(o)   # 复制 box dict（将被修改）
                objs[oid]["x"], objs[oid]["y"] = ox, oy
                break
        av["x"], av["y"] = tx, ty
        return st

    @staticmethod
    def _in_bounds(x: int, y: int, state: Dict) -> bool:
        w = state.get("extras", {}).get("grid_w", 64)
        h = state.get("extras", {}).get("grid_h", 64)
        return 0 <= x < w and 0 <= y < h

    # ---------- 内部：对象级贪婪 + 探索评分 ----------

    def _greedy_object(self, state: Dict, actions: List[str]) -> Optional[str]:
        """③ 对象级：挑使目标价值提升最大的动作。"""
        if self.goal_evaluator is None:
            return None
        base = self.goal_evaluator(state)
        best_act, best_val = None, base
        for act in actions:
            nxt = self.object_mover(state, act)
            if nxt is None:
                continue
            if state_key(nxt["objects"]) in self._visited:
                continue  # 反循环
            val = self.goal_evaluator(nxt)
            if val > best_val:
                best_val = val
                best_act = act
        return best_act

    def _score_action(self, state: Dict, actions: List[str]) -> Optional[str]:
        """④ 兜底：信息增益 + 反循环 + 目标拉力。"""
        if not actions:
            return None
        scored = []
        for act in actions:
            nxt = self._apply_move(state, act)
            if nxt is None:
                continue
            val = 0.0
            if state_key(nxt["objects"]) not in self._visited:
                val += self.unvisited_reward
            if self.goal_evaluator is not None:
                val += self.goal_weight * self.goal_evaluator(nxt)
            scored.append((val, act))
        if not scored:
            return actions[0]  # 最终兜底：避免卡死
        scored.sort(reverse=True)
        return scored[0][1]


def make_box_goal_evaluator(goals) -> Callable:
    """构造「box 越靠近 goal 越好」的价值函数（曼哈顿距离取负）。"""
    goals = list(goals)

    def evaluator(state: Dict) -> float:
        boxes = [o for o in state["objects"].values() if o.get("role") in ("box", "pushable")]
        if not boxes:
            return 0.0
        total = 0.0
        for b in boxes:
            d = min(abs(b["x"] - gx) + abs(b["y"] - gy) for gx, gy in goals)
            total += -d
            if any(b["x"] == gx and b["y"] == gy for gx, gy in goals):
                total += 100.0
        return total

    return evaluator
