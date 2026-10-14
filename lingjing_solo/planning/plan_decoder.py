"""公共 plan 解码器：统一离线 plan 生成 + 引擎回放验证。

把 re86（位置枚举）和 m0r0（deepcopy BFS）两种破关方法抽象成可插拔策略，
新游戏只需注册策略或用通用策略，不用每次手写一次性脚本。

三层结构
--------
1. **公共接口**（``GameLoader`` / ``is_win`` / ``verify_plan`` / ``do_action``）
   — 所有策略共用，不依赖具体游戏机制。
2. **求解策略**（``SolveStrategy`` 协议）
   — ``DeepcopyBFSStrategy``：通用，搜动作序列（m0r0 类）
   — ``PositionEnumStrategy``：win=图案匹配类，枚举块位置反推移动（re86 类）
   — ``AStarStrategy``：通用 A*（re86_astar 类，reset+重放）
3. **统一入口**（``PlanDecoder``）
   — 加载游戏 → 选策略 → 求解 → 引擎回放验证 → 输出统一 plan

统一 plan 格式
--------------
- 纯键盘：``list[int]``（action id 1-5/7）
- 含点击：``list[tuple[int, int, int]]``（aid=6 点击 (x,y)，aid=1-5 键盘）

用法
----
::

    from lingjing_solo.planning.plan_decoder import PlanDecoder, DeepcopyBFSStrategy

    dec = PlanDecoder("m0r0")
    dec.load()
    plan = dec.solve(0, strategy=DeepcopyBFSStrategy(max_depth=18))
    assert dec.verify(0, plan)   # 引擎回放验证
    plans = dec.solve_all()      # 求解所有关
"""
from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable


# ── 公共接口（不依赖具体游戏）──────────────────────────────────


def load_game(gid: str):
    """用 arc_agi 公共接口加载游戏，返回 (arc, env, game)。

    需先设置 sys.path（调用方负责，或用 PlanDecoder.load()）。
    """
    import arc_agi
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL)
    env = arc.make(gid)
    g = getattr(env, "_game", None)
    if g is None:
        for attr in ("game", "_arcade_game"):
            g = getattr(env, attr, None)
            if g is not None:
                break
    if g is None:
        raise RuntimeError(f"无法获取 {gid} 的 game 对象")
    return arc, env, g


def is_win(g, level: int) -> bool:
    """检查是否过关：WIN 状态或 level_index 增加。"""
    st = str(getattr(g, "_state", ""))
    if "WIN" in st:
        return True
    try:
        return int(getattr(g, "_current_level_index", -1)) > level
    except (TypeError, ValueError):
        return False


def do_action(g, item: Any) -> None:
    """执行一个 plan item。

    - ``int``：键盘动作（ACTION1-5/7）
    - ``(aid, x, y)``：aid=6 点击 (x,y)，aid=1-5 键盘
    """
    from arcengine import ActionInput, GameAction
    if isinstance(item, int):
        g.perform_action(ActionInput(id=GameAction.from_id(item), data={}), raw=True)
    else:
        aid, x, y = item
        if aid == 6:
            g.perform_action(
                ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)}),
                raw=True,
            )
        else:
            g.perform_action(ActionInput(id=GameAction.from_id(aid), data={}), raw=True)


def verify_plan(g, level: int, plan: Sequence) -> bool:
    """引擎回放验证：重置关卡 → 执行 plan → 检查过关。

    **不修改原 game 状态**（在 deepcopy 上验证）。
    """
    try:
        g2 = copy.deepcopy(g)
    except Exception:
        # deepcopy 失败时回退到 reset+重放（会修改 g，调用方自行重置）
        try:
            g2 = g
            g2._current_level_index = level
            g2.level_reset()
        except Exception:
            return False
    else:
        try:
            g2._current_level_index = level
            g2.level_reset()
        except Exception:
            return False
    for item in plan:
        try:
            do_action(g2, item)
        except Exception:
            return False
        if is_win(g2, level):
            return True
    return is_win(g2, level)


def state_key(g) -> int:
    """状态哈希：捕获 sprite 的 name + 位置 + pixels（形状）。

    pixels.tobytes() 确保碰撞改形状后状态不同（re86 的坑）。
    """
    items = []
    for s in g.current_level.get_sprites():
        if getattr(s, "is_visible", True):
            try:
                px = int(hash(s.pixels.tobytes()) & 0xFFFFFFFF)
            except Exception:
                px = 0
            items.append((s.name, s.x, s.y, px))
    items.append(int(getattr(g, "_current_level_index", -1)))
    return hash(tuple(items))


# ── 求解策略协议 ──────────────────────────────────────────────


@runtime_checkable
class SolveStrategy(Protocol):
    """求解策略协议：给定 game + level，返回 plan 或 None。"""

    name: str

    def solve(self, g, level: int, **opts) -> Optional[list]:
        ...


# ── 策略 1：DeepcopyBFS（通用，m0r0 类）────────────────────────


@dataclass
class DeepcopyBFSStrategy:
    """deepcopy BFS：每节点 1 次 deepcopy + 1 次 perform_action。

    适合状态空间小的纯键盘/点击游戏（m0r0 镜像联动、块合并类）。
    state_key 含 pixels，捕获碰撞改形状。
    """

    name: str = "deepcopy_bfs"
    actions: tuple = (1, 2, 3, 4, 5)  # 默认纯键盘；含点击传 (1,2,3,4,5,6)
    max_depth: int = 18
    max_nodes: int = 60000
    t_limit: float = 180.0

    def solve(self, g, level: int, **opts) -> Optional[list]:
        g._current_level_index = level
        g.level_reset()
        t0 = time.time()
        try:
            start = copy.deepcopy(g)
        except Exception as e:
            print(f"[DeepcopyBFS] deepcopy 失败: {e}")
            return None
        queue: deque = deque([(start, [])])
        seen = {state_key(start)}
        nodes = 0
        while queue and nodes < self.max_nodes:
            if time.time() - t0 > self.t_limit:
                print(f"[DeepcopyBFS] TIME OUT nodes={nodes}")
                return None
            cur, path = queue.popleft()
            if is_win(cur, level):
                print(f"[DeepcopyBFS] FOUND len={len(path)} nodes={nodes}")
                return path
            if len(path) >= self.max_depth:
                continue
            nodes += 1
            for aid in self.actions:
                ng = copy.deepcopy(cur)
                do_action(ng, aid)
                if is_win(ng, level):
                    print(f"[DeepcopyBFS] FOUND len={len(path) + 1} nodes={nodes}")
                    return path + [aid]
                k = state_key(ng)
                if k not in seen:
                    seen.add(k)
                    queue.append((ng, path + [aid]))
            if nodes % 2000 == 0:
                print(f"[DeepcopyBFS] nodes={nodes} q={len(queue)} seen={len(seen)} t={time.time() - t0:.1f}s")
        print(f"[DeepcopyBFS] EXHAUSTED nodes={nodes} seen={len(seen)}")
        return None


# ── 策略 2：PositionEnum（re86 类，win=图案匹配）──────────────


@dataclass
class PositionEnumStrategy:
    """位置枚举：win=图案匹配类。

    枚举块位置组合，检查叠加图案是否匹配目标，找到后反推移动序列。
    适合 re86 类（块移动 + 图案匹配，无障碍或障碍少）。

    需游戏特定参数：
    - block_tag: 可移动块的 sprite tag
    - target_tag: 目标 sprite tag
    - step: 每次移动格数（re86=3）
    - canvas_size: 画布大小（re86=64）
    - win_fn: 自定义 win 检查（默认用 is_win）
    """

    name: str = "position_enum"
    block_tag: str = ""
    target_tag: str = ""
    step: int = 3
    canvas_size: int = 64
    max_combos: int = 600000
    t_limit: float = 300.0

    def solve(self, g, level: int, **opts) -> Optional[list]:
        """枚举块位置组合 → 检查 win → 反推移动序列。

        通用版：枚举位置后用引擎回放检查 win（不依赖具体图案匹配逻辑）。
        子类可重写 _check_pattern 做纯计算检查（更快）。
        """
        g._current_level_index = level
        g.level_reset()
        blocks = g.current_level.get_sprites_by_tag(self.block_tag)
        if not blocks:
            print(f"[PositionEnum] 无 block tag={self.block_tag}")
            return None
        t0 = time.time()
        # 块初始位置 + 可达位置（步长 step）
        block_info = []
        for b in blocks:
            xs = range(0, self.canvas_size, self.step)
            ys = range(0, self.canvas_size, self.step)
            block_info.append({"x0": b.x, "y0": b.y, "xs": list(xs), "ys": list(ys)})
        # 枚举位置组合（笛卡尔积）
        import itertools
        combos = 0
        for positions in itertools.product(*[[(x, y) for x in bi["xs"] for y in bi["ys"]] for bi in block_info]):
            combos += 1
            if combos > self.max_combos or time.time() - t0 > self.t_limit:
                print(f"[PositionEnum] 枚举上限 combos={combos}")
                return None
            # 用引擎回放检查：移动块到 positions，检查 win
            plan = self._make_move_plan(g, level, blocks, block_info, positions)
            if plan is not None and verify_plan(g, level, plan):
                print(f"[PositionEnum] FOUND combos={combos} plan_len={len(plan)}")
                return plan
        print(f"[PositionEnum] EXHAUSTED combos={combos}")
        return None

    def _make_move_plan(self, g, level, blocks, block_info, target_positions) -> Optional[list]:
        """给定块目标位置，生成 切换+移动 序列。"""
        g._current_level_index = level
        g.level_reset()
        blocks = g.current_level.get_sprites_by_tag(self.block_tag)

        def selected_index():
            for i, b in enumerate(blocks):
                cy, cx = b.height // 2, b.width // 2
                if b.pixels[cy, cx] == 0:
                    return i
            return -1

        plan = []
        cur_sel = selected_index()
        for bi in range(len(blocks)):
            tx, ty = target_positions[bi]
            x0, y0 = block_info[bi]["x0"], block_info[bi]["y0"]
            dx, dy = tx - x0, ty - y0
            n_right = dx // self.step if dx > 0 else 0
            n_left = -dx // self.step if dx < 0 else 0
            n_down = dy // self.step if dy > 0 else 0
            n_up = -dy // self.step if dy < 0 else 0
            if n_right == 0 and n_left == 0 and n_down == 0 and n_up == 0:
                continue
            # 切换到块 bi
            while cur_sel != bi:
                plan.append(5)
                do_action(g, 5)
                cur_sel = selected_index()
                if is_win(g, level):
                    return plan
            for _ in range(n_up):
                plan.append(1)
                do_action(g, 1)
                if is_win(g, level):
                    return plan
            for _ in range(n_down):
                plan.append(2)
                do_action(g, 2)
                if is_win(g, level):
                    return plan
            for _ in range(n_left):
                plan.append(3)
                do_action(g, 3)
                if is_win(g, level):
                    return plan
            for _ in range(n_right):
                plan.append(4)
                do_action(g, 4)
                if is_win(g, level):
                    return plan
        return plan


# ── 统一入口 ─────────────────────────────────────────────────


@dataclass
class SolveResult:
    """一次求解的结果。"""
    level: int
    plan: list
    strategy: str
    verified: bool
    elapsed: float


class PlanDecoder:
    """公共 plan 解码器：加载游戏 → 选策略 → 求解 → 引擎回放验证。

    用法::

        dec = PlanDecoder("re86")
        dec.load()
        # 自动选策略（按注册的策略）或指定
        plan = dec.solve(0, strategy=DeepcopyBFSStrategy())
        assert dec.verify(0, plan)
        plans = dec.solve_all()
    """

    # 游戏默认策略注册（game_id → 策略实例 + 参数）
    _GAME_STRATEGIES: dict[str, SolveStrategy] = {}

    def __init__(self, gid: str):
        self.gid = gid
        self.arc = None
        self.env = None
        self.g = None
        self.num_levels = 0

    @classmethod
    def register_strategy(cls, gid: str, strategy: SolveStrategy):
        """注册游戏默认策略。"""
        cls._GAME_STRATEGIES[gid] = strategy

    def load(self):
        """加载游戏（设置 sys.path + arc_agi 公共接口）。"""
        import sys, os
        from pathlib import Path
        # 自动设置 sys.path（integrations/ARC-AGI-3-Kaggle-Starter）
        here = Path(__file__).resolve()
        for candidate in (here.parents[3] / "integrations" / "ARC-AGI-3-Kaggle-Starter",):
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))
                v = candidate / "vendor"
                if v.exists():
                    sys.path.insert(0, str(v))
                sys.path.insert(0, str(candidate / "vendor" / "ARC-AGI-3-Agents"))
                break
        self.arc, self.env, self.g = load_game(self.gid)
        self.num_levels = len(self.g._levels)
        return self

    def verify(self, level: int, plan: Sequence) -> bool:
        """引擎回放验证 plan 真的过关。"""
        return verify_plan(self.g, level, plan)

    def solve(self, level: int, strategy: Optional[SolveStrategy] = None, **opts) -> Optional[list]:
        """求解指定关。

        - strategy=None：用注册的默认策略，否则用 DeepcopyBFS 兜底
        - 返回 plan（list[int] 或 list[(aid,x,y)]），失败返回 None
        """
        strat = strategy or self._GAME_STRATEGIES.get(self.gid) or DeepcopyBFSStrategy()
        t0 = time.time()
        plan = strat.solve(self.g, level, **opts)
        dt = time.time() - t0
        if plan is not None:
            ok = self.verify(level, plan)
            print(f"[PlanDecoder] {self.gid} L{level}: {strat.name} plan_len={len(plan)} verified={ok} {dt:.1f}s")
            if not ok:
                print(f"[PlanDecoder] 验证失败，丢弃")
                return None
        return plan

    def solve_all(self, max_levels: Optional[int] = None, strategy: Optional[SolveStrategy] = None) -> dict[int, list]:
        """求解所有关，返回 {level: plan}。遇到失败停止。"""
        plans = {}
        n = max_levels or self.num_levels
        for lv in range(n):
            plan = self.solve(lv, strategy=strategy)
            if plan is None:
                print(f"[PlanDecoder] {self.gid} L{lv} 失败，停止")
                break
            plans[lv] = plan
        return plans


# ── 预注册已知游戏策略 ───────────────────────────────────────

# re86: 位置枚举（块 tag + 目标 tag + 步长 3 + 画布 64）
PlanDecoder.register_strategy(
    "re86",
    PositionEnumStrategy(block_tag="0031cppcuvqlbi", target_tag="0054xnsuqceejm", step=3, canvas_size=64),
)
# m0r0: deepcopy BFS（纯键盘 1-5，镜像联动状态空间小）
PlanDecoder.register_strategy("m0r0", DeepcopyBFSStrategy(actions=(1, 2, 3, 4, 5), max_depth=18))



# ── 策略 1：DeepcopyBFS（通用，m0r0 类）────────────────────────


@dataclass
class DeepcopyBFSStrategy:
    """deepcopy BFS：每节点 1 次 deepcopy + 1 次 perform_action。

    适合状态空间小的纯键盘/点击游戏（m0r0 镜像联动、块合并类）。
    state_key 含 pixels，捕获碰撞改形状。
    """

    name: str = "deepcopy_bfs"
    actions: tuple = (1, 2, 3, 4, 5)  # 默认纯键盘；含点击传 (1,2,3,4,5,6)
    max_depth: int = 18
    max_nodes: int = 60000
    t_limit: float = 180.0

    def solve(self, g, level: int, **opts) -> Optional[list]:
        g._current_level_index = level
        g.level_reset()
        t0 = time.time()
        try:
            start = copy.deepcopy(g)
        except Exception as e:
            print(f"[DeepcopyBFS] deepcopy 失败: {e}")
            return None
        queue: deque = deque([(start, [])])
        seen = {state_key(start)}
        nodes = 0
        while queue and nodes < self.max_nodes:
            if time.time() - t0 > self.t_limit:
                print(f"[DeepcopyBFS] TIME OUT nodes={nodes}")
                return None
            cur, path = queue.popleft()
            if is_win(cur, level):
                print(f"[DeepcopyBFS] FOUND len={len(path)} nodes={nodes}")
                return path
            if len(path) >= self.max_depth:
                continue
            nodes += 1
            for aid in self.actions:
                ng = copy.deepcopy(cur)
                do_action(ng, aid)
                if is_win(ng, level):
                    print(f"[DeepcopyBFS] FOUND len={len(path) + 1} nodes={nodes}")
                    return path + [aid]
                k = state_key(ng)
                if k not in seen:
                    seen.add(k)
                    queue.append((ng, path + [aid]))
            if nodes % 2000 == 0:
                print(f"[DeepcopyBFS] nodes={nodes} q={len(queue)} seen={len(seen)} t={time.time() - t0:.1f}s")
        print(f"[DeepcopyBFS] EXHAUSTED nodes={nodes} seen={len(seen)}")
        return None


# ── 策略 2：PositionEnum（re86 类，win=图案匹配）──────────────


@dataclass
class PositionEnumStrategy:
    """位置枚举：win=图案匹配类。

    枚举块位置组合，检查叠加图案是否匹配目标，找到后反推移动序列。
    适合 re86 类（块移动 + 图案匹配，无障碍或障碍少）。

    需游戏特定参数：
    - block_tag: 可移动块的 sprite tag
    - target_tag: 目标 sprite tag
    - step: 每次移动格数（re86=3）
    - canvas_size: 画布大小（re86=64）
    - win_fn: 自定义 win 检查（默认用 is_win）
    """

    name: str = "position_enum"
    block_tag: str = ""
    target_tag: str = ""
    step: int = 3
    canvas_size: int = 64
    max_combos: int = 600000
    t_limit: float = 300.0

    def solve(self, g, level: int, **opts) -> Optional[list]:
        """枚举块位置组合 → 检查 win → 反推移动序列。

        通用版：枚举位置后用引擎回放检查 win（不依赖具体图案匹配逻辑）。
        子类可重写 _check_pattern 做纯计算检查（更快）。
        """
        g._current_level_index = level
        g.level_reset()
        blocks = g.current_level.get_sprites_by_tag(self.block_tag)
        if not blocks:
            print(f"[PositionEnum] 无 block tag={self.block_tag}")
            return None
        t0 = time.time()
        # 块初始位置 + 可达位置（步长 step）
        block_info = []
        for b in blocks:
            xs = range(0, self.canvas_size, self.step)
            ys = range(0, self.canvas_size, self.step)
            block_info.append({"x0": b.x, "y0": b.y, "xs": list(xs), "ys": list(ys)})
        # 枚举位置组合（笛卡尔积）
        import itertools
        combos = 0
        for positions in itertools.product(*[[(x, y) for x in bi["xs"] for y in bi["ys"]] for bi in block_info]):
            combos += 1
            if combos > self.max_combos or time.time() - t0 > self.t_limit:
                print(f"[PositionEnum] 枚举上限 combos={combos}")
                return None
            # 用引擎回放检查：移动块到 positions，检查 win
            plan = self._make_move_plan(g, level, blocks, block_info, positions)
            if plan is not None and verify_plan(g, level, plan):
                print(f"[PositionEnum] FOUND combos={combos} plan_len={len(plan)}")
                return plan
        print(f"[PositionEnum] EXHAUSTED combos={combos}")
        return None

    def _make_move_plan(self, g, level, blocks, block_info, target_positions) -> Optional[list]:
        """给定块目标位置，生成 切换+移动 序列。"""
        g._current_level_index = level
        g.level_reset()
        blocks = g.current_level.get_sprites_by_tag(self.block_tag)

        def selected_index():
            for i, b in enumerate(blocks):
                cy, cx = b.height // 2, b.width // 2
                if b.pixels[cy, cx] == 0:
                    return i
            return -1

        plan = []
        cur_sel = selected_index()
        for bi in range(len(blocks)):
            tx, ty = target_positions[bi]
            x0, y0 = block_info[bi]["x0"], block_info[bi]["y0"]
            dx, dy = tx - x0, ty - y0
            n_right = dx // self.step if dx > 0 else 0
            n_left = -dx // self.step if dx < 0 else 0
            n_down = dy // self.step if dy > 0 else 0
            n_up = -dy // self.step if dy < 0 else 0
            if n_right == 0 and n_left == 0 and n_down == 0 and n_up == 0:
                continue
            # 切换到块 bi
            while cur_sel != bi:
                plan.append(5)
                do_action(g, 5)
                cur_sel = selected_index()
                if is_win(g, level):
                    return plan
            for _ in range(n_up):
                plan.append(1)
                do_action(g, 1)
                if is_win(g, level):
                    return plan
            for _ in range(n_down):
                plan.append(2)
                do_action(g, 2)
                if is_win(g, level):
                    return plan
            for _ in range(n_left):
                plan.append(3)
                do_action(g, 3)
                if is_win(g, level):
                    return plan
            for _ in range(n_right):
                plan.append(4)
                do_action(g, 4)
                if is_win(g, level):
                    return plan
        return plan


# ── 统一入口 ─────────────────────────────────────────────────


@dataclass
class SolveResult:
    """一次求解的结果。"""
    level: int
    plan: list
    strategy: str
    verified: bool
    elapsed: float


class PlanDecoder:
    """公共 plan 解码器：加载游戏 → 选策略 → 求解 → 引擎回放验证。

    用法::

        dec = PlanDecoder("re86")
        dec.load()
        # 自动选策略（按注册的策略）或指定
        plan = dec.solve(0, strategy=DeepcopyBFSStrategy())
        assert dec.verify(0, plan)
        plans = dec.solve_all()
    """

    # 游戏默认策略注册（game_id → 策略实例 + 参数）
    _GAME_STRATEGIES: dict[str, SolveStrategy] = {}

    def __init__(self, gid: str):
        self.gid = gid
        self.arc = None
        self.env = None
        self.g = None
        self.num_levels = 0

    @classmethod
    def register_strategy(cls, gid: str, strategy: SolveStrategy):
        """注册游戏默认策略。"""
        cls._GAME_STRATEGIES[gid] = strategy

    def load(self):
        """加载游戏（设置 sys.path + arc_agi 公共接口）。"""
        import sys, os
        from pathlib import Path
        # 自动设置 sys.path（integrations/ARC-AGI-3-Kaggle-Starter）
        here = Path(__file__).resolve()
        for candidate in (here.parents[3] / "integrations" / "ARC-AGI-3-Kaggle-Starter",):
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))
                v = candidate / "vendor"
                if v.exists():
                    sys.path.insert(0, str(v))
                sys.path.insert(0, str(candidate / "vendor" / "ARC-AGI-3-Agents"))
                break
        self.arc, self.env, self.g = load_game(self.gid)
        self.num_levels = len(self.g._levels)
        return self

    def verify(self, level: int, plan: Sequence) -> bool:
        """引擎回放验证 plan 真的过关。"""
        return verify_plan(self.g, level, plan)

    def solve(self, level: int, strategy: Optional[SolveStrategy] = None, **opts) -> Optional[list]:
        """求解指定关。

        - strategy=None：用注册的默认策略，否则用 DeepcopyBFS 兜底
        - 返回 plan（list[int] 或 list[(aid,x,y)]），失败返回 None
        """
        strat = strategy or self._GAME_STRATEGIES.get(self.gid) or DeepcopyBFSStrategy()
        t0 = time.time()
        plan = strat.solve(self.g, level, **opts)
        dt = time.time() - t0
        if plan is not None:
            ok = self.verify(level, plan)
            print(f"[PlanDecoder] {self.gid} L{level}: {strat.name} plan_len={len(plan)} verified={ok} {dt:.1f}s")
            if not ok:
                print(f"[PlanDecoder] 验证失败，丢弃")
                return None
        return plan

    def solve_all(self, max_levels: Optional[int] = None, strategy: Optional[SolveStrategy] = None) -> dict[int, list]:
        """求解所有关，返回 {level: plan}。遇到失败停止。"""
        plans = {}
        n = max_levels or self.num_levels
        for lv in range(n):
            plan = self.solve(lv, strategy=strategy)
            if plan is None:
                print(f"[PlanDecoder] {self.gid} L{lv} 失败，停止")
                break
            plans[lv] = plan
        return plans


# ── 预注册已知游戏策略 ───────────────────────────────────────

# re86: 位置枚举（块 tag + 目标 tag + 步长 3 + 画布 64）
PlanDecoder.register_strategy(
    "re86",
    PositionEnumStrategy(block_tag="0031cppcuvqlbi", target_tag="0054xnsuqceejm", step=3, canvas_size=64),
)
# m0r0: deepcopy BFS（纯键盘 1-5，镜像联动状态空间小）
PlanDecoder.register_strategy("m0r0", DeepcopyBFSStrategy(actions=(1, 2, 3, 4, 5), max_depth=18))



