"""从一次次「按下键 → 画面变了」里，反推出"谁动了、怎么动的、还顺带改了什么"。

这是整个自动推理的第一步，也是最关键的一步：**不告诉 agent 哪块是"玩家"**。
它只在所有可能的连通块里找"整体平移能解释最多变化"的那一个。

一个转移 (g0, a, g1) 被拆成三层：

    body    被整体搬动的那个连通块（若无则动作无效 / 被挡）
    move    位移向量 d
    side    搬动之外还变了的格子 —— 这些是**副作用**，也就是开关、门、计数器、
            可收集物。agent 完全不需要知道它们叫什么，只需要知道"谁一碰就变"。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..judge import beta_mean
from . import world as W
from .world import Cell, Grid

# 「这个键的语义定下来了吗」的概率门槛。
#
# 这不是新加的人为规定，而是把原来那条 `len(c) == 1`（零冲突票）翻译成概率：
# 零冲突 + 1 票 → Beta 后验 0.667，零冲突 + 2 票 → 0.75。取 0.65 恰好保留
# "1 票一致也算知道"这条原有行为，同时第一次允许**压倒性多数**算知道。
#
# 为什么必须换：ar25 的 L1 上 ACTION3 出现 `西×9 + 东×1`，旧规则因为有冲突票
# 而永远返回 False，于是 agent 每 3 步重做一次判别实验、永远不收敛。
# 证据不是"有没有异议"，是"异议占多大比例"。
KNOWN_AT = 0.65


@dataclass
class Motion:
    """一次转移的解释：谁动了、各动了多少、还剩下什么解释不了。"""
    motions: dict[frozenset[Cell], Cell] = field(default_factory=dict)
    side: dict[Cell, tuple[int, int]] = field(default_factory=dict)  # 移动之外的变化
    miss: int = 0                            # 用"搬动 + 副作用"解释后还差几格
    focus: Optional[frozenset[Cell]] = None  # 本次重点跟踪的那一块（供上层做规划）
    # 块 → 身份（形状 + 配色）。身份让"同一个体跑到别处"仍能对上，
    # 也让 ar25 那种"左块向西、右块向东"能分成两张语义表而不是一张分裂的表。
    idents: dict[frozenset[Cell], tuple] = field(default_factory=dict)

    def ident(self, C: frozenset[Cell]) -> Optional[tuple]:
        return self.idents.get(C)

    @property
    def ident_moves(self) -> dict[tuple, Cell]:
        """{身份 → 位移}。一次动作有几个体在动，这里就有几条。"""
        out: dict[tuple, Cell] = {}
        for C, d in self.motions.items():
            k = self.idents.get(C)
            if k is not None:
                out[k] = d
        return out

    # ── 单块接口：上层（按键语义、规划）只需要盯住一个受控体 ──
    @property
    def body(self) -> frozenset[Cell]:
        return self.focus or frozenset()

    @property
    def move(self) -> Cell:
        return self.motions.get(self.focus, (0, 0)) if self.focus else (0, 0)

    @property
    def body_new(self) -> frozenset[Cell]:
        return W.shift(self.body, self.move)

    # ── 全部受控体：副作用归因要用全集 ────────────────────────
    @property
    def bodies(self) -> frozenset[Cell]:
        out: set[Cell] = set()
        for C in self.motions:
            out |= C
        return frozenset(out)

    @property
    def moved(self) -> bool:
        return bool(self.motions)

    def text(self) -> str:
        if not self.motions:
            s = "没有整体平移"
            if self.side:
                s += f"，只变了 {W.cells_text(self.side)}"
            return s
        parts = [f"{W.cells_text(C)}向{W.dir_text(d)}移{abs(d[0]) + abs(d[1])}"
                 for C, d in sorted(self.motions.items(),
                                    key=lambda kv: W.bbox(kv[0]))[:3]]
        s = "；".join(parts)
        if len(self.motions) > 3:
            s += f"…（共 {len(self.motions)} 块）"
        if self.side:
            s += f"，副作用 {W.cells_text(self.side)}"
        return s


def _twin_moves(g0: Grid, bg: int) -> list[tuple[frozenset[Cell], Cell]]:
    """找"同形同色的两块"：一块移到另一块的位置，就是最干净的平移证据。

    为什么先走这条：真玩家几乎总有一个"孪生"目标（幽灵、影子、目标格），
    而用它反推位移是 O(n²) 而不是 O(n·|Δ|·range²)，快一个数量级。
    """
    comps = W.components(g0, conn=4, same_colour=True, bg=bg)
    groups: dict[tuple, list[frozenset[Cell]]] = defaultdict(list)
    for s in comps:
        if len(s) < 1 or len(s) > 400:
            continue
        groups[(W.normalise(s), W.colour_map(g0, s))].append(s)
    out = []
    for lst in groups.values():
        if len(lst) < 2:
            continue
        for s in lst:
            o1 = W.origin(s)
            for t in lst:
                if t is s or t == s:
                    continue
                o2 = W.origin(t)
                d = (o2[0] - o1[0], o2[1] - o1[1])
                if d == (0, 0):
                    continue
                out.append((s, d))
    return out


def _shift_score(g0: Grid, g1: Grid, C: frozenset[Cell], d: Cell,
                 delta: dict[Cell, tuple[int, int]],
                 ignore: frozenset[Cell] | set[Cell]) -> Optional[int]:
    """把 C 平移 d 以后，它的每个格子还和 g1 对得上几个。越界返回 None。"""
    h, w = W.dims(g0)
    s = 0
    for r, c in C:
        nr, nc = r + d[0], c + d[1]
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if (nr, nc) in ignore or g1[nr][nc] == g0[r][c]:
            s += 1
    return s


def _best_shift(g0: Grid, g1: Grid, C: frozenset[Cell], span: int,
                delta: dict[Cell, tuple[int, int]],
                ignore: frozenset[Cell] | set[Cell]) -> Cell:
    """这块自己该往哪走？**逐块独立判定**，所以多块反向移动也能解释。

    基准是"不动"：这块里没参与本次变化的格子数（= base）。
    只有找到严格更优的位移才认为它动了 —— 静止的东西永远不动，
    一整列同色墙也不会因为"往北挪一格还有 20/21 格对得上"而被判成在动。
    """
    base = sum(1 for p in C if p not in delta)
    best_d, best_s = (0, 0), base
    for dr in range(-span, span + 1):
        for dc in range(-span, span + 1):
            if dr == 0 and dc == 0:
                continue
            s = _shift_score(g0, g1, C, (dr, dc), delta, ignore)
            if s is None:
                continue
            if s > best_s:
                best_d, best_s = (dr, dc), s
            elif s == best_s and s > base:
                if abs(dr) + abs(dc) < abs(best_d[0]) + abs(best_d[1]):
                    best_d = (dr, dc)
    return best_d if best_s > base else (0, 0)


def _pick_focus(motions: dict[frozenset[Cell], Cell],
                prefer: Optional[tuple[frozenset[Cell], tuple[int, ...]]],
                g0: Grid) -> Optional[frozenset[Cell]]:
    """挑一个"我"来跟踪。先认上一关那块的形状/配色，认不出就取最大的。"""
    if not motions:
        return None
    if prefer is not None:
        shape, cols = prefer
        for C in motions:
            if W.normalise(C) == shape and W.colour_map(g0, C) == cols:
                return C
        for C in motions:
            if W.normalise(C) == shape:
                return C
    return sorted(motions, key=lambda C: (-len(C), W.bbox(C)))[0]


def fit(g0: Grid, g1: Grid, span: int = 5, budget: int = 12,
        ignore: frozenset[Cell] | set[Cell] = frozenset(),
        prefer: Optional[tuple[frozenset[Cell], tuple[int, ...]]] = None
        ) -> Optional[Motion]:
    """解释一次转移。返回 None 表示"一个像素都没变"。

    假设空间：**每一个非背景连通块，要么不动，要么整体平移一个短位移。**
    逐块独立判定 → 支持"多块同时动、方向各不相同"。
    然后按"重放一遍和真实画面的差"打分，并对个别块做一次贪心修正。
    """
    delta = W.diff(g0, g1, ignore)
    if not delta:
        return None
    bg = W.bg_colour(g0)
    h, w = W.dims(g0)
    limit = max(4, (h * w) // 2)
    comps = [s for s in W.components(g0, conn=4, same_colour=True, bg=bg)
             if 1 <= len(s) <= limit]

    motions: dict[frozenset[Cell], Cell] = {}
    for C in comps:
        d = _best_shift(g0, g1, C, span, delta, ignore)
        if d != (0, 0):
            motions[C] = d

    if not motions:
        # 有变化，但没有任何块整体平移能解释 —— 纯副作用（开关、计数器、画面重绘）
        return Motion(motions={}, side=delta, miss=len(delta))

    # ── 奥卡姆剪枝 ───────────────────────────────────────────
    # 逐块独立找位移很容易撞上巧合：两格宽的墙体碎片"往左上挪一格"也能对上
    # 一半格子。判据只有一个 —— 把这个块真的搬过去，整体失配必须下降。
    # 不加这道检验，ar25 上会出现"墙每隔一步往左上跳三格"这种幻觉。
    kept: dict[frozenset[Cell], Cell] = {}
    cur = W.mismatch(g0, g1, ignore)
    left = dict(motions)
    while left:
        pick = None
        for C, d in left.items():
            trial = dict(kept)
            trial[C] = d
            p = W.move_many(g0, trial, bg)
            if p is None:
                continue
            m2 = W.mismatch(p, g1, ignore)
            if m2 < cur and (pick is None or m2 < pick[0]):
                pick = (m2, C, d)
        if pick is None:
            break
        cur, C, d = pick
        kept[C] = d
        left.pop(C)

    if not kept:
        return Motion(motions={}, side=delta, miss=len(delta))
    motions = kept
    pred = W.move_many(g0, motions, bg)
    if pred is None:
        return Motion(motions={}, side=delta, miss=len(delta))
    best = W.mismatch(pred, g1, ignore)

    side: dict[Cell, tuple[int, int]] = {}
    for p, (_old, new) in delta.items():
        if pred[p[0]][p[1]] != new:
            side[p] = (pred[p[0]][p[1]], new)
    return Motion(motions=motions, side=side, miss=best,
                  focus=_pick_focus(motions, prefer, g0),
                  idents={C: (W.normalise(C), W.colour_map(g0, C))
                          for C in motions})


# ─────────────────────────────────────────────────────────────
# 按键语义表：动作 → 位移
# ─────────────────────────────────────────────────────────────
@dataclass
class KeySemantics:
    """每个动作对"可动体"做了什么。同一个动作在不同局面下位移不同是很常见的
    （碰墙、被推、踩传送带），所以这里存的是**分布**而不是唯一值。

    更重要的是**按体分表**（`by_id`）。ar25 实测：按 A3，左房间那块向西、
    右房间那块向东。所有块挤在一张表里，A3 永远有两个互相矛盾的位移，
    于是每个键都"还没学会"，agent 只能一直做判别实验，永远不收敛。
    分成两张表以后，A3 对左块是确定的西、对右块是确定的东 —— 两个都学会。

    全局表（`votes`）保留：它仍然是跨关迁移和"遇到没见过的体"时的兜底。
    """
    votes: dict[str, Counter] = field(default_factory=dict)
    noop: dict[str, int] = field(default_factory=dict)     # 按了没用/被挡的次数
    unclear: dict[str, int] = field(default_factory=dict)  # 画面变了，但说不清是谁动的
    death: dict[str, int] = field(default_factory=dict)    # 按了就死
    trials: int = 0
    by_id: dict[tuple, dict[str, Counter]] = field(default_factory=dict)
    noop_id: dict[tuple, dict[str, int]] = field(default_factory=dict)
    unclear_id: dict[tuple, dict[str, int]] = field(default_factory=dict)

    # ── 写入 ────────────────────────────────────────────────
    def observe(self, act: str, d: Optional[Cell], dead: bool = False,
                unclear: bool = False, ident: Optional[tuple] = None) -> None:
        self.trials += 1
        if dead:
            self.death[act] = self.death.get(act, 0) + 1
            return
        if unclear:
            self.unclear[act] = self.unclear.get(act, 0) + 1
            if ident is not None:
                t = self.unclear_id.setdefault(ident, {})
                t[act] = t.get(act, 0) + 1
            return
        moved = d is not None and d != (0, 0)
        if ident is not None:
            if moved:
                self.by_id.setdefault(ident, {}).setdefault(act, Counter())[d] += 1
            else:
                t = self.noop_id.setdefault(ident, {})
                t[act] = t.get(act, 0) + 1
                # 单体的 noop 不进全局表：别的体可能正在动，会让全局结论失真
                return
        if not moved:
            self.noop[act] = self.noop.get(act, 0) + 1
            return
        self.votes.setdefault(act, Counter())[d] += 1

    # ── 读取：给了身份就先看它自己那张表 ──────────────────
    def best(self, act: str, ident: Optional[tuple] = None
             ) -> tuple[Optional[Cell], int, int]:
        """(最可能的位移, 支持数, 冲突数)。冲突数 > 0 说明这个键的语义还没定。"""
        if ident is not None:
            c0 = self.by_id.get(ident, {}).get(act)
            if c0:
                d, n = c0.most_common(1)[0]
                return (d, n, sum(c0.values()) - n)
        c = self.votes.get(act)
        if not c:
            return (None, 0, 0)
        d, n = c.most_common(1)[0]
        return (d, n, sum(c.values()) - n)

    def known(self, act: str, ident: Optional[tuple] = None) -> bool:
        """语义定下来了吗 —— 由**证据量**回答，不由"有没有异议"回答。

        旧实现是硬布尔：只要票型里出现第二个位移，就永远返回 False。
        这在单受控体世界里等价于"有冲突就是没学会"，但在 ar25 的 L1 上它会
        卡死：A3 在左块上拿到 9 票西、1 票异常，agent 于是每 3 步重做一次
        判别实验，永远等不到"零冲突"。
        """
        if act in self.death:
            return True
        p, _n, _c = self.belief(act, ident)
        return p >= KNOWN_AT

    def belief(self, act: str, ident: Optional[tuple] = None
               ) -> tuple[float, int, int]:
        """(这个语义成立的概率, 票数, 冲突票数)，Beta(1,1) 后验均值。

        1 票 → 0.667，2 票 → 0.75，9 票 + 1 异议 → 0.833。
        证据越多越接近 1，但永远不到 1 —— 学到的结论随时可被新票推翻。
        """
        c = None
        if ident is not None:
            c = self.by_id.get(ident, {}).get(act)
        if not c:
            c = self.votes.get(act)
        if not c:
            return (0.0, 0, 0)
        top = c.most_common(1)[0][1]
        total = sum(c.values())
        return (beta_mean(top, total), total, total - top)

    def noop_count(self, act: str, ident: Optional[tuple] = None) -> int:
        if ident is not None:
            n = self.noop_id.get(ident, {}).get(act, 0)
            if n:
                return n
        return self.noop.get(act, 0)

    def unit_steps(self, acts: list[str], ident: Optional[tuple] = None
                   ) -> dict[Cell, str]:
        """{单位位移 → 键}。只有 1 格的步子才能一格格走，多步的键不能用来寻路。"""
        out: dict[Cell, str] = {}
        for a in acts:
            d, _n, _c = self.best(a, ident)
            if d and abs(d[0]) + abs(d[1]) == 1 and d not in out:
                out[d] = a
        return out

    def known_acts(self, ident: tuple) -> int:
        return sum(1 for a, c in self.by_id.get(ident, {}).items() if len(c) == 1)

    def text(self) -> str:
        parts = []
        for a in sorted(self.votes):
            d, n, conf = self.best(a)
            parts.append(f"{a}→{W.dir_text(d)}×{n}" + (f"(有{conf}次不同)" if conf else ""))
        for a, n in sorted(self.noop.items()):
            if not self.votes.get(a):
                parts.append(f"{a}→无效×{n}")
        for a, n in sorted(self.death.items()):
            parts.append(f"{a}→致命×{n}")
        s = " ".join(parts) if parts else "（还没按过任何键）"
        if len(self.by_id) > 1:
            s += f"｜按体 ×{len(self.by_id)}"
        return s

    def body_text(self, ident: tuple) -> str:
        """某一个体自己学到的表 —— 多体游戏里这才是有用的那一栏。"""
        tbl = self.by_id.get(ident, {})
        if not tbl:
            return "（还没学到）"
        return " ".join(f"{a}→{W.dir_text(c.most_common(1)[0][0])}"
                        for a, c in sorted(tbl.items()))


# ─────────────────────────────────────────────────────────────
# 副作用：谁一碰就变
# ─────────────────────────────────────────────────────────────
@dataclass
class Effect:
    """一个"被碰过就变"的区域。不命名它是什么，只记它怎么变、在什么条件下变。"""
    cells: frozenset[Cell]
    touch: Counter = field(default_factory=Counter)     # 接触时的变化类型计数
    n: int = 0                                          # 观测次数
    last: Optional[dict[Cell, tuple[int, int]]] = None

    def signature(self) -> tuple:
        return (len(self.cells), tuple(sorted(self.touch.items())))

    def text(self) -> str:
        kinds = ", ".join(f"{k}×{v}" for k, v in self.touch.most_common(3))
        return f"{W.cells_text(self.cells)} [{kinds}] ×{self.n}"


class EffectBook:
    """按"变化发生的位置"聚类副作用。同一块地方反复变，就是同一个机制。"""

    def __init__(self) -> None:
        self.effects: list[Effect] = []

    def observe(self, side: dict[Cell, tuple[int, int]], body_new: frozenset[Cell]) -> None:
        if not side:
            return
        # 变化格子按连通聚成区域（4 邻接，不看颜色 —— 只看"一起变"）
        groups = _cluster(set(side))
        for grp in groups:
            kind = Counter()
            for p in grp:
                pr, nx = side[p]
                kind[("erase" if nx == 0 else "recolour") if pr != nx else "same"] += 1
            # 找最近的已知机制：重心距离最近且不重叠
            hit = None
            gc = _centroid(grp)
            best = 1e9
            for e in self.effects:
                if e.cells & grp:
                    hit = e
                    break
                d = _dist(_centroid(e.cells), gc)
                if d < best:
                    best = d
                    hit = e
            if hit is not None and (hit.cells & grp or best <= 2.0):
                hit.cells = hit.cells | grp
                hit.n += 1
                for k, v in kind.items():
                    hit.touch[k] += v
                hit.last = {p: side[p] for p in grp}
            else:
                e = Effect(cells=frozenset(grp), n=1, last={p: side[p] for p in grp})
                for k, v in kind.items():
                    e.touch[k] += v
                self.effects.append(e)

    def text(self, limit: int = 4) -> str:
        if not self.effects:
            return "（还没观察到副作用）"
        es = sorted(self.effects, key=lambda e: -e.n)[:limit]
        return " | ".join(e.text() for e in es)


def _cluster(ps: set[Cell]) -> list[set[Cell]]:
    out: list[set[Cell]] = []
    seen: set[Cell] = set()
    for p in ps:
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        blob = {p}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in ps and q not in seen:
                        seen.add(q)
                        stack.append(q)
                        blob.add(q)
        out.append(blob)
    return out


def _centroid(ps: Iterable[Cell]) -> tuple[float, float]:
    ps = list(ps)
    return (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
