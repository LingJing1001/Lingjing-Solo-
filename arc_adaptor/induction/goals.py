"""目标不是猜出来的，是**从"过关那一刻"倒推出来的**。

引擎手里只有一个免费标签：`levels_completed` 从 n 跳到 n+1 的那一次转移。
这一次转移是全世界唯一不需要人去标注的"正样本"。于是：

    过关前一帧  g0 ──动作 a──▶ 过关后一帧 g1

把 g0→g1 拆开（谁动了、动了多少、还变了什么），再问一句：
"g1 里成立、g0 里不成立的**最简陈述**是什么？" —— 那就是目标。

三种陈述（假设空间，不是针对某关的答案）：

    coincide  可动体最终和一块与它**同形**的静止区域重合
    touch     可动体最终碰到某块**固定结构**
    clear     某个颜色的格子从画面上**全部消失**

选哪个由数据决定：谁能用更少的格子解释这一次过关，就用谁。
换到下一关时，把这条陈述**带过去**重新定位（同形结构 / 同色格子），
所以目标是一个「概念」，不是一个写死的坐标。

这就是"不要人写规则"的具体含义：人只提供三种可能的**句式**，
句子填什么，由每一次过关事件自己填。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from . import world as W
from .world import Cell, Grid


@dataclass
class Goal:
    kind: str                       # "coincide" | "touch" | "clear"
    target: frozenset[Cell]         # 在过关那一帧里成立的"要去的地方"
    colour: Optional[int] = None    # clear/touch 用
    shape: Optional[frozenset[Cell]] = None   # coincide 用：同形指纹
    # 目标区域**自己**的配色（取自过关**前**那一帧，即被盖住之前的原色）。
    # 跨关迁移时这是最有判别力的键：ar25 的目标在两关里大小变了（5 格 → 8 格）、
    # 形状也变了，但颜色不变。只按"格数最接近"挑候选，会挑到可动块自己身上。
    cols: tuple[int, ...] = ()
    evidence: int = 0               # 由几次过关事件支持
    origin_level: int = -1
    note: str = ""
    evidence_total: int = 1         # clear 用：起始总格数

    # ── 判定 ──────────────────────────────────────────────
    def progress(self, g: Grid, body: frozenset[Cell]) -> float:
        """0..1，越大越接近达成。只在 body 认得出时有意义。"""
        if not body or not self.target:
            return 0.0
        if self.kind == "coincide":
            return len(body & self.target) / max(len(self.target), 1)
        if self.kind == "touch":
            near = _dilate(body, 1)
            return 1.0 if near & self.target else 0.0
        if self.kind == "clear":
            left = sum(1 for row in g for v in row if v == self.colour)
            tot = max(self.evidence_total, 1)
            return max(0.0, 1.0 - left / tot)
        return 0.0

    def done(self, g: Grid, body: frozenset[Cell]) -> bool:
        if self.kind == "coincide":
            return bool(body) and len(body & self.target) >= max(1, int(0.9 * len(body)))
        if self.kind == "touch":
            return bool(_dilate(body, 1) & self.target)
        if self.kind == "clear":
            return all(v != self.colour for row in g for v in row)
        return False

    # ── 跨关迁移：概念不变，坐标重定位 ─────────────────────
    def relocate(self, g: Grid, body: frozenset[Cell],
                 avoid: frozenset[Cell] | set[Cell] = frozenset()
                 ) -> Optional["Goal"]:
        """把这条目标陈述搬到新画面上。

        `avoid` 是"已知会动的块"占的格子。目标必须是**不动的东西**：一个
        会跟着按键跑的块不可能是"要去的那个地方"。这条不是关卡知识，是
        "目标只有在被达成之后才停止存在"的必然推论 —— 会动的块达成一次
        还会再动。
        """
        h, w = W.dims(g)
        if not body:
            return None
        bshape = W.normalise(body)
        bsize = len(body)
        bg = W.bg_colour(g)

        def pool() -> list[frozenset[Cell]]:
            out: list[frozenset[Cell]] = []
            for same in (True, False):
                for s in W.components(g, conn=4, same_colour=same, bg=bg):
                    if s == body or (s & body) or (s & avoid):
                        continue
                    if len(s) > bsize * 4:
                        continue
                    out.append(s)
                if out:
                    break
            return out

        if self.kind in ("coincide", "touch"):
            want = frozenset(self.target)
            tshape = W.normalise(want)
            tlen = len(want)
            want_cols = set(self.cols)
            best: Optional[tuple[tuple, int, frozenset[Cell]]] = None
            for s in pool():
                cols = set(W.colour_map(g, s))
                score = 0
                # 配色是主键。用**集合**而不是多重集：目标在两关里大小会变
                # （ar25 从 5 格变 8 格），多重集永远对不上，集合还能对上。
                if want_cols:
                    score += 2 * len(want_cols - cols)
                # 形状是次键；目标自己长什么样也跨关保留
                score += 0 if W.normalise(s) == tshape else 3
                if self.kind == "coincide":
                    # coincide 要求目标与**当前**的体同形，体换了形状就别硬套
                    score += 0 if W.normalise(s) == bshape else 2
                score += abs(len(s) - tlen) // 2
                gap = _gap(s, body)
                key = (score, gap)
                if best is None or key < best[0]:
                    best = (key, gap, s)
            if best is None:
                return None
            tgt = frozenset(best[2])
            if tgt == body or (tgt & body):
                return None
            return Goal(kind=self.kind, target=tgt, colour=self.colour,
                        shape=bshape if self.kind == "coincide" else tshape,
                        cols=W.colour_map(g, tgt),
                        evidence=self.evidence, origin_level=self.origin_level,
                        note=self.note)

        if self.kind == "clear":
            ps = frozenset((r, c) for r in range(h) for c in range(w)
                           if g[r][c] == self.colour)
            if not ps:
                return None
            return Goal(kind="clear", target=ps, colour=self.colour,
                        cols=(self.colour,) * len(ps) if self.colour is not None else (),
                        evidence_total=len(ps),
                        evidence=self.evidence, origin_level=self.origin_level,
                        note=self.note)
        return None

    def text(self) -> str:
        if self.kind == "coincide":
            return f"让可动体完全压住同形区域 {W.cells_text(self.target)}"
        if self.kind == "touch":
            return f"碰到固定结构 {W.cells_text(self.target)}"
        if self.kind == "clear":
            return f"清掉全部 {self.colour} 号色（{len(self.target)} 格）"
        return self.kind


def _dilate(ps: frozenset[Cell], k: int) -> frozenset[Cell]:
    out = set(ps)
    for _ in range(k):
        add = set()
        for r, c in out:
            add |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        out |= add
    return frozenset(out)


def _gap(a: frozenset[Cell], b: frozenset[Cell]) -> int:
    """两个块之间的最小切比雪夫距离，用来在多个候选里挑"该去的那个"。"""
    best = 10 ** 6
    for r1, c1 in a:
        for r2, c2 in b:
            d = max(abs(r1 - r2), abs(c1 - c2))
            if d < best:
                best = d
        if best == 0:
            break
    return best


# ─────────────────────────────────────────────────────────────
# 归纳：从一次过关事件里读出目标
# ─────────────────────────────────────────────────────────────
def induce(g0: Grid, g1: Grid, body_old: frozenset[Cell], move: Cell,
           side: dict[Cell, tuple[int, int]], level: int = -1) -> list[Goal]:
    """返回按"解释成本"排序的候选目标（越低越可信）。

    打分只用一个原则：**能解释这次过关的最简陈述**。复杂度 ≈ 目标格数。
    """
    out: list[Goal] = []
    if not body_old:
        return out
    body_new = W.shift(body_old, move)
    bshape = W.normalise(body_old)

    # ── coincide：过关后可动体正好压在某个区域上 ──────────────
    # 目标区域 = 这次移动的**落点脚印**，只保留其中"原本就有东西、或这次被改写"
    # 的格子。为什么用落点而不是"原本就在那儿的同形区域"：实测 ar25 的目标会被
    # 逐步吃掉 —— 块压过去的格子变成块的颜色，剩下的残留只有 1 格。要求
    # "g0 里该处存在同形区域"，那一关就永远归纳不出目标。
    delta = W.diff(g0, g1)
    bg0 = W.bg_colour(g0)
    tgt = frozenset(p for p in body_new
                    if (0 <= p[0] < len(g0) and 0 <= p[1] < len(g0[p[0]]))
                    and (g0[p[0]][p[1]] != bg0 or p in delta))
    # 落点必须**真的压在"不是我"的东西上**，否则"块走到空地"也会被当成目标。
    # 实测 ar25：过关那一步两个块都在动，先试的那个是向左挪一格（落点里有一半
    # 是它自己原来的格子），于是归纳出"压住 5格@15-17,1-3"（整片背景），
    # 迁移时颜色是 9（背景），对任何候选都没有判别力，最后挑中了隔壁的可动块。
    lands_on_something = any(g0[p[0]][p[1]] != bg0 and p not in body_old
                             for p in tgt)
    if tgt and lands_on_something:
        # 目标区域的配色：**去掉可动体自己的颜色和背景色**。过关那一刻块已经
        # 盖在目标上，目标格会被改写成块色；不扣掉，目标在下一关就会认成"块本身"。
        # 实测 ar25：不给体色扣掉时 cols 变成 (4,4,11,11,11)，到 L1 一搜
        # 就搜到另一个 4 号色块，目标永远指错。
        body_cols = set(W.colour_map(g0, body_old))
        tcols = tuple(c for c in W.colour_map(g0, tgt)
                      if c not in body_cols and c != bg0)
        if not tcols:
            tcols = W.colour_map(g0, tgt)
        note = ("过关时块刚好落在同形区域上"
                if W.normalise(tgt) == bshape else "过关时块压在既有结构上")
        out.append(Goal(kind="coincide", target=tgt, shape=bshape, cols=tcols,
                        evidence=1, origin_level=level, note=note))

    # ── touch：过关后紧邻某块固定结构 ────────────────────────
    if side:
        side_cells = frozenset(side)
        known = [p for p in side_cells if p in _cells_of(g0)]
        out.append(Goal(kind="touch", target=side_cells, evidence=1,
                        cols=W.colour_map(g0, known) if known else (),
                        origin_level=level, note="过关同时有结构发生变化"))

    # ── clear：某个颜色整片消失，而且消失处**变回背景** ──────────
    # "被盖住"不算消失：ar25 上块压住幽灵，幽灵那几格变成了块的颜色，
    # 颜色计数确实清零了，但那是覆盖不是清除。差别就在这儿，必须查。
    cnt0, cnt1 = Counter(), Counter()
    for row in g0:
        cnt0.update(row)
    for row in g1:
        cnt1.update(row)
    bg1 = W.bg_colour(g1)
    gone = []
    for c, n in cnt0.items():
        if not n or c == bg1 or cnt1.get(c, 0) != 0:
            continue
        ps = frozenset((r, cc) for r in range(len(g0))
                       for cc in range(len(g0[r])) if g0[r][cc] == c)
        if all(g1[r][cc] == bg1 for r, cc in ps if r < len(g1) and cc < len(g1[r])):
            gone.append((c, ps))
    for c, ps in gone:
        out.append(Goal(kind="clear", target=ps, colour=c, evidence_total=len(ps),
                        evidence=1, origin_level=level, note=f"{c} 号色整片变回背景"))

    out.sort(key=lambda g: len(g.target))
    return out


def _cells_of(g: Grid) -> set[Cell]:
    return {(r, c) for r in range(len(g)) for c in range(len(g[r]))}
