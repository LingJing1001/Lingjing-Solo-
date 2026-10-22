"""世界原语：只描述格子、颜色、连通、平移。

这里**刻意**不包含任何关于某个游戏的假设：

  ✗ 没有"目标 = 稀有 + 静止 + 内部"
  ✗ 没有"键是上下左右"
  ✗ 没有"可动块很小"
  ✗ 没有"颜色 N 是墙"

只留下一个认知上的先验——**世界由可以被整体搬动的组件构成，动作通过接触
产生副作用**。这是假设空间（hypothesis space），不是答案。先验越窄，能解释的
世界越少；先验越宽，搜索越贵。这里的取舍是"刚性平移 + 接触副作用"，因为它
覆盖了 ARC-AGI-3 里绝大多数可交互机制，同时搜索代价还能接受。

一个原则：凡是能从数据里读出来的，绝不写死在这里。
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

Cell = tuple[int, int]
Grid = tuple[tuple[int, ...], ...]


# ─────────────────────────────────────────────────────────────
# 基本读写
# ─────────────────────────────────────────────────────────────
def to_grid(obj) -> Grid:
    """任何 grid 形态（list/ndarray/tuple）→ 不可变 Grid，可直接当 dict key。"""
    if obj is None:
        return ()
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    try:
        return tuple(tuple(int(v) for v in row) for row in obj)
    except TypeError:
        return ()


def dims(g: Grid) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def bg_colour(g: Grid) -> int:
    """背景 = 出现次数最多的颜色；并列取编号小的，保证同样的图永远同样答案。"""
    cnt = Counter()
    for row in g:
        cnt.update(row)
    if not cnt:
        return 0
    best = max(cnt.values())
    return min(c for c, n in cnt.items() if n == best)


def colours(g: Grid) -> dict[int, int]:
    cnt = Counter()
    for row in g:
        cnt.update(row)
    return dict(cnt)


def diff(g0: Grid, g1: Grid, ignore: frozenset[Cell] | set[Cell] = frozenset()
         ) -> dict[Cell, tuple[int, int]]:
    """只有真正变了的格子：(r,c) -> (旧色, 新色)。

    `ignore` 是"环境自己在动"的格子集合（精灵动画、计时器、状态栏）。
    它不是游戏规则，是背景减除：这些格子每帧都变，和按了什么键无关，
    所以不能拿它们当"这个动作造成了什么"的证据。
    """
    out: dict[Cell, tuple[int, int]] = {}
    h = min(len(g0), len(g1))
    for r in range(h):
        a, b = g0[r], g1[r]
        if a == b:
            continue
        for c in range(min(len(a), len(b))):
            if a[c] != b[c] and (r, c) not in ignore:
                out[(r, c)] = (a[c], b[c])
    return out


def mismatch(g0: Grid, g1: Grid, ignore: frozenset[Cell] | set[Cell] = frozenset()) -> int:
    if dims(g0) != dims(g1):
        return 10 ** 9
    n = 0
    for r in range(len(g0)):
        a, b = g0[r], g1[r]
        if a == b:
            continue
        for c in range(len(a)):
            if a[c] != b[c] and (r, c) not in ignore:
                n += 1
    return n


def bbox(ps: Iterable[Cell]) -> tuple[int, int, int, int]:
    ps = list(ps)
    rs = [p[0] for p in ps]
    cs = [p[1] for p in ps]
    return (min(rs), min(cs), max(rs), max(cs))


def origin(ps: Iterable[Cell]) -> Cell:
    r0, c0, _, _ = bbox(ps)
    return (r0, c0)


def normalise(ps: Iterable[Cell]) -> frozenset[Cell]:
    r0, c0, _, _ = bbox(ps)
    return frozenset((r - r0, c - c0) for r, c in ps)


def shift(ps: Iterable[Cell], d: Cell) -> frozenset[Cell]:
    return frozenset((r + d[0], c + d[1]) for r, c in ps)


def colour_map(g: Grid, ps: Iterable[Cell]) -> tuple[int, ...]:
    """组件的有序颜色序列，用来判断"是不是同一个东西"。"""
    return tuple(sorted(g[r][c] for r, c in ps))


# ─────────────────────────────────────────────────────────────
# 连通分量
# ─────────────────────────────────────────────────────────────
def components(g: Grid, conn: int = 4, same_colour: bool = True,
               bg: Optional[int] = None) -> list[frozenset[Cell]]:
    """按连通性切块。same_colour=False 时把相邻的非背景格粘成一块（多色 sprite）。"""
    h, w = dims(g)
    if bg is None:
        bg = bg_colour(g)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn == 8:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    seen: set[Cell] = set()
    out: list[frozenset[Cell]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            v = g[r][c]
            if same_colour and v == bg:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            blob = []
            while stack:
                y, x = stack.pop()
                blob.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen:
                        if same_colour:
                            if g[ny][nx] != v:
                                continue
                        else:
                            if g[ny][nx] == bg:
                                continue
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            out.append(frozenset(blob))
    return out


# ─────────────────────────────────────────────────────────────
# 预测：把一组格子整体搬走
# ─────────────────────────────────────────────────────────────
def move(g: Grid, ps: frozenset[Cell] | set[Cell], d: Cell,
         bg: Optional[int] = None) -> Optional[Grid]:
    """把 ps 整体平移 d，原位置回填背景色。越界返回 None（= 被边界挡住）。

    注意顺序：先清后画。这样"块压到别的东西上"时块的颜色获胜 —— 压制关系
    不需要另外规定，它是"后画的盖住先画的"这一条通用规则的自然结果。
    """
    if bg is None:
        bg = bg_colour(g)
    h, w = dims(g)
    out = [list(row) for row in g]
    for r, c in ps:
        out[r][c] = bg
    for r, c in ps:
        nr, nc = r + d[0], c + d[1]
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        out[nr][nc] = g[r][c]
    return tuple(tuple(row) for row in out)


def move_many(g: Grid, motions: dict[frozenset[Cell], Cell],
              bg: Optional[int] = None) -> Optional[Grid]:
    """多块各移各的：{块: 位移} 同时生效。任一格子被推出边界就整体返回 None。

    "一次动作只搬一块"是一个**假**假设 —— 实测 ar25 一次按键有两个 L 形
    反向移动（左边向西、右边向东），单块模型永远解释不了它，于是每步都判成
    "纯副作用"，语义学不到。改成"任意个子集，各自一位移"不需要任何游戏知识，
    只是把假设空间放大到能容纳"多个受控体"。
    """
    if bg is None:
        bg = bg_colour(g)
    h, w = dims(g)
    out = [list(row) for row in g]
    for C in motions:
        for r, c in C:
            out[r][c] = bg
    for C, d in motions.items():
        for r, c in C:
            nr, nc = r + d[0], c + d[1]
            if not (0 <= nr < h and 0 <= nc < w):
                return None
            out[nr][nc] = g[r][c]
    return tuple(tuple(row) for row in out)


def paint(g: Grid, assigns: dict[Cell, int]) -> Grid:
    out = [list(row) for row in g]
    h, w = dims(g)
    for (r, c), v in assigns.items():
        if 0 <= r < h and 0 <= c < w:
            out[r][c] = v
    return tuple(tuple(row) for row in out)


def erase(g: Grid, ps: Iterable[Cell], bg: Optional[int] = None) -> Grid:
    if bg is None:
        bg = bg_colour(g)
    return paint(g, {p: bg for p in ps})


# ─────────────────────────────────────────────────────────────
# 文本：给日志和人看
# ─────────────────────────────────────────────────────────────
_DIRS = {(0, 0): "不动", (-1, 0): "北", (1, 0): "南",
         (0, -1): "西", (0, 1): "东"}


def dir_text(d: Cell) -> str:
    if d in _DIRS:
        return _DIRS[d]
    return f"({d[0]:+d},{d[1]:+d})"


def cells_text(ps: Iterable[Cell]) -> str:
    ps = list(ps)
    if not ps:
        return "∅"
    r0, c0, r1, c1 = bbox(ps)
    return f"{len(ps)}格@{r0}-{r1},{c0}-{c1}"


_HEX = "0123456789abcdef"


def to_text(g: Grid) -> str:
    """一屏能看全的字符画：一色一字符，0-15 → 0-9a-f。

    人眼在这一层比数字强得多：ar25 的 L0 一眼就能看出"左边 555、右边 444、
    中间一列墙"，而 [[9,9,...]] 读十遍也看不出来。只用于调试与日志。
    """
    return "\n".join("".join(_HEX[v] if 0 <= v < 16 else "?" for v in row) for row in g)
