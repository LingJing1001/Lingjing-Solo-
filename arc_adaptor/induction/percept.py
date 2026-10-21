"""感知前端：把像素画面还原成"原生格"，再按对象描述变化。

为什么要这一层，用实测数字说：ar25 的帧是 64×64，但一个原生格在画面上是
3×3 像素。一次按键会改动 12 个原生格 —— 在像素上就是 108 个点。
推理层如果在像素上做连通分析，两个并排移动的图形会因为像素间距、边缘
抗锯齿而连成一片，于是"整体平移"永远解释不了这次变化。

所以这一层的职责只有一条：**把画布还原成游戏真正在用的那张格子纸**。
它不知道哪一格是墙、哪一格是玩家、什么算过关 —— 那些仍然全靠 induction。

────────────────────────────────────────────────────────────
为什么用冠军的算法，而不是自己写一个
────────────────────────────────────────────────────────────
自己那版（`guess_block` + `quantise`）只有一个假设：画面从 (0,0) 开始按 k
对齐。实测 25 个盘对拍（`scripts/probe_percept_vs_champion.py`）：
23 盘两路完全一致，但有 1 盘是真差别 —— **cn04 的画面有 2 像素边距**，
k=3 在 offset=0 下对齐失败，我们判成"没缩放"（64×64 全像素），
冠军搜 offset 得到 (2,2)/k=3 的 20×20，逐块全同色（uniform=1.00）。

那 2 像素就让整个推理层在一个 3 倍缩放的画面上按像素工作，
"整体平移 1 格"变成了"平移 3 像素"，假设空间直接错位。

所以这一层换成冠军 `lexicon` 的两步：

    scale_lock(整关的帧)     一关锁一个 (k, oy, ox)，被任何一帧推翻就重锁
    native_grid(当前帧, lock) 按锁定的 (k, oy, ox) 取每块众数

自己那版降级为**对照组**（`guess_block`/`quantise` 仍导出，探针要用）。
这不是"人定了一条规则"，是把一个已在 25 个盘上验证过的量纲估计搬过来；
结论（k、offset）仍然每帧重新落定，可以被新证据推翻。
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from ..duck.champion import lexicon as LX
from . import world as W
from .world import Cell, Grid


# ─────────────────────────────────────────────────────────────
# 1. 对照组：单帧估 k、假设 offset=0（保留给探针与回归测试）
# ─────────────────────────────────────────────────────────────
def uniformity(g: Grid, k: int) -> float:
    """按 k×k 对齐切开后，内部完全同色的方块占比。"""
    h, w = W.dims(g)
    hs, ws = h // k, w // k
    if hs < 2 or ws < 2:
        return 0.0
    uni = tot = 0
    for bi in range(hs):
        r0 = bi * k
        for bj in range(ws):
            c0 = bj * k
            first = g[r0][c0]
            same = True
            for r in range(r0, r0 + k):
                row = g[r]
                for c in range(c0, c0 + k):
                    if row[c] != first:
                        same = False
                        break
                if not same:
                    break
            tot += 1
            if same:
                uni += 1
    return uni / tot if tot else 0.0


def guess_block(g: Grid, max_k: int = 8, need: float = 0.97) -> int:
    """哪个 k 最像"一格"。同分取小的 —— 能被 2 整除的画布通常也能被 4 整除，
    取小的才不会把一个 2 像素的格子合并掉。"""
    best_k, best_f = 1, 0.0
    for k in range(2, max_k + 1):
        f = uniformity(g, k)
        if f >= need and f > best_f:
            best_k, best_f = k, f
    return best_k


def quantise(g: Grid, k: int) -> Grid:
    """每 k×k 块取出现最多的颜色（众数），得到原生格。

    众数而不是左上角取样：块跨在两种颜色之间时，取众数等于"多数票决定
    这一格是什么"，比取角点稳。同票取编号小的，保证同样的画面永远同样结果。
    """
    if k <= 1 or not g:
        return g
    h, w = W.dims(g)
    hs, ws = h // k, w // k
    out: list[tuple[int, ...]] = []
    for bi in range(hs):
        row: list[int] = []
        for bj in range(ws):
            cnt: Counter[int] = Counter()
            for r in range(bi * k, (bi + 1) * k):
                gr = g[r]
                for c in range(bj * k, (bj + 1) * k):
                    cnt[gr[c]] += 1
            top = max(cnt.values())
            row.append(min(v for v, n in cnt.items() if n == top))
        out.append(tuple(row))
    return tuple(out)


# ─────────────────────────────────────────────────────────────
# 2. 对象：把格子重新组织成"东西"
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Obj:
    cells: frozenset[Cell]
    colour: int
    shape: frozenset[Cell]                 # 归一化后的形状（只描述"长什么样"）
    box: tuple[int, int, int, int]

    def text(self) -> str:
        r0, c0, r1, c1 = self.box
        return (f"{self.colour}号色 {len(self.cells)}格@r{r0}-{r1},c{c0}-{c1}"
                f"（形状 {len(self.shape)}）")


def objects(g: Grid, max_frac: float = 0.6) -> list[Obj]:
    """按同色 4 连通切出对象，丢掉"超过画面六成"的块（那是背景/地面/边框）。"""
    if not g:
        return []
    h, w = W.dims(g)
    bg = W.bg_colour(g)
    out: list[Obj] = []
    for s in W.components(g, conn=4, same_colour=True, bg=bg):
        if len(s) > max_frac * h * w:
            continue
        out.append(Obj(cells=s, colour=g[W.bbox(s)[0]][W.bbox(s)[1]],
                       shape=W.normalise(s), box=W.bbox(s)))
    out.sort(key=lambda o: o.box)
    return out


# ─────────────────────────────────────────────────────────────
# 3. 会话级感知：一关锁一个量纲，任何一个新帧都能推翻它
# ─────────────────────────────────────────────────────────────
class Perception:
    """把一帧像素变成原生格。

    k 与 offset 不是常量，也不是"开局测一次"——是一关内所有见过的帧**共同**
    同意的那组 (k, oy, ox)。

    但"能被推翻"不等于"一帧就能推翻"。第一版是每帧都比对，只要有一帧自己
    估得不一样就整关重锁 —— ar25 跑 48 步就崩了：某个过渡帧（全屏重绘、重置
    入场）过不了 k=3 的检验，`scale_lock` 于是把整关降级到 k=1，画面尺寸从
    21×21 跳成 64×64，上一帧记的位置全部越界。

    所以尺度按**票**换：连续 `need` 帧自己的估计都与当前锁不符，才重锁。

    但还有一条来自崩溃现场的约束：**同一关里，原生画布的尺寸不变**。
    ar25 跑到第 22 帧时画面结构大变（游程数 254 → 404，90% 是 3 的倍数
    → 64%），`_detect` 判成"没缩放"，21×21 一次性跳成 64×64，上一帧记的
    位置全部越界。换分辨率只可能发生在换关，而换关有 `levels_completed`
    这个免费信号；所以尺寸一变就拒收新锁，等外层调 `reset_scale()` 再学。
    这不是关卡知识，是"一个世界的格子数不会自己变"这条形状约束。
    """

    def __init__(self, max_k: int = 8, history: int = 40, need: int = 3) -> None:
        self.max_k = max_k
        self.scales = tuple(range(max_k, 1, -1))
        self.history = max(4, history)
        self.need = max(1, need)
        # 以下是结论，不是配置；每帧按新证据重写
        self.k = 1
        self.offset = (0, 0)
        self.uniform = 1.0
        self.residue = 0
        self.seen = 0
        self.relocks = 0
        self.rejected = 0
        self._frames: list[list[list[int]]] = []
        self._lock: Optional[tuple[int, int, int]] = None
        self._prior: Optional[tuple[int, int, int]] = None   # 上一关的尺度，跨关先验
        self._mismatch = 0
        self._shape: Optional[tuple[int, int]] = None

    def reset_scale(self) -> None:
        """换关了：整关的帧历史与锁作废，但**上一关的尺度留作先验**。

        依据是数据：25 个公开盘里，跨关改变绘制尺度的只有 4 个，其余 21 个不变。
        所以"新关的第一帧看不出周期"时，优先试上一关的尺度，而不是跳到像素级。
        """
        if self._lock is not None and self._lock[0] > 1:
            self._prior = self._lock
        self._frames.clear()
        self._lock = None
        self._mismatch = 0
        self._shape = None
        self.relocks = 0
        self.rejected = 0

    def _pick_lock(self, g: Grid, pure: list[list[int]]) -> Optional[tuple[int, int, int]]:
        """这一关该用哪个尺度。

        k=1 不是一个模型，是"没找到模型"。所以当检测回到 1 时，先问一句：
        已知的尺度还能不能解释这张画面（k×k 块里 75% 以上是单色）？
        能，就继续用；不能，才接受"这一关没有缩放"。
        """
        lk = LX.scale_lock(self._frames, scales=self.scales, tolerance=0.05)
        cand = tuple(lk) if lk else None
        if cand is None or cand[0] == 1:
            prior = self._lock if (self._lock and self._lock[0] > 1) else self._prior
            if prior is not None and prior[0] > 1:
                k0, oy0, ox0 = prior
                if k0 * 2 <= min(len(g), len(g[0])) and \
                        LX._uniform_fraction(pure, k0, oy0, ox0) >= 0.75:
                    return prior
        return cand

    def see(self, g: Grid) -> Grid:
        """像素 → 原生格。同一帧重复喂不会重复计票。"""
        if not g:
            return g
        self.seen += 1
        pure = [list(r) for r in g]
        fresh = not self._frames or self._frames[-1] != pure
        if fresh:
            self._frames.append(pure)
            if len(self._frames) > self.history:
                # 留一帧最初画面（关卡的原始样子）+ 最近的
                keep = max(2, self.history // 4)
                self._frames = self._frames[:keep] + self._frames[-(self.history - keep):]
            own = LX._detect(pure, self.scales)
            if os.environ.get("PERCEPT_DEBUG"):
                runs = LX._interior_runs(pure)
                n = max(len(runs), 1)
                ok3 = sum(1 for x in runs if x % 3 == 0) / n
                print(f"[percept-dbg] own={own[:3]} runs={len(runs)} "
                      f"ok3={ok3:.2f} uni3={LX._uniform_fraction(pure, 3, 0, 0):.2f} "
                      f"短run={sum(1 for x in runs if x == 1)}", flush=True)
            if self._lock is not None and (own[0], own[1], own[2]) != self._lock:
                self._mismatch += 1
            else:
                self._mismatch = 0
        else:
            own = LX._detect(pure, self.scales)

        # 锁分两种：帧数不够时只是**临时读数**（换关的第一帧常是黑屏/白屏过渡，
        # 拿它定调会把整关钉在像素级 —— ar25 的 L1 就是这么废掉的），
        # 攒够 need 帧才落成确定锁，此后只有尺寸不变的候选才被接受。
        lock = self._lock
        if self._lock is None:
            if len(self._frames) >= self.need:
                self._lock = self._pick_lock(g, pure)
                lock = self._lock
                self.relocks += 1
                self._mismatch = 0
            else:
                prior = self._prior
                lock = prior if prior is not None else (own[0], own[1], own[2])
        elif self._mismatch >= self.need:
            cand = self._pick_lock(g, pure)
            cells = LX.native_grid(g, scales=self.scales, lock=cand)["cells"]
            cand_shape = (len(cells), len(cells[0]) if cells else 0)
            if self._shape is not None and cand_shape != self._shape:
                # 尺寸变了 → 这更像是读了一张异常帧，不是世界换了分辨率：拒收
                self.rejected += 1
            else:
                if cand != self._lock:
                    self.relocks += 1
                self._lock = cand
                lock = cand
            self._mismatch = 0

        r = LX.native_grid(g, scales=self.scales, lock=lock)
        self.k = int(r["scale"])
        self.offset = tuple(r["offset"])
        self.uniform = float(r["uniform"])
        self.residue = int(r["residue"])
        out = tuple(tuple(int(v) for v in row) for row in r["cells"])
        if self._lock is not None:
            self._shape = W.dims(out)
        return out

    def decode(self, g: Grid) -> Grid:
        """按**当前锁**把一帧像素翻译成原生格，不学任何东西。

        为什么需要它：过关那一步返回的帧序列里，除最后一帧（已是下一关）之外
        还有胜利瞬间的画面。那一帧要拿来和"过关前"做对比归纳目标，但它不能
        参与尺度学习 —— 它属于上一关，喂进新的帧历史会污染新关的锁。
        """
        if not g:
            return g
        r = LX.native_grid(g, scales=self.scales, lock=self._lock)
        return tuple(tuple(int(v) for v in row) for row in r["cells"])

    def note(self, g: Grid) -> str:
        h, w = W.dims(g)
        objs = objects(g)
        return (f"[感知] {h}×{w}，{len(set(v for row in g for v in row))} 色，"
                f"{len(objs)} 个对象（最大 {max((len(o.cells) for o in objs), default=0)} 格）")

    def text(self) -> str:
        if self.k <= 1:
            s = "画面已是原生分辨率"
        else:
            s = f"一格 {self.k} 像素"
            if self.offset != (0, 0):
                s += f"，从 r{self.offset[0]}c{self.offset[1]} 起算"
        if self.residue:
            s += f"（还原残差 {self.residue} 像素、块纯色率 {self.uniform:.2f}）"
        return s
