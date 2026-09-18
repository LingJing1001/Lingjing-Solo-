"""Φ 信息密度与泡壁几何 —— 协议 2/3/7/9 的 Lite 实现。

不做真实场方程；用分块统计 + 差分 ROI 实现：
  - 面积律：清晰区小、迷雾区大 → 算力集中在泡壁
  - ∇Φ：密度梯度能量作为「时空弯曲」代理，偏置探索
"""
from __future__ import annotations

import numpy as np
from .types import PhiDensity, BubbleWall
from .utils import clamp


def compute_phi(grid: np.ndarray, block: int = 8) -> PhiDensity:
    """粗粒化信息密度：非零/彩色像素的分块占比。"""
    if grid is None:
        return PhiDensity(blocks=np.zeros((1, 1), dtype=np.float32))
    g = np.asarray(grid)
    H, W = g.shape
    bh = max(1, H // block)
    bw = max(1, W // block)
    # 实际块数
    nby = max(1, H // bh) if bh else 1
    nbx = max(1, W // bw) if bw else 1
    # 统一用 block 参数切
    n = max(1, min(H, W) // max(1, block))
    bs = max(1, min(H, W) // n)
    nby = max(1, H // bs)
    nbx = max(1, W // bs)
    blocks = np.zeros((nby, nbx), dtype=np.float32)
    for i in range(nby):
        for j in range(nbx):
            y0, y1 = i * bs, min(H, (i + 1) * bs)
            x0, x1 = j * bs, min(W, (j + 1) * bs)
            patch = g[y0:y1, x0:x1]
            # 密度 = 非背景占比 + 颜色多样性
            nz = float(np.count_nonzero(patch)) / max(1, patch.size)
            uniq = float(len(np.unique(patch))) / 16.0
            blocks[i, j] = 0.7 * nz + 0.3 * uniq

    # 梯度能量；AR25 等小网格可能粗粒化为 1×1，单轴时只计算可用方向。
    values = blocks.astype(np.float64)
    gy = np.gradient(values, axis=0) if values.shape[0] >= 2 else np.zeros_like(values)
    gx = np.gradient(values, axis=1) if values.shape[1] >= 2 else np.zeros_like(values)
    grad_e = float((gy * gy + gx * gx).sum())
    peak = tuple(int(x) for x in np.unravel_index(int(np.argmax(blocks)), blocks.shape))
    return PhiDensity(
        blocks=blocks,
        mean=float(blocks.mean()),
        grad_energy=grad_e,
        peak_block=peak,
    )


def compute_bubble(
    prev: np.ndarray | None,
    curr: np.ndarray,
    pad: int = 3,
    grid_size: int = 64,
) -> BubbleWall:
    """由帧差分构造泡壁清晰区；无差分时以非零连通外包络为清晰区。"""
    curr = np.asarray(curr)
    H, W = curr.shape
    if prev is not None:
        diff = np.argwhere(np.asarray(prev) != curr)
    else:
        diff = np.argwhere(curr > 0)

    if len(diff) == 0:
        # 全迷雾：给中心一小块清晰区，避免除零
        c = grid_size // 2
        rects = [(c - 4, c - 4, c + 4, c + 4)]
        clear_px = 81
        fog = 1.0
        ent = 0.0
    else:
        ys, xs = diff[:, 0], diff[:, 1]
        y0 = clamp(int(ys.min()) - pad, 0, H - 1)
        y1 = clamp(int(ys.max()) + pad, 0, H - 1)
        x0 = clamp(int(xs.min()) - pad, 0, W - 1)
        x1 = clamp(int(xs.max()) + pad, 0, W - 1)
        rects = [(y0, x0, y1, x1)]
        clear_px = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
        fog = 1.0 - min(1.0, clear_px / float(H * W))
        region = curr[y0:y1 + 1, x0:x1 + 1]
        hist = np.bincount(region.flatten().astype(np.int64), minlength=16)[:16]
        p = hist / max(1, hist.sum())
        p = p[p > 0]
        ent = float(-(p * np.log(p + 1e-12)).sum())

    return BubbleWall(
        clear_rects=rects,
        fog_ratio=fog,
        clear_pixels=clear_px,
        wall_entropy=ent,
    )


def curvature_proxy(phi: PhiDensity) -> float:
    """协议 3 Lite：用 ∇Φ 能量规范化为 [0,1]「弯曲」代理。"""
    if phi is None:
        return 0.0
    # 经验缩放：8×8 块梯度能量通常 < 若干单位
    return float(clamp(phi.grad_energy / 8.0, 0.0, 1.0))


def prefer_block_actions(phi: PhiDensity) -> dict[str, float]:
    """根据密度峰相对中心的位置，给方向动作轻微偏置（测地线直觉 Lite）。"""
    bias = {a: 0.0 for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")}
    if phi is None or phi.blocks is None:
        return bias
    by, bx = phi.peak_block
    cy, cx = (phi.blocks.shape[0] - 1) / 2.0, (phi.blocks.shape[1] - 1) / 2.0
    dy, dx = by - cy, bx - cx
    if abs(dy) >= abs(dx) and abs(dy) > 0.3:
        bias["ACTION2" if dy > 0 else "ACTION1"] += 0.35
    elif abs(dx) > 0.3:
        bias["ACTION4" if dx > 0 else "ACTION3"] += 0.35
    else:
        bias["ACTION5"] += 0.15
    return bias
