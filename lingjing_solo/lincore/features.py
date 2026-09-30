"""SSA-L0/L1 · 块特征编码 φ 与像素活跃掩码。

理论依据：SSA 白皮书 §5.1（docs/SSA_谱子空间认知架构_理论白皮书_v1.md）。
网格 x ∈ {0..15}^{H×W} 映射为块级特征向量 φ(x) ∈ R^{2B²}：
  - 占用率（每块非背景像素比例）→ "哪里有东西"
  - 颜色多样性（每块不同颜色数 / 16）→ "哪里复杂"
效应向量 Δ = φ(x_{t+1}) − φ(x_t) 是子空间世界模型的原子观测（[S6] Ch1）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def encode_grid(grid: Optional[np.ndarray], blocks: int = 8) -> Optional[np.ndarray]:
    """网格 → 块特征向量（占用率 + 多样性，长度 2·blocks²）。

    对任意 H×W 自适应切块（向上取整补零），64×64/8 块 → 128 维。
    """
    if grid is None:
        return None
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        return None
    H, W = g.shape
    bh = max(1, -(-H // blocks))
    bw = max(1, -(-W // blocks))
    ph, pw = bh * blocks, bw * blocks
    padded = np.zeros((ph, pw), dtype=np.int16)
    padded[:H, :W] = g
    cells = padded.reshape(blocks, bh, blocks, bw).transpose(0, 2, 1, 3)
    cells = cells.reshape(blocks * blocks, bh * bw)

    occ = (cells != 0).mean(axis=1).astype(np.float64)

    colors = np.clip(cells, 0, 15).astype(np.int64)
    counts = np.zeros((blocks * blocks, 16), dtype=np.float64)
    idx = np.arange(blocks * blocks).repeat(bh * bw)
    np.add.at(counts, (idx, colors.ravel()), 1.0)
    diversity = (counts > 0).sum(axis=1) / 16.0

    return np.concatenate([occ, diversity])


def block_of(x: int, y: int, shape: Tuple[int, int], blocks: int = 8) -> int:
    """像素坐标 → 展平块索引（用于 EigenSkillMap 的区域编号）。"""
    H, W = int(shape[0]), int(shape[1])
    bh = max(1, -(-H // blocks))
    bw = max(1, -(-W // blocks))
    bi = min(blocks - 1, max(0, int(y) // bh))
    bj = min(blocks - 1, max(0, int(x) // bw))
    return bi * blocks + bj


def block_center(region: int, shape: Tuple[int, int], blocks: int = 8) -> Tuple[int, int]:
    """展平块索引 → 块中心像素坐标（score 的逆映射）。"""
    H, W = int(shape[0]), int(shape[1])
    bh = max(1, -(-H // blocks))
    bw = max(1, -(-W // blocks))
    bi, bj = divmod(int(region), blocks)
    return (int(bj * bw + bw / 2), int(bi * bh + bh / 2))


def activity_mask(prev: Optional[np.ndarray], curr: Optional[np.ndarray]) -> np.ndarray:
    """像素级"曾经变化"掩码（零空间 N(E) 支撑集的逐像素保守估计，[S6] Ch3）。"""
    if prev is None or curr is None:
        curr_only = curr if curr is not None else prev
        if curr_only is None:
            return np.zeros((1, 1), dtype=bool)
        return np.zeros_like(np.asarray(curr_only), dtype=bool)
    p = np.asarray(prev)
    c = np.asarray(curr)
    if p.shape != c.shape:
        return np.ones_like(c, dtype=bool)
    return (p != c)


def grid_summary(grid: Optional[np.ndarray], blocks: int = 8) -> Optional[np.ndarray]:
    """紧凑摘要（用于 RidgeLinear/SGD 的状态侧特征）：
    [全局占用, 多样性, 变化块数(相对上次), 活跃块占比]。由 mind 维护增量。
    """
    feat = encode_grid(grid, blocks=blocks)
    if feat is None:
        return None
    b2 = blocks * blocks
    occ, div = feat[:b2], feat[b2:]
    return np.array(
        [occ.mean(), div.mean(), (occ > 0.02).sum() / b2, (div > 0.06).sum() / b2],
        dtype=np.float64,
    )
