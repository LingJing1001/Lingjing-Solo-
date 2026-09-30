"""SSA-E4 · 谱聚类对象分割（图拉普拉斯第二特征向量的块级归一化割）。

对象 = 共变化区域（spectral clustering 的认知解读，白皮书 §2 Ch7 行、§7 E4）：
  1) 帧历史 → 块级共变化指示矩阵 Z（frames × B²），亲和 A = ẐẐᵀ（余弦归一）
  2) 归一化拉普拉斯 L = I − D^{-1/2} A D^{-1/2}
  3) 取 L 的最小非平凡特征向量（Fiedler 向量及其邻维）作嵌入 → k-means 分割

纯 numpy：B² = 64 → eigh(64×64) 微秒级。分出的簇即"候选对象"，
供槽位世界模型与反事实引擎做对象级操作（路线图 E4 的前置件）。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def block_change_matrix(history: Sequence[np.ndarray], blocks: int = 8) -> Optional[np.ndarray]:
    """帧序列 → 块共激活矩阵 Z ∈ {0,1}^{F×B²}。

    Z[f, i] = 1 当第 f 帧的块 i 内容与基准帧（第 0 帧）不同。
    对象 = 共激活块群：同一物体的各块签名（时间列）相同/相似，
    不同物体的签名相位不同 → 余弦亲和后自然分离。
    单块对象也有签名（相邻帧差分视角下会被漏掉），故用基准帧差分。
    """
    frames = [np.asarray(g) for g in history if g is not None]
    if len(frames) < 2:
        return None
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        return None
    H, W = frames[0].shape
    bh = max(1, -(-H // blocks))
    bw = max(1, -(-W // blocks))
    ph, pw = bh * blocks, bw * blocks

    def _blocks(g: np.ndarray) -> np.ndarray:
        padded = np.zeros((ph, pw), dtype=np.int32)
        padded[:H, :W] = g
        cells = padded.reshape(blocks, bh, blocks, bw).transpose(0, 2, 1, 3)
        return cells.reshape(blocks * blocks, bh * bw)

    base = _blocks(frames[0])
    rows: List[np.ndarray] = []
    for f in frames[1:]:
        bf = _blocks(f)
        rows.append((bf != base).any(axis=1).astype(np.float64))
    return np.vstack(rows)


def cochange_affinity(history: Sequence[np.ndarray], blocks: int = 8) -> Optional[np.ndarray]:
    """块×块共变化亲和矩阵（余弦归一），对角置零。"""
    Z = block_change_matrix(history, blocks)
    if Z is None or Z.shape[0] < 2:
        return None
    if float((Z != 0).sum()) <= 0:
        return None  # 全程无变化 → 无对象结构
    norms = np.linalg.norm(Z, axis=0)
    safe = norms > 0
    Zn = np.zeros_like(Z)
    Zn[:, safe] = Z[:, safe] / norms[safe][None, :]
    A = Zn.T @ Zn
    # 对角保留自相似（=1）：单块对象的度来自自身，孤立组件才能被归一化割分开
    return A


def spectral_segments(
    affinity: np.ndarray,
    n_clusters: int = 2,
    kmeans_iters: int = 25,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """亲和矩阵 → 每块簇标签（归一化割；嵌入维 = n_clusters−1 的 Fiedler 邻域）。"""
    A = np.asarray(affinity, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 4:
        return None
    n_clusters = max(2, min(int(n_clusters), 4))
    deg = A.sum(axis=1)
    if float(deg.max()) <= 1e-12:
        return None  # 无任何共变化 → 无对象结构
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    L = np.eye(A.shape[0]) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    try:
        w, V = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return None
    m = min(n_clusters - 1, V.shape[1] - 1)
    emb = V[:, 1 : 1 + m] if m >= 1 else V[:, :1]
    emb = emb * d_inv_sqrt[:, None]  # 反归一化回原空间
    labels = _kmeans(emb, n_clusters, kmeans_iters, seed)
    # 零度块（从未激活）标记为 -1：无对象证据，不参与簇
    labels = np.where(deg > 1e-12, labels, -1)
    return labels


def _kmeans(X: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    """轻量 k-means（k-means++ 初始化，固定种子可复现）。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    centers = [X[int(rng.integers(n))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        total = float(d2.sum())
        if total <= 1e-12:
            centers.append(X[int(rng.integers(n))])
            continue
        probs = d2 / total
        centers.append(X[int(rng.choice(n, p=probs))])
    C = np.asarray(centers, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max(1, iters)):
        dist = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = dist.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            mask = labels == j
            if mask.any():
                C[j] = X[mask].mean(axis=0)
    return labels


def object_regions(history: Sequence[np.ndarray], blocks: int = 8, n_clusters: int = 2):
    """一步到位：帧历史 → {簇 id: 块索引列表}（只保留非平凡簇）。"""
    A = cochange_affinity(history, blocks)
    if A is None:
        return {}
    labels = spectral_segments(A, n_clusters=n_clusters)
    if labels is None:
        return {}
    out: dict = {}
    for i, lb in enumerate(labels.tolist()):
        out.setdefault(int(lb), []).append(i)
    return {k: v for k, v in out.items() if len(v) < len(labels)}
