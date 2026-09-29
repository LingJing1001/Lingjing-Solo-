"""卷积++ 变体族 — 多种 CNN 架构变体，用于前沿搜索。

每个变体共享相同的 encode 接口 (grid → embedding)，
但内部卷积结构不同，探索不同方向的表达能力。

变体列表:
  base       — 基线 3×3 stride-2 标准卷积 (NumpyCNN 原版)
  dilated    — 空洞卷积 (dilation=2)，扩大感受野不增参数
  separable  — 深度可分离卷积 (depthwise → pointwise)，参数更少
  residual   — 残差连接 (skip)，梯度流更稳
  attention  — 通道注意力 (SE-like squeeze-excite)
  wide       — 更宽通道 (2× channels)
  deep       — 4 层而非 3 层

Public API:
    build_variant(name, **kw)  — 按名称构造变体
    VARIANT_NAMES              — 所有可用变体名
    ConvVariant.encode(grid)   — 与 NumpyCNN 完全兼容
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=np.float32)


def _he_init(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    std = np.sqrt(2.0 / max(1, fan_in))
    return (rng.standard_normal(shape) * std).astype(np.float32)


def _conv2d_stride2(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    c_in, h, w = x.shape
    c_out = int(weight.shape[0])
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    s0, s1, s2 = xp.strides
    shape = (c_in, h2, w2, 3, 3)
    strides = (s0, s1 * 2, s2 * 2, s1, s2)
    patches = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    out = np.einsum("oijk,ihwjk->ohw", weight, patches, optimize=True)
    out = out + bias[:, None, None]
    return out.astype(np.float32, copy=False)


def _conv2d_dilated(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    dilation: int = 2,
) -> np.ndarray:
    """Dilated conv: 3×3 kernel with dilation, stride=2, pad=dilation.

    Vectorized: build dilated-patch index arrays then fancy-index once.
    """
    c_in, h, w = x.shape
    c_out = int(weight.shape[0])
    pad = dilation
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    hp, wp = xp.shape[1], xp.shape[2]
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    d = dilation
    oy = np.arange(h2) * 2
    ox = np.arange(w2) * 2
    ki = np.arange(3) * d
    kj = np.arange(3) * d
    row_idx = (oy[:, None] + ki[None, :]).ravel()
    col_idx = (ox[:, None] + kj[None, :]).ravel()
    row_idx = np.clip(row_idx, 0, hp - 1)
    col_idx = np.clip(col_idx, 0, wp - 1)
    sampled = xp[:, row_idx, :][:, :, col_idx]
    patches = sampled.reshape(c_in, h2, 3, w2, 3).transpose(0, 1, 3, 2, 4)
    out = np.einsum("oijk,ihwjk->ohw", weight, patches, optimize=True)
    out = out + bias[:, None, None]
    return out.astype(np.float32, copy=False)


def _depthwise_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Depthwise conv: weight shape (C, 1, 3, 3), each channel independent."""
    c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    s0, s1, s2 = xp.strides
    shape = (c, h2, w2, 3, 3)
    strides = (s0, s1 * 2, s2 * 2, s1, s2)
    patches = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    w_dw = weight[:, 0, :, :]
    out = np.einsum("cij,chwij->chw", w_dw, patches)
    out = out + bias[:, None, None]
    return out.astype(np.float32, copy=False)


def _pointwise_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """1×1 conv: weight shape (C_out, C_in, 1, 1)."""
    c_in = x.shape[0]
    c_out = int(weight.shape[0])
    w_pw = weight[:, :, 0, 0]
    flat = x.reshape(c_in, -1)
    out = (w_pw @ flat).reshape(c_out, x.shape[1], x.shape[2])
    out = out + bias[:, None, None]
    return out.astype(np.float32, copy=False)


def _channel_attention(
    x: np.ndarray,
    w_fc1: np.ndarray,
    b_fc1: np.ndarray,
    w_fc2: np.ndarray,
    b_fc2: np.ndarray,
    reduction: int = 4,
) -> np.ndarray:
    """SE-like squeeze-excite: global pool → fc → sigmoid → scale."""
    c = x.shape[0]
    squeezed = x.mean(axis=(1, 2))
    mid = max(1, c // reduction)
    h = _relu(squeezed[:mid] if c <= mid else (w_fc1 @ squeezed + b_fc1))
    exc = w_fc2 @ h + b_fc2
    if exc.shape[0] < c:
        exc_full = np.zeros(c, dtype=np.float32)
        exc_full[:exc.shape[0]] = exc
        exc = exc_full
    scale = 1.0 / (1.0 + np.exp(-np.clip(exc, -20, 20)))
    return x * scale[:, None, None]


class ConvVariant:
    """统一接口的卷积变体编码器。"""

    name: str = "base"

    def __init__(
        self,
        num_colors: int = 16,
        embed_dim: int = 64,
        channels: Tuple[int, ...] = (12, 24, 32),
        seed: int = 7,
    ):
        self.num_colors = int(num_colors)
        self.embed_dim = int(embed_dim)
        self.channels = tuple(int(c) for c in channels)
        self.rng = np.random.default_rng(seed)

    def _one_hot(self, grid: np.ndarray) -> np.ndarray:
        g = np.asarray(grid, dtype=np.int16)
        if g.ndim == 2 and g.shape[0] >= 58:
            g = g[:58]
        h, w = g.shape
        if h > 32 or w > 32:
            ys = np.linspace(0, h, 32, endpoint=False).astype(np.int32)
            xs = np.linspace(0, w, 32, endpoint=False).astype(np.int32)
            g = g[ys][:, xs]
        c = self.num_colors
        clipped = np.clip(g, 0, c - 1)
        eye = np.eye(c, dtype=np.float32)
        return eye[clipped].transpose(2, 0, 1)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _normalize(self, z: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(z) + 1e-8)
        return (z / n).astype(np.float32)

    def param_count(self) -> int:
        total = 0
        for attr in dir(self):
            v = getattr(self, attr)
            if isinstance(v, np.ndarray) and v.dtype in (np.float32, np.float64):
                total += v.size
        return total


class BaseVariant(ConvVariant):
    """基线: 标准 3×3 stride-2 卷积 (NumpyCNN 原版)。"""
    name = "base"

    def __init__(self, **kw):
        super().__init__(**kw)
        c0 = self.num_colors
        ch = self.channels
        self.w1 = _he_init((ch[0], c0, 3, 3), self.rng)
        self.b1 = np.zeros(ch[0], dtype=np.float32)
        self.w2 = _he_init((ch[1], ch[0], 3, 3), self.rng)
        self.b2 = np.zeros(ch[1], dtype=np.float32)
        self.w3 = _he_init((ch[2], ch[1], 3, 3), self.rng)
        self.b3 = np.zeros(ch[2], dtype=np.float32)
        self.w_fc = _he_init((ch[2], self.embed_dim), self.rng)
        self.b_fc = np.zeros(self.embed_dim, dtype=np.float32)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        x = self._one_hot(grid)
        h = _relu(_conv2d_stride2(x, self.w1, self.b1))
        h = _relu(_conv2d_stride2(h, self.w2, self.b2))
        h = _relu(_conv2d_stride2(h, self.w3, self.b3))
        pooled = h.mean(axis=(1, 2))
        z = _relu(pooled @ self.w_fc + self.b_fc)
        return self._normalize(z)


class DilatedVariant(ConvVariant):
    """空洞卷积: dilation=2, 感受野 5×5 等效, 参数量不变。"""
    name = "dilated"

    def __init__(self, **kw):
        super().__init__(**kw)
        c0 = self.num_colors
        ch = self.channels
        self.w1 = _he_init((ch[0], c0, 3, 3), self.rng)
        self.b1 = np.zeros(ch[0], dtype=np.float32)
        self.w2 = _he_init((ch[1], ch[0], 3, 3), self.rng)
        self.b2 = np.zeros(ch[1], dtype=np.float32)
        self.w3 = _he_init((ch[2], ch[1], 3, 3), self.rng)
        self.b3 = np.zeros(ch[2], dtype=np.float32)
        self.w_fc = _he_init((ch[2], self.embed_dim), self.rng)
        self.b_fc = np.zeros(self.embed_dim, dtype=np.float32)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        x = self._one_hot(grid)
        h = _relu(_conv2d_dilated(x, self.w1, self.b1, dilation=2))
        h = _relu(_conv2d_dilated(h, self.w2, self.b2, dilation=2))
        h = _relu(_conv2d_dilated(h, self.w3, self.b3, dilation=2))
        pooled = h.mean(axis=(1, 2))
        z = _relu(pooled @ self.w_fc + self.b_fc)
        return self._normalize(z)


class SeparableVariant(ConvVariant):
    """深度可分离: depthwise 3×3 → pointwise 1×1, 参数量约 1/C_in。"""
    name = "separable"

    def __init__(self, **kw):
        super().__init__(**kw)
        c0 = self.num_colors
        ch = self.channels
        self.dw1 = _he_init((c0, 1, 3, 3), self.rng)
        self.bd1 = np.zeros(c0, dtype=np.float32)
        self.pw1 = _he_init((ch[0], c0, 1, 1), self.rng)
        self.bp1 = np.zeros(ch[0], dtype=np.float32)
        self.dw2 = _he_init((ch[0], 1, 3, 3), self.rng)
        self.bd2 = np.zeros(ch[0], dtype=np.float32)
        self.pw2 = _he_init((ch[1], ch[0], 1, 1), self.rng)
        self.bp2 = np.zeros(ch[1], dtype=np.float32)
        self.dw3 = _he_init((ch[1], 1, 3, 3), self.rng)
        self.bd3 = np.zeros(ch[1], dtype=np.float32)
        self.pw3 = _he_init((ch[2], ch[1], 1, 1), self.rng)
        self.bp3 = np.zeros(ch[2], dtype=np.float32)
        self.w_fc = _he_init((ch[2], self.embed_dim), self.rng)
        self.b_fc = np.zeros(self.embed_dim, dtype=np.float32)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        x = self._one_hot(grid)
        h = _relu(_pointwise_conv2d(_depthwise_conv2d(x, self.dw1, self.bd1), self.pw1, self.bp1))
        h = _relu(_pointwise_conv2d(_depthwise_conv2d(h, self.dw2, self.bd2), self.pw2, self.bp2))
        h = _relu(_pointwise_conv2d(_depthwise_conv2d(h, self.dw3, self.bd3), self.pw3, self.bp3))
        pooled = h.mean(axis=(1, 2))
        z = _relu(pooled @ self.w_fc + self.b_fc)
        return self._normalize(z)


class ResidualVariant(ConvVariant):
    """残差连接: 每层 h = relu(conv(x)) + x_proj, 梯度流更稳。"""
    name = "residual"

    def __init__(self, **kw):
        super().__init__(**kw)
        c0 = self.num_colors
        ch = self.channels
        self.w1 = _he_init((ch[0], c0, 3, 3), self.rng)
        self.b1 = np.zeros(ch[0], dtype=np.float32)
        self.proj1 = _he_init((c0, ch[0]), self.rng)
        self.w2 = _he_init((ch[1], ch[0], 3, 3), self.rng)
        self.b2 = np.zeros(ch[1], dtype=np.float32)
        self.proj2 = _he_init((ch[0], ch[1]), self.rng)
        self.w3 = _he_init((ch[2], ch[1], 3, 3), self.rng)
        self.b3 = np.zeros(ch[2], dtype=np.float32)
        self.proj3 = _he_init((ch[1], ch[2]), self.rng)
        self.w_fc = _he_init((ch[2], self.embed_dim), self.rng)
        self.b_fc = np.zeros(self.embed_dim, dtype=np.float32)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        x = self._one_hot(grid)
        c = _relu(_conv2d_stride2(x, self.w1, self.b1))
        pooled_x = x.mean(axis=(1, 2))
        skip = pooled_x @ self.proj1
        h = c + skip[:, None, None]
        c = _relu(_conv2d_stride2(h, self.w2, self.b2))
        pooled_h = h.mean(axis=(1, 2))
        skip = pooled_h @ self.proj2
        h = c + skip[:, None, None]
        c = _relu(_conv2d_stride2(h, self.w3, self.b3))
        pooled_h = h.mean(axis=(1, 2))
        skip = pooled_h @ self.proj3
        h = c + skip[:, None, None]
        pooled = h.mean(axis=(1, 2))
        z = _relu(pooled @ self.w_fc + self.b_fc)
        return self._normalize(z)


class AttentionVariant(ConvVariant):
    """通道注意力: conv → SE squeeze-excite → conv, 自适应重标通道。"""
    name = "attention"

    def __init__(self, reduction: int = 4, **kw):
        super().__init__(**kw)
        self.reduction = reduction
        c0 = self.num_colors
        ch = self.channels
        self.w1 = _he_init((ch[0], c0, 3, 3), self.rng)
        self.b1 = np.zeros(ch[0], dtype=np.float32)
        self.se1_w1 = _he_init((max(1, ch[0] // reduction), ch[0]), self.rng)
        self.se1_b1 = np.zeros(max(1, ch[0] // reduction), dtype=np.float32)
        self.se1_w2 = _he_init((ch[0], max(1, ch[0] // reduction)), self.rng)
        self.se1_b2 = np.zeros(ch[0], dtype=np.float32)
        self.w2 = _he_init((ch[1], ch[0], 3, 3), self.rng)
        self.b2 = np.zeros(ch[1], dtype=np.float32)
        self.se2_w1 = _he_init((max(1, ch[1] // reduction), ch[1]), self.rng)
        self.se2_b1 = np.zeros(max(1, ch[1] // reduction), dtype=np.float32)
        self.se2_w2 = _he_init((ch[1], max(1, ch[1] // reduction)), self.rng)
        self.se2_b2 = np.zeros(ch[1], dtype=np.float32)
        self.w3 = _he_init((ch[2], ch[1], 3, 3), self.rng)
        self.b3 = np.zeros(ch[2], dtype=np.float32)
        self.w_fc = _he_init((ch[2], self.embed_dim), self.rng)
        self.b_fc = np.zeros(self.embed_dim, dtype=np.float32)

    def _se(self, x, w1, b1, w2, b2):
        return _channel_attention(x, w1, b1, w2, b2, self.reduction)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        x = self._one_hot(grid)
        h = _relu(_conv2d_stride2(x, self.w1, self.b1))
        h = self._se(h, self.se1_w1, self.se1_b1, self.se1_w2, self.se1_b2)
        h = _relu(_conv2d_stride2(h, self.w2, self.b2))
        h = self._se(h, self.se2_w1, self.se2_b1, self.se2_w2, self.se2_b2)
        h = _relu(_conv2d_stride2(h, self.w3, self.b3))
        pooled = h.mean(axis=(1, 2))
        z = _relu(pooled @ self.w_fc + self.b_fc)
        return self._normalize(z)


class WideVariant(ConvVariant):
    """更宽通道: 2× channels, 表达能力更强但参数更多。"""
    name = "wide"

    def __init__(self, **kw):
        kw.setdefault("channels", (24, 48, 64))
        super().__init__(**kw)
        c0 = self.num_colors
        ch = self.channels
        self.w1 = _he_init((ch[0], c0, 3, 3), self.rng)
        self.b1 = np.zeros(ch[0], dtype=np.float32)
        self.w2 = _he_init((ch[1], ch[0], 3, 3), self.rng)
        self.b2 = np.zeros(ch[1], dtype=np.float32)
        self.w3 = _he_init((ch[2], ch[1], 3, 3), self.rng)
        self.b3 = np.zeros(ch[2], dtype=np.float32)
        self.w_fc = _he_init((ch[2], self.embed_dim), self.rng)
        self.b_fc = np.zeros(self.embed_dim, dtype=np.float32)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        x = self._one_hot(grid)
        h = _relu(_conv2d_stride2(x, self.w1, self.b1))
        h = _relu(_conv2d_stride2(h, self.w2, self.b2))
        h = _relu(_conv2d_stride2(h, self.w3, self.b3))
        pooled = h.mean(axis=(1, 2))
        z = _relu(pooled @ self.w_fc + self.b_fc)
        return self._normalize(z)


class DeepVariant(ConvVariant):
    """更深: 4 层卷积, 更深的非线性变换。"""
    name = "deep"

    def __init__(self, **kw):
        kw.setdefault("channels", (12, 20, 28, 32))
        super().__init__(**kw)
        c0 = self.num_colors
        ch = self.channels
        if len(ch) < 4:
            ch = ch + (ch[-1],)
            self.channels = ch
        self.w1 = _he_init((ch[0], c0, 3, 3), self.rng)
        self.b1 = np.zeros(ch[0], dtype=np.float32)
        self.w2 = _he_init((ch[1], ch[0], 3, 3), self.rng)
        self.b2 = np.zeros(ch[1], dtype=np.float32)
        self.w3 = _he_init((ch[2], ch[1], 3, 3), self.rng)
        self.b3 = np.zeros(ch[2], dtype=np.float32)
        self.w4 = _he_init((ch[3], ch[2], 3, 3), self.rng)
        self.b4 = np.zeros(ch[3], dtype=np.float32)
        self.w_fc = _he_init((ch[3], self.embed_dim), self.rng)
        self.b_fc = np.zeros(self.embed_dim, dtype=np.float32)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        x = self._one_hot(grid)
        h = _relu(_conv2d_stride2(x, self.w1, self.b1))
        h = _relu(_conv2d_stride2(h, self.w2, self.b2))
        h = _relu(_conv2d_stride2(h, self.w3, self.b3))
        h = _relu(_conv2d_stride2(h, self.w4, self.b4))
        pooled = h.mean(axis=(1, 2))
        z = _relu(pooled @ self.w_fc + self.b_fc)
        return self._normalize(z)


_VARIANT_REGISTRY: Dict[str, Type[ConvVariant]] = {
    "base": BaseVariant,
    "dilated": DilatedVariant,
    "separable": SeparableVariant,
    "residual": ResidualVariant,
    "attention": AttentionVariant,
    "wide": WideVariant,
    "deep": DeepVariant,
}

VARIANT_NAMES: Tuple[str, ...] = tuple(_VARIANT_REGISTRY.keys())


def build_variant(name: str, **kw) -> ConvVariant:
    """按名称构造卷积变体。"""
    cls = _VARIANT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"未知变体 '{name}', 可用: {VARIANT_NAMES}")
    return cls(**kw)