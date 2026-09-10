"""Pure-numpy CNN encoder for ARC grids (optional torch backend).

Design:
  - Input: HxW integer color indices (0..C-1), typically 64x64 / 16 colors
  - Pipeline: one-hot → 3×3 stride-2 conv stack → global pool → linear
  - No torch required; PerceptionEncoder can still prefer torch when present.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=np.float32)


def _conv2d_stride2(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """x: (C_in, H, W), weight: (C_out, C_in, 3, 3), bias: (C_out,).

    Vectorized via as_strided im2col (stride-2, pad-1).
    """
    c_in, h, w = x.shape
    c_out = int(weight.shape[0])
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    # strided windows: (C_in, h2, w2, 3, 3)
    s0, s1, s2 = xp.strides
    shape = (c_in, h2, w2, 3, 3)
    strides = (s0, s1 * 2, s2 * 2, s1, s2)
    patches = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    # (C_out, C_in, 3, 3) · (C_in, h2, w2, 3, 3) → (C_out, h2, w2)
    out = np.einsum("oijk,ihwjk->ohw", weight, patches, optimize=True)
    out = out + bias[:, None, None]
    return out.astype(np.float32, copy=False)


def _he_init(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    std = np.sqrt(2.0 / max(1, fan_in))
    return (rng.standard_normal(shape) * std).astype(np.float32)


class NumpyCNN:
    """Lightweight 3-layer stride-2 CNN → fixed embedding (default 64-d)."""

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
        c0 = self.num_colors
        self.w1 = _he_init((self.channels[0], c0, 3, 3), self.rng)
        self.b1 = np.zeros(self.channels[0], dtype=np.float32)
        self.w2 = _he_init((self.channels[1], self.channels[0], 3, 3), self.rng)
        self.b2 = np.zeros(self.channels[1], dtype=np.float32)
        self.w3 = _he_init((self.channels[2], self.channels[1], 3, 3), self.rng)
        self.b3 = np.zeros(self.channels[2], dtype=np.float32)
        self.w_fc = _he_init((self.channels[2], self.embed_dim), self.rng)
        self.b_fc = np.zeros(self.embed_dim, dtype=np.float32)
        self._n_encode = 0

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
        """Return L2-normalized embedding of shape (embed_dim,)."""
        x = self._one_hot(grid)
        h = _relu(_conv2d_stride2(x, self.w1, self.b1))
        h = _relu(_conv2d_stride2(h, self.w2, self.b2))
        h = _relu(_conv2d_stride2(h, self.w3, self.b3))
        pooled = h.mean(axis=(1, 2))
        z = pooled @ self.w_fc + self.b_fc
        z = _relu(z)
        n = float(np.linalg.norm(z) + 1e-8)
        self._n_encode += 1
        return (z / n).astype(np.float32)

    def encode_padded(self, grid: np.ndarray, dim: int) -> np.ndarray:
        z = self.encode(grid)
        out = np.zeros(int(dim), dtype=np.float32)
        n = min(len(z), int(dim))
        out[:n] = z[:n]
        return out


def try_torch_encode(grid: np.ndarray, embed_dim: int = 64) -> Optional[np.ndarray]:
    """Optional torch path; returns None if torch missing or fails."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    try:
        g = np.asarray(grid, dtype=np.float32)
        if g.ndim != 2:
            return None
        if g.shape[0] >= 58:
            g = g[:58]
        ys = np.linspace(0, g.shape[0], 32, endpoint=False).astype(np.int32)
        xs = np.linspace(0, g.shape[1], 32, endpoint=False).astype(np.int32)
        g = g[ys][:, xs] / 15.0
        x = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0)
        torch.manual_seed(7)
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, embed_dim),
            nn.ReLU(),
        )
        model.eval()
        with torch.no_grad():
            z = model(x).squeeze(0).numpy().astype(np.float32)
        n = float(np.linalg.norm(z) + 1e-8)
        return z / n
    except Exception:
        return None
