"""Layer 0/L3 · 感知编码 = 协议 4「求解场 → 感知世界」

泡壁思想：全盘每步送 LLM 浪费；只在变化像素周围 ROI 做高精度。
输出 PerceptionSnapshot：特征 + 对象 + Φ + 泡壁 + 弯曲代理。
"""
from __future__ import annotations

import numpy as np
from ..core import (
    SoloConfig, GameObject, Logger, PerceptionSnapshot,
    compute_phi, compute_bubble, curvature_proxy,
)


class PerceptionEncoder:
    def __init__(self, cfg: SoloConfig, logger: Logger = None):
        self.cfg = cfg
        self.log = logger or Logger()
        self._ar25 = None
        self._cnn = None
        try:
            from .cnn import LingjingCNN

            self._cnn = LingjingCNN(cfg)
            self._cnn.build()
        except Exception:
            self._cnn = None

    @property
    def ar25(self):
        """Lazy AR25 encoder (Layer 0 game-specific)."""
        if self._ar25 is None:
            from .ar25_encoder import Ar25Encoder
            self._ar25 = Ar25Encoder()
        return self._ar25

    def compute_delta(self, prev: np.ndarray, curr: np.ndarray):
        if prev is None:
            H, W = curr.shape
            return [(i, j) for i in range(H) for j in range(W)]
        return [
            (int(i), int(j))
            for i, j in zip(*np.where(prev != curr))
        ]

    def segment(self, grid: np.ndarray, delta_pixels=None) -> list[GameObject]:
        H, W = grid.shape
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        region = set(delta_pixels) if delta_pixels else None
        for i in range(H):
            for j in range(W):
                if region and (i, j) not in region:
                    continue
                c = int(grid[i, j])
                if c == 0:
                    continue
                key = (i, j)
                parent.setdefault(key, key)
                for di, dj in [(-1, 0), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and int(grid[ni, nj]) == c:
                        neigh = (ni, nj)
                        if neigh in parent:
                            union(key, neigh)

        groups = {}
        for (i, j), root in parent.items():
            groups.setdefault(find(root), []).append((i, j))

        objs = []
        for pixs in groups.values():
            color = int(grid[pixs[0]])
            ys = [p[0] for p in pixs]
            xs = [p[1] for p in pixs]
            objs.append(GameObject(
                color=color, pixels=pixs,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
            ))
        return objs

    def encode(self, grid: np.ndarray) -> np.ndarray:
        if self._cnn is not None:
            try:
                feat = self._cnn.forward(grid)
                if feat is not None and getattr(feat, "size", 0):
                    dim = int(self.cfg.cnn_feature_dim)
                    out = np.zeros(dim, dtype=np.float32)
                    # fuse: CNN front + histogram/block fallback tail for stability
                    fb = self._fallback_features(grid)
                    n = min(dim, int(feat.shape[0]))
                    out[:n] = np.asarray(feat[:n], dtype=np.float32)
                    # blend last half with classical stats so online heads stay grounded
                    half = dim // 2
                    out[half:] = 0.65 * out[half:] + 0.35 * fb[half:]
                    return out
            except Exception:
                pass
        return self._fallback_features(grid)

    def _fallback_features(self, grid: np.ndarray) -> np.ndarray:
        H, W = grid.shape
        dim = self.cfg.cnn_feature_dim
        feat = np.zeros(dim, dtype=np.float32)
        hist, _ = np.histogram(grid.flatten(), bins=self.cfg.num_colors, range=(0, self.cfg.num_colors))
        if hist.sum() > 0:
            hist = hist / hist.sum()
        end = min(self.cfg.num_colors, dim)
        feat[:end] = hist[:end]
        q = 4
        idx = self.cfg.num_colors
        for bi in range(q):
            for bj in range(q):
                if idx >= dim:
                    break
                yi0, yi1 = int(H * bi / q), int(H * (bi + 1) / q)
                xj0, xj1 = int(W * bj / q), int(W * (bj + 1) / q)
                block = grid[yi0:yi1, xj0:xj1]
                feat[idx] = float(block.mean()) / self.cfg.num_colors
                idx += 1
        if idx < dim:
            ys, xs = np.where(grid > 0)
            if len(xs) > 0:
                feat[idx] = float(xs.mean()) / W
                idx += 1
                feat[idx] = float(ys.mean()) / H
                idx += 1
                feat[idx] = float(grid.var())
        return feat

    def perceive(self, prev, curr, version: int = 0, tick: int = 0,
                 levels: int = 0, env_state: str = "NOT_FINISHED") -> PerceptionSnapshot:
        """协议 4 主入口：求解局部场 → PerceptionSnapshot。"""
        delta = self.compute_delta(prev, curr)
        # 泡壁内分割（面积律：只在 ROI 高精度）
        bubble = compute_bubble(prev, curr, pad=self.cfg.bubble_pad, grid_size=self.cfg.grid_size)
        roi_pixels = delta if delta else None
        objs = self.segment(curr, delta_pixels=roi_pixels) if roi_pixels else []
        feat = self.encode(curr)
        phi = compute_phi(curr, block=self.cfg.phi_block_size)
        curv = curvature_proxy(phi)
        return PerceptionSnapshot(
            version=version,
            tick=tick,
            feature=feat,
            objects=objs,
            delta_pixels=delta if isinstance(delta, list) else list(delta),
            bubble=bubble,
            phi=phi,
            curvature_proxy=curv,
            levels=levels,
            env_state=env_state,
        )

    def __call__(self, prev: np.ndarray, curr: np.ndarray) -> dict:
        """向后兼容：返回 dict；新代码请用 perceive()。"""
        snap = self.perceive(prev, curr)
        return {
            "feature": snap.feature,
            "delta_pixels": snap.delta_pixels,
            "objects": snap.objects,
            "bubble": snap.bubble,
            "phi": snap.phi,
            "curvature_proxy": snap.curvature_proxy,
            "snapshot": snap,
        }

    def encode_ar25_level(self, level_index: int, source_path=None):
        """R2 AR25 path: structured obs from engine source (not raw frame)."""
        from .ar25_encoder import Ar25Encoder
        return Ar25Encoder(source_path).encode_level(level_index)
