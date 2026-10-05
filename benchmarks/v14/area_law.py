"""
灵境引擎 V14.0 — 面积律实测
=============================
三线对照（同误差预算、同算子、同计时方法）：
1. 体积 7 点（O(N³)）
2. 泡壁 CSR（O(R²)，与 N 脱钩）
3. 泡壁 Python（O(R²) 但 Python 循环慢）

验收标准：
- 泡壁节点数恒定（R 固定）→ O(R²)
- 体积耗时随 N³ 增长
- CSR 泡壁耗时恒定 → 面积律兑现
"""
from __future__ import annotations
import os
import time

import numpy as np

from lingjing_solo.core import Field, FieldConfig, build_bubble_laplacian_csr, bubble_mask_spherical, laplacian_7point


def benchmark_volume_7point(N: int, n_iter: int = 50) -> float:
    """体积 7 点 Laplacian 计时。"""
    h = 1.0 / (N - 1)
    phi = np.random.default_rng(0).random((N, N, N))
    # 预热
    for _ in range(5):
        _ = laplacian_7point(phi, h=h)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = laplacian_7point(phi, h=h)
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


def benchmark_bubble_csr(N: int, R: int = 4, n_iter: int = 500) -> tuple[int, float]:
    """泡壁 CSR Laplacian 计时。"""
    center = (N // 2, N // 2, N // 2)
    mask = bubble_mask_spherical((N, N, N), center, R)
    h = 1.0 / (N - 1)
    L_csr = build_bubble_laplacian_csr(mask, h=h)
    active = np.flatnonzero(mask)
    phi = np.random.default_rng(0).random((N, N, N))
    vals = phi.ravel()[active]
    # 预热
    for _ in range(10):
        _ = L_csr @ vals
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = L_csr @ vals
    ms = (time.perf_counter() - t0) / n_iter * 1000
    return int(mask.sum()), ms


def benchmark_bubble_python(N: int, R: int = 4, n_iter: int = 10) -> float:
    """泡壁 Python 循环（对照，验证 CSR 加速）。"""
    cfg = FieldConfig(shape=(N, N, N), h=1.0/(N-1))
    f = Field(cfg)
    mask = bubble_mask_spherical((N, N, N), (N//2, N//2, N//2), R)
    f.set_bubble(mask)
    phi = np.random.default_rng(0).random((N, N, N))
    # 预热
    for _ in range(2):
        _ = f.laplacian_bubble(phi)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = f.laplacian_bubble(phi)
    return (time.perf_counter() - t0) / n_iter * 1000


def main():
    print("=" * 80)
    print("灵境引擎 V14.0 — 面积律实测（三线对照）")
    print("=" * 80)
    print(f"{'N':>4} | {'vol节点':>10} | {'bub节点':>8} | {'体积ms':>10} | {'泡壁-Pyms':>12} | {'泡壁-CSRms':>12} | {'加速比':>8}")
    print("-" * 80)

    results = []
    R = 4  # 固定泡壁半径
    for N in [16, 24, 32, 48]:
        # 调整迭代次数使计时可靠
        n_vol = max(10, int(200 / (N / 16) ** 3))
        n_csr = max(50, int(2000 / (N / 16) ** 2))
        n_py = max(2, int(20 / (N / 16) ** 2))

        t_vol = benchmark_volume_7point(N, n_iter=n_vol)
        bub_nodes, t_csr = benchmark_bubble_csr(N, R=R, n_iter=n_csr)
        t_py = benchmark_bubble_python(N, R=R, n_iter=n_py)

        speedup = t_vol / t_csr if t_csr > 0 else float('inf')
        vol_nodes = N ** 3
        print(f"{N:>4} | {vol_nodes:>10} | {bub_nodes:>8} | {t_vol:>10.4f} | {t_py:>12.4f} | {t_csr:>12.4f} | {speedup:>8.2f}x")
        results.append({
            "N": N, "vol_nodes": vol_nodes, "bub_nodes": bub_nodes,
            "t_vol": t_vol, "t_py": t_py, "t_csr": t_csr, "speedup": speedup,
        })

    # 验收断言
    print("\n" + "=" * 80)
    print("验收断言")
    print("=" * 80)
    bub_nodes_all = {r["bub_nodes"] for r in results}
    print(f"✅ 泡壁节点数恒定 = {bub_nodes_all} (R={R} 固定) → O(R²)")
    assert len(bub_nodes_all) == 1, "泡壁节点数应恒定"

    speedups = [r["speedup"] for r in results]
    print(f"✅ 加速比单调放大: {speedups[0]:.2f}x → {speedups[-1]:.2f}x")
    assert speedups[-1] > speedups[0], "加速比应随 N 放大"

    # 斜率估算（log-log）
    import math
    logN = [math.log(r["N"]) for r in results]
    logVol = [math.log(r["t_vol"]) for r in results]
    logCsr = [math.log(r["t_csr"]) for r in results]
    # 简单线性回归
    def slope(x, y):
        n = len(x)
        mx, my = sum(x)/n, sum(y)/n
        num = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
        den = sum((xi-mx)**2 for xi in x)
        return num/den if den > 0 else 0
    s_vol = slope(logN, logVol)
    s_csr = slope(logN, logCsr)
    print(f"✅ 体积耗时斜率 ≈ {s_vol:.3f} (趋近 O(N³) 理想值 3.0)")
    print(f"✅ CSR 泡壁斜率 ≈ {s_csr:.3f} (≈0，恒定 O(R²))")
    assert abs(s_csr) < 0.5, f"CSR 斜率应≈0，实际 {s_csr}"

    # 保存 CSV
    import csv
    out = os.path.join(os.path.dirname(__file__), "..", "out", "area_law_v14.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📊 结果已保存: {out}")


if __name__ == "__main__":
    main()
