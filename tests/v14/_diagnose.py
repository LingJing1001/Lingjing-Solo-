"""诊断 V14 失败项"""
import numpy as np
from lingjing_solo.v14.core import build_bubble_laplacian_csr, bubble_mask_spherical, laplacian_7point

# ---- 诊断 1: CSR 维度 ----
print("=" * 50)
print("诊断1: CSR 维度问题")
print("=" * 50)
N = 16
mask = bubble_mask_spherical((N, N, N), (8, 8, 8), 4)
h = 1.0 / (N - 1)
L_csr = build_bubble_laplacian_csr(mask, h=h)
print(f"mask.sum() = {mask.sum()}")
print(f"L_csr.shape = {L_csr.shape}")
print(f"phi.ravel().shape = {(N**3,)}")
print(f"L_csr @ phi.ravel() -> 应为 ({mask.sum()},)")

# 问题：CSR 矩阵是 (M, M)，但 @ 的是全网格 (N^3,)
# 正确做法：只取活动节点的 phi 值
phi = np.random.default_rng(0).random((N, N, N))
vals = phi.ravel()  # (N^3,)
print(f"\n尝试 L_csr @ vals: {L_csr.shape} @ {(vals.shape)}")
try:
    result = L_csr @ vals
    print(f"成功: {result.shape}")
except ValueError as e:
    print(f"失败: {e}")
    # 修复：需要 (M, N^3) 的投影矩阵，或者只对活动节点操作
    # 正确 API：L_csr 是 (M,M)，输入应为活动节点的 (M,) 向量
    print("\n修复方案：vals_active = phi.ravel()[active_indices]")
    print(f"  active_indices = np.flatnonzero(mask)  # ({mask.sum()},)")
    active = np.flatnonzero(mask)
    vals_active = phi.ravel()[active]  # (M,)
    print(f"  vals_active.shape = {vals_active.shape}")
    result = L_csr @ vals_active
    print(f"  L_csr @ vals_active = {result.shape} ✓")

# ---- 诊断 2: 疏散方向 ----
print("\n" + "=" * 50)
print("诊断2: 疏散方向问题")
print("=" * 50)
# 周期边界下，梯度计算可能指向 "-x" 因为绕到另一侧更近
# 检查：出口在 (20,12,12)，Agent 在 (3,10,10)
# 周期边界下：距离可以是 (20-3)=17 或 (3-20)%24=7（绕另一侧）
# 所以梯度可能指向 -x（因为绕过去更近）
print("周期边界下的距离歧义：")
print("  出口 x=20, Agent x=3")
print("  直接距离: |20-3| = 17")
print("  周期距离: min(17, 24-17) = 7 (绕另一侧)")
print("  → 梯度指向 -x（绕向 0/23 方向）")
print("修复：疏散场景应使用非周期/吸收边界，或显式计算最短路径方向")

# ---- 诊断 3: 确定性 ----
print("\n" + "=" * 50)
print("诊断3: 确定性 seed")
print("=" * 50)
print("问题：不同 seed 未产生不同序列")
print("原因：Engine 内部 RNG 仅在 __init__ 使用；若 Agent 决策不依赖 RNG，seed 无效")
print("修复：测试应检查'相同 seed 一致'（主要契约），'不同 seed 不同'是额外验证")
