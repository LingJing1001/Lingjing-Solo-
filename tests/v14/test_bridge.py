"""
V14.3.1 契约测试 —— 2D→3D 切片桥接。

  C18 phi_blocks_to_3d_slice 形状正确 (B,B,1)
  C19 曲率桥接：均匀 blocks → 曲率≈0；非均匀 blocks → 曲率>0
"""
import numpy as np

from lingjing_solo.core.phi import phi_blocks_to_3d_slice, curvature_from_2d_blocks


def test_c18_blocks_to_3d_slice_shape():
    """C18：2D blocks (B,B) → 3D slice (B,B,1)。"""
    blocks = np.random.rand(8, 8).astype(np.float64)
    slice3d = phi_blocks_to_3d_slice(blocks, h=1.0 / 7)
    assert slice3d.shape == (8, 8, 1), f"shape = {slice3d.shape}, expected (8,8,1)"
    # z=0 平面的值与原 blocks 一致
    assert np.allclose(slice3d[:, :, 0], blocks), "z=0 切片值与 blocks 不一致"


def test_c19_curvature_bridge_uniform_vs_nonuniform():
    """C19：均匀 blocks 曲率≈0；高斯斑 blocks 曲率显著>0。"""
    h = 1.0 / 7

    # 均匀 blocks = Φ₀ = 0 → 平坦 → 曲率≈0，度规迹=3
    uniform_blocks = np.zeros((8, 8), dtype=np.float64)
    R_uniform, tr_dev_uniform = curvature_from_2d_blocks(uniform_blocks, h=h)
    assert abs(R_uniform) < 1e-6, f"零场 blocks 曲率非零: R={R_uniform:.2e}"
    assert tr_dev_uniform < 1e-6, f"零场 blocks 度规迹偏离: dev={tr_dev_uniform:.2e}"

    # 高斯斑 blocks → 非平坦 → 曲率显著>0
    i = np.arange(8)[:, None]
    j = np.arange(8)[None, :]
    gauss_blocks = 0.5 * np.exp(-((i - 3.5) ** 2 + (j - 3.5) ** 2) / 2.0)
    R_gauss, tr_dev_gauss = curvature_from_2d_blocks(gauss_blocks, h=h)
    # 高斯斑梯度大 → 曲率代理应显著大于均匀场
    assert tr_dev_gauss > tr_dev_uniform + 0.01, \
        f"高斯斑未弯曲度规: uniform_dev={tr_dev_uniform:.4e}, gauss_dev={tr_dev_gauss:.4e}"
