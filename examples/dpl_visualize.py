"""DPL 几何投影可视化 — 保存 PNG 图像。

运行方式：
    python examples/dpl_visualize.py
"""
import os
import numpy as np

from lingjing_solo.dpl.types import UniversalConstants
from lingjing_solo.dpl.projection import project


def save_visualization(obs: np.ndarray, ps, output_dir: str = "examples/output"):
    """保存 D1-D3 可视化图像。

    输出三张图：
    1. 01_original_obs.png — 原始观测场
    2. 02_d1_gradient.png — D1 梯度场
    3. 03_d2_curvature.png — D2 曲率热力图
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无显示环境
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 不可用，跳过可视化")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 图 1：原始观测
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(obs, cmap="viridis")
    ax.set_title("Original Observation (2D Field)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_original_obs.png"), dpi=100)
    plt.close()
    print(f"  Saved: 01_original_obs.png")

    # 图 2：D1 梯度场
    if ps.D1.available:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(ps.D1.gradient, cmap="hot")
        axes[0].set_title("D1 Gradient Magnitude")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(ps.D1.divergence, cmap="RdBu", vmin=-np.abs(ps.D1.divergence).max(), vmax=np.abs(ps.D1.divergence).max())
        axes[1].set_title("D1 Divergence")
        plt.colorbar(im1, ax=axes[1])

        plt.suptitle("D1 Information Flow Lines", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "02_d1_gradient.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: 02_d1_gradient.png")

    # 图 3：D2 曲率 + 涡旋
    if ps.D2.available:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(ps.D2.curvature, cmap="RdBu")
        axes[0].set_title("D2 Gaussian Curvature")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(ps.D2.vorticity, cmap="RdBu")
        axes[1].set_title("D2 Vorticity")
        plt.colorbar(im1, ax=axes[1])

        plt.suptitle("D2 Information Surface", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_d2_curvature.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: 03_d2_curvature.png")

    print(f"\n  All visualizations saved to: {output_dir}/")


def main():
    print("DPL 可视化生成中...")

    # 高斯势阱
    size = 32
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    obs = np.exp(-(X**2 + Y**2) / 2.0).astype(np.float32)

    c = UniversalConstants()
    ps = project(obs, has_temporal=False, constants=c)

    save_visualization(obs, ps, output_dir="examples/output")


if __name__ == "__main__":
    main()
