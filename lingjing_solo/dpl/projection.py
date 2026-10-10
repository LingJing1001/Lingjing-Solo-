"""流形分解接口：project（附录 B B.4 + 附录 C）。"""
from __future__ import annotations

import numpy as np

from .types import (
    ProjectedState, DimensionInfo, D1FlowLine, D2Surface, D3Volume, D4Spacetime,
    UniversalConstants,
)
from .dimension import detect_dimension


def project(env_state: np.ndarray, has_temporal: bool, constants: UniversalConstants) -> ProjectedState:
    """将环境状态投影到 D1-D4 信息流形层级。

    Args:
        env_state: 环境观测张量
        has_temporal: 是否含时间维度
        constants: 已校准的宇宙常数

    Returns:
        ProjectedState: D1-D4 流形特征集合
    """
    dim_info = detect_dimension(env_state, has_temporal)
    I_norm = env_state / constants.I_0

    # D1：信息流线维
    d1 = _compute_d1(I_norm, constants.D)

    # D2：信息面维
    d2 = _compute_d2(I_norm, dim_info)

    # D3：信息体维（2D 也计算拓扑欧拉示性数）
    d3 = _compute_d3(I_norm, dim_info)

    # D4：时空信息流维（赝时间映射，单帧也可用）
    d4 = _compute_d4(I_norm, d1, d2, dim_info)

    return ProjectedState(
        D1=d1, D2=d2, D3=d3, D4=d4,
        dim_info=dim_info,
        input_shape=env_state.shape,
    )


def _compute_d1(I_norm: np.ndarray, D: float) -> D1FlowLine:
    """D1 信息流线维：梯度、散度、方向（附录 C C.3）。"""
    if I_norm.ndim < 2:
        return D1FlowLine(available=False)

    # 梯度向量场
    grads = np.gradient(I_norm)
    if not isinstance(grads, (list, tuple)):
        grads = [grads]
    gradient_vector = np.stack(grads, axis=-1)  # (..., ndim)

    # 梯度模
    gradient = np.linalg.norm(gradient_vector, axis=-1)

    # 信息流 J_I = -D * grad I
    J_I = -D * gradient_vector

    # 散度 ∇·J_I = -D * ∇²I
    divergence = np.zeros_like(I_norm)
    for i, g in enumerate(grads):
        divergence += np.gradient(g, axis=i)
    divergence *= -D

    # 主流线方向：全局平均梯度方向归一化
    mean_grad = np.mean(gradient_vector, axis=tuple(range(gradient_vector.ndim - 1)))
    grad_norm = np.linalg.norm(mean_grad)
    if grad_norm > 1e-12:
        direction = mean_grad / grad_norm
    else:
        direction = np.zeros_like(mean_grad)

    return D1FlowLine(
        gradient=gradient,
        divergence=divergence,
        direction=direction,
        gradient_vector=gradient_vector,
        available=True,
    )


def _compute_d2(I_norm: np.ndarray, dim_info: DimensionInfo) -> D2Surface:
    """D2 信息面维：Hessian、涡旋、曲率（附录 C C.4）。"""
    if I_norm.ndim < 2:
        return D2Surface(available=False)

    if I_norm.ndim == 2:
        # 2D 网格
        Ix = np.gradient(I_norm, axis=0)
        Iy = np.gradient(I_norm, axis=1)

        # Hessian
        Ixx = np.gradient(Ix, axis=0)
        Iyy = np.gradient(Iy, axis=1)
        Ixy = np.gradient(Ix, axis=1)
        hessian = np.stack([Ixx, Ixy, Ixy, Iyy], axis=-1).reshape(*I_norm.shape, 2, 2)

        # 高斯曲率 K
        det_H = Ixx * Iyy - Ixy**2
        denom = (1 + Ix**2 + Iy**2) ** 2
        curvature = det_H / (denom + 1e-12)

        # 涡旋（2D 标量涡旋）
        v_Ix = -Ix / (np.abs(I_norm) + 1e-12)
        v_Iy = -Iy / (np.abs(I_norm) + 1e-12)
        vorticity = np.gradient(v_Iy, axis=0) - np.gradient(v_Ix, axis=1)

        return D2Surface(
            hessian=hessian,
            vorticity=vorticity,
            curvature=curvature,
            vorticity_is_vector=False,  # 2D 标量
            available=True,
        )
    else:
        # 3D+：简化版，只算标量曲率代理
        return D2Surface(available=False)


def _compute_d3(I_norm: np.ndarray, dim_info: DimensionInfo) -> D3Volume:
    """D3 信息体维：密度、欧拉示性数、Ricci 标量（附录 C C.5）。

    2D 场景：计算图像拓扑的欧拉示性数（连通域数 - 孔洞数）。
    3D+ 场景：密度场 + 简化拓扑。
    """
    density = np.abs(I_norm)

    if dim_info.spatial >= 3.0:
        # 3D+ 简化版
        return D3Volume(
            density=density,
            euler_char=0,
            ricci_scalar=None,
            available=True,
        )

    # 2D 场景：计算欧拉示性数 χ = C - H
    # 用 50% 分位数二值化
    threshold = np.percentile(I_norm, 50)
    binary = (I_norm > threshold).astype(np.int32)

    # 计算连通域数 C 和孔洞数 H
    euler_char = _compute_2d_euler_char(binary)

    return D3Volume(
        density=density,
        euler_char=euler_char,
        ricci_scalar=None,
        available=True,
    )


def _compute_2d_euler_char(binary: np.ndarray) -> int:
    """计算 2D 二值图像的欧拉示性数 χ = C - H。

    用连通域标记法：
    - 4-连通前景：连通域数 C
    - 8-连通背景：孔洞数 H
    - χ = C - H
    """
    try:
        from scipy import ndimage
    except ImportError:
        # scipy 不可用时用简化版
        return _compute_euler_char_fallback(binary)

    # 标记前景连通域（4-连通）
    structure_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled_fg, n_components = ndimage.label(binary, structure=structure_4)

    # 标记背景连通域（8-连通）
    structure_8 = np.ones((3, 3), dtype=int)
    labeled_bg, n_background = ndimage.label(1 - binary, structure=structure_8)

    # 孔洞数 = 背景连通域数 - 1（减去图像外边界那个）
    n_holes = n_background - 1

    # 欧拉示性数 = 连通域数 - 孔洞数
    euler = n_components - n_holes
    return int(euler)


def _compute_euler_char_fallback(binary: np.ndarray) -> int:
    """scipy 不可用时的 fallback：用前景像素数近似。"""
    n_fg = int(np.sum(binary > 0))
    return n_fg



def _compute_d4(
    I_norm: np.ndarray,
    d1: D1FlowLine,
    d2: D2Surface,
    dim_info: DimensionInfo,
) -> D4Spacetime:
    """D4 时空信息流维：时间箭头、赝时间映射、因果连接（附录 C C.6）。

    单帧输入也可用：用空间梯度构造赝时间演化。
    """
    if I_norm.ndim < 2:
        return D4Spacetime(available=False, is_fake_time=True)

    # 1. 时间箭头推断：从梯度场的不对称性
    #    如果梯度场有明显的主导方向 → 有时间箭头
    #    如果对称 → 静态，时间箭头 = 0
    time_arrow = 0.0
    if d1.available:
        grad_mag = d1.gradient
        # 计算梯度场的方向偏斜度
        # 简单代理：全局平均梯度方向和大小
        mean_grad_norm = np.mean(grad_mag)
        if mean_grad_norm > 0.01:
            # 有明显梯度 → 有演化方向
            time_arrow = 1.0  # 简化：默认未来方向
        else:
            time_arrow = 0.0  # 平坦场，无时间箭头

    # 2. 赝时间映射：沿梯度下降方向生成伪时间演化
    #    从高值区（源）到低值区（汇）的路径
    evolution = None
    is_fake_time = True  # 单帧输入默认赝时间

    if d1.available:
        # 简化版：沿梯度方向做 N 步迭代，模拟时间演化
        n_steps = 5
        h = 0.1  # 步长
        evolution = np.zeros((n_steps,) + I_norm.shape, dtype=np.float32)
        evolution[0] = I_norm.copy()

        current = I_norm.copy()
        for t in range(1, n_steps):
            # 沿梯度方向演化（简化版：扩散）
            laplacian = np.gradient(np.gradient(current, axis=0), axis=0) + \
                        np.gradient(np.gradient(current, axis=1), axis=1)
            current = current + h * laplacian
            evolution[t] = current

    # 3. 因果连接：从散度场判断源和汇
    #    正散度 = 源（因），负散度 = 汇（果）
    causal_link = None
    if d1.available:
        div = d1.divergence
        # 简化：直接用散度场作为因果连接强度
        causal_link = div.astype(np.float32)

    return D4Spacetime(
        evolution=evolution,
        causal_link=causal_link,
        is_fake_time=is_fake_time,
        available=True,
    )
