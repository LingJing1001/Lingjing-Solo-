"""维度检测接口：detect_dimension（附录 B B.2）。"""
from __future__ import annotations

import numpy as np

from .types import DimensionInfo


def detect_dimension(env_state: np.ndarray, has_temporal: bool) -> DimensionInfo:
    """检测环境观测张量的维度信息。

    Args:
        env_state: 环境观测张量，任意维度
        has_temporal: 是否包含时间维度（如 4D 时空张量）

    Returns:
        DimensionInfo: spatial / temporal / effective_dim / is_height

    Raises:
        ValueError: env_state.ndim == 0 或 == 1
    """
    if env_state.ndim == 0:
        raise ValueError("env_state.ndim == 0: 标量输入无维度意义")
    if env_state.ndim == 1:
        raise ValueError("env_state.ndim == 1: 一维序列无空间流形")

    ndim = env_state.ndim

    # 规则表（附录 B B.2.4）
    if ndim == 2:
        # (H, W) → 2D 网格
        return DimensionInfo(
            spatial=2.0, temporal=False, effective_dim=2, is_height=False
        )
    elif ndim == 3:
        if env_state.shape[-1] == 1:
            # (H, W, 1) → 2.5D 高度场
            return DimensionInfo(
                spatial=2.5, temporal=False, effective_dim=3, is_height=True
            )
        else:
            # (H, W, C) C>1 → 3D 体素
            return DimensionInfo(
                spatial=3.0, temporal=False, effective_dim=3, is_height=False
            )
    elif ndim == 4:
        if has_temporal:
            if env_state.shape[-2] == 1:
                # (H, W, 1, T) → 2.5D + 时间
                return DimensionInfo(
                    spatial=2.5, temporal=True, effective_dim=4, is_height=True
                )
            else:
                # (H, W, D, T) → 4D 时空
                return DimensionInfo(
                    spatial=3.0, temporal=True, effective_dim=4, is_height=False
                )
        else:
            # (H, W, D, ?) 无时序标记 → 3D
            return DimensionInfo(
                spatial=3.0, temporal=False, effective_dim=3, is_height=False
            )
    else:
        # ndim > 4：按最高空间维处理
        return DimensionInfo(
            spatial=float(ndim - 1),
            temporal=has_temporal,
            effective_dim=ndim - 1 + int(has_temporal),
            is_height=False,
        )
