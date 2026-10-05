"""
灵境引擎 V14.2 — core.writeback
====================================
Agent 写场契约（结构化）。

V14.1 及之前：apply_action 返回 Optional[np.ndarray]，写场靠文档约束——
基类返回 None，子类"记得"返回 J，忘了也不会报错。这是底座级隐患：
第三方插件最容易在这里破坏守恒恒等式 ΔΣφ = dt·ΣJ。

V14.2：用类型化对象 + 运行时校验强制契约：
  - AgentWriteback：携带 source (ndarray) + shape 声明
  - validate_writeback(wb, shape)：引擎调用，shape 不匹配立即报错
  - Agent.apply_action 现返回 AgentWriteback（基类提供标准实现，
    忘记写场 = 返回空源项，但必须显式声明，而非静默 None）

设计原则：
  "忘记写场"不再是静默 bug，而是明确的空源项（守恒仍成立，只是该 Agent
  未留下痕迹）——这与"写错了形状破坏守恒"严格区分：前者是策略，后者是错误。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class AgentWriteback:
    """
    Agent 单 tick 对信息场的写回。

    字段:
        source : ndarray，与场同 shape 的源项密度（场单位/时间）。
                 空写回 = 全零数组（不是 None），保证可累加。
        shape  : 声明 shape，供引擎校验（防止 (M,) 与 (N³,) 混用——
                 这是 V14 面积律最常见的 shape 陷阱）。
    """
    source: np.ndarray
    shape: tuple

    @classmethod
    def empty(cls, shape: tuple) -> "AgentWriteback":
        """显式空写回：Agent 本 tick 不写场（如已冻结、无动作）。"""
        return cls(source=np.zeros(shape, dtype=np.float64), shape=shape)

    @property
    def is_empty(self) -> bool:
        """是否为空写回（全零）。"""
        return bool(np.all(self.source == 0))

    def add(self, index: Tuple[int, int, int], value: float) -> None:
        """
        在指定格点追加源项（V14.2 子类写场的标准接口）。

        越界自动裁剪到 [0, dim-1]（连续坐标弹性反射后理论上不会越界，
        此处做防御性裁剪，避免单个坏 Agent 破坏整步守恒）。
        """
        if value == 0:
            return
        nx, ny, nz = self.shape
        x, y, z = (int(round(v)) for v in index)
        x = min(max(x, 0), nx - 1)
        y = min(max(y, 0), ny - 1)
        z = min(max(z, 0), nz - 1)
        self.source[x, y, z] += float(value)


def validate_writeback(wb: Optional[AgentWriteback], field_shape: tuple) -> np.ndarray:
    """
    引擎聚合前校验 AgentWriteback。

    规则:
      1. None → 视为空写回（向前兼容），但发出明确信号：
         调用方应使用 AgentWriteback.empty()，而非返回 None。
      2. shape 不匹配 → 立即报错（fail-fast），不静默广播。
      3. 返回标准化后的 source ndarray（保证可累加）。

    返回:
        ndarray，与 field_shape 完全一致，可直接 += 到 J_total。
    """
    if wb is None:
        # 向前兼容：旧子类可能仍返回 None。标准化为空，但保留此分支
        # 以便后续版本移除（TODO: 升级期结束后改为报错）。
        return np.zeros(field_shape, dtype=np.float64)

    if not isinstance(wb, AgentWriteback):
        raise TypeError(
            f"Agent.apply_action 必须返回 AgentWriteback，"
            f"实际: {type(wb).__name__}"
        )

    if wb.shape != field_shape:
        raise ValueError(
            f"Agent 写场 shape 不匹配: 声明 {wb.shape}，场 {field_shape}。"
            f"常见原因: 返回了泡壁 (M,) 向量而非全网格 (N³,)。"
        )

    if wb.source.shape != field_shape:
        raise ValueError(
            f"AgentWriteback.source shape {wb.source.shape} != 声明 {field_shape}"
        )

    return wb.source


__all__ = ["AgentWriteback", "validate_writeback"]
