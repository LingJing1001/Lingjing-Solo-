"""核心数据结构定义：帧、对象、规则假设、转移、Φ 场快照。

所有跨层数据结构在此统一，避免层间耦合。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


@dataclass
class Frame:
    """单帧：64x64 整数网格，值域 [0, 15] 表示 16 色。"""
    grid: np.ndarray            # shape (H, W)
    t: int = 0                  # 帧序号 / 时间步


@dataclass
class GameObject:
    """连通域分割出的对象：颜色 + 像素坐标列表 + 包围盒。"""
    color: int
    pixels: List[Tuple[int, int]] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x0, y0, x1, y1)


@dataclass
class RuleHypothesis:
    """一条规则假设：带置信度的 (前提 -> 结论)。"""
    premise: str                # 自然语言 / DSL 描述的条件
    conclusion: str             # 预期结果
    confidence: float = 0.5
    evidence: int = 0           # 支持证据计数


@dataclass
class Transition:
    """状态转移记录：(state_hash, action, next_state_hash, delta_summary)。"""
    state_before: str
    action: str
    state_after: str
    delta_pixels: int           # 变化像素数
    t: int = 0


@dataclass
class GoalHypothesis:
    """对"什么算赢"的假设，带置信度。"""
    description: str
    confidence: float = 0.3


@dataclass
class ReflectionSignal:
    """触发反思的三类信号。"""
    loop_trapped: bool = False
    rule_conflict: bool = False
    budget_warning: bool = False

    @property
    def should_reflect(self) -> bool:
        return self.loop_trapped or self.rule_conflict or self.budget_warning


@dataclass
class FieldSnapshot:
    """Φ 场的压缩摘要，供 LLM 上下文打包使用。"""
    grid_summary: str
    rules: List[RuleHypothesis]
    goals: List[GoalHypothesis]
    recent_transitions: List[Transition]
    visited_count: int
    step: int
