"""核心数据结构：帧、Φ、泡壁、源项、感知快照、规则与场摘要。

钱学森灵境转义：
  Φ          → 信息密度粗粒场（非体素物理仿真）
  泡壁       → 变化区 ROI（清晰区）
  ΔJ / 源项  → Agent 动作对场的写入意图
  版本号     → 每 Tick 单调递增的事实源版本（协议 6 Solo 化）
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Frame:
    """单帧：64x64 整数网格 + 环境元数据。"""
    grid: np.ndarray
    t: int = 0
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: List[str] = field(default_factory=list)
    raw: Any = None


@dataclass
class GameObject:
    """连通域对象（泡壁内高精度分割结果）。"""
    color: int
    pixels: List[Tuple[int, int]] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class BubbleWall:
    """泡壁 = 清晰区边界（面积律：只对墙内做高精度）。

    clear_rects: 清晰区矩形列表 (y0,x0,y1,x1)
    fog_ratio:   迷雾区占比 ∈ [0,1] —— 越大越该探索/展开
    """
    clear_rects: List[Tuple[int, int, int, int]] = field(default_factory=list)
    fog_ratio: float = 1.0
    clear_pixels: int = 0
    wall_entropy: float = 0.0       # 清晰区信息熵（代理）


@dataclass
class PhiDensity:
    """信息密度 Φ 的粗粒表示（协议 2 Lite）。

    不用 N³ 节点场方程；用分块均值 + 梯度能量刻画「哪里信息集中」。
    """
    blocks: np.ndarray              # shape (B, B) 块均值密度
    mean: float = 0.0
    grad_energy: float = 0.0        # ‖∇Φ‖² 总能量 → 「弯曲」代理
    peak_block: Tuple[int, int] = (0, 0)  # 密度峰所在块


@dataclass
class SourceTerm:
    """协议 5：Agent 动作写回场的源项 ΔJ。"""
    agent_id: str = "solo"
    action: str = ""
    delta_phi: float = 0.0          # 预期信息扰动强度
    tick: int = 0
    version_read: int = 0           # 决策时所读场版本（协议 6）
    rationale: str = ""             # explore|plan|reflect|reset


@dataclass
class PerceptionSnapshot:
    """协议 4：从 Φ 求解出的可感知世界摘要。"""
    version: int = 0
    tick: int = 0
    feature: Optional[np.ndarray] = None
    objects: List[GameObject] = field(default_factory=list)
    delta_pixels: List[Tuple[int, int]] = field(default_factory=list)
    bubble: Optional[BubbleWall] = None
    phi: Optional[PhiDensity] = None
    curvature_proxy: float = 0.0    # = phi.grad_energy 规范化
    levels: int = 0
    env_state: str = "NOT_FINISHED"


@dataclass
class RuleHypothesis:
    premise: str
    conclusion: str
    confidence: float = 0.5
    evidence: int = 0


@dataclass
class Transition:
    state_before: str
    action: str
    state_after: str
    delta_pixels: int
    t: int = 0
    progressed: bool = False
    version: int = 0                # 写入时的场版本


@dataclass
class GoalHypothesis:
    description: str
    confidence: float = 0.3
    state_hash: str = ""
    kind: str = "generic"


@dataclass
class ReflectionSignal:
    """L4 自指监视信号（含 C 值相变）。"""
    loop_trapped: bool = False
    rule_conflict: bool = False
    budget_warning: bool = False
    hypotheses_exhausted: bool = False
    search_budget_exhausted: bool = False
    c_value: float = 0.0            # 自指复杂度代理 ∈ [0,1]
    c_phase: bool = False           # C 超过阈值 → 该请战略层

    @property
    def should_reflect(self) -> bool:
        return (
            self.loop_trapped
            or self.rule_conflict
            or self.budget_warning
            or self.hypotheses_exhausted
            or self.search_budget_exhausted
            or self.c_phase
        )


@dataclass
class FieldSnapshot:
    """Φ 场压缩摘要（供 LLM / 审计）。"""
    grid_summary: str
    rules: List[RuleHypothesis]
    goals: List[GoalHypothesis]
    recent_transitions: List[Transition]
    visited_count: int
    step: int
    graph_edges: int = 0
    best_goal: str = ""
    version: int = 0
    c_value: float = 0.0
    fog_ratio: float = 0.0
    phi_grad_energy: float = 0.0
    manifesto_tag: str = "Lingjing EtherealRealm-Solo"


# ---------- AR25 (kaleidoscope mirror) structured observation ----------


@dataclass
class Ar25Piece:
    """Movable / fixed polyomino for AR25."""
    id: str
    x: int
    y: int
    pixels: List[List[int]]
    tags: Tuple[str, ...] = ()
    fixed: bool = False

    @property
    def pixels_hash(self) -> str:
        # Stable across processes (unlike built-in hash()).
        return repr(tuple(tuple(row) for row in self.pixels))


@dataclass
class Ar25Axis:
    """Reflection axis: kind V (vertical line) or H (horizontal line)."""
    id: str
    kind: str  # "V" | "H"
    x: int
    y: int
    tags: Tuple[str, ...] = ()
    fixed: bool = False


@dataclass
class Ar25Obs:
    """Layer-0 structured observation for one AR25 level."""
    grid_w: int
    grid_h: int
    targets: List[Tuple[int, int]]
    pieces: List[Ar25Piece]
    axes: List[Ar25Axis]
    selected_id: Optional[str] = None
    steps_left: int = 64
    rotate_dist: Dict[str, int] = field(default_factory=dict)
    level_index: Optional[int] = None


@dataclass
class Ar25Config:
    """Target poses in configuration space (not action sequences)."""
    piece_xy: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    axis_coord: Dict[str, int] = field(default_factory=dict)

    def state_key(self) -> str:
        pieces = tuple(sorted((k, v[0], v[1]) for k, v in self.piece_xy.items()))
        axes = tuple(sorted(self.axis_coord.items()))
        return f"p={pieces}|a={axes}"


@dataclass
class Ar25CoverReport:
    covered: set
    uncovered: List[Tuple[int, int]]
    ok: bool
