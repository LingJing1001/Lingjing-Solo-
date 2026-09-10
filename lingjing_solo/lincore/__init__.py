"""lincore · SSA 谱代数认知内核（Spectral–Subspace Architecture core）。

理论依据：docs/SSA_谱子空间认知架构_理论白皮书_v1.md
数学基础：Strang《线性代数（原书第6版）》Ch1–Ch10 → 认知组件映射（白皮书 §2/§6）。

模块：
  features  —— L0/L1 块特征 φ、像素活跃掩码（N(E) 支撑集）
  subspace  —— L2 四大基本子空间世界模型 + 三信号（惊讶 r / 可控 g / 进度 s）
  spectral  —— SVD 稳定岭回归、增量正规方程、周期检测、Krylov 前瞻
  eigen     —— L4 特征选项（Eigenoptions / empowerment 闭式代理）
  gradient  —— L5 SGD+动量进度预测器、对偶探索预算
  memory    —— L5 跨游戏低秩补全迁移（最小范数，防迁移幻觉）
  mind      —— L3/L4 编排器（三信号 → 决策融合）
"""
from .eigen import EigenSkillMap
from .features import activity_mask, block_center, block_of, encode_grid, grid_summary
from .gradient import DualBudgetScheduler, LogisticSGD
from .hypothesis_lab import Hypothesis, HypothesisLab, extract_suggestions_json
from .memory import MatrixMemory
from .mind import SpectralMind
from .segments import block_change_matrix, cochange_affinity, object_regions, spectral_segments
from .spectral import RidgeLinear, detect_period, krylov_horizon, ridge_solve, robust_rank
from .subspace import SubspaceModel
from .ttt import SharedBasis, TransitionReplay, combined_row_space, residual_novelty

__all__ = [
    "SubspaceModel",
    "EigenSkillMap",
    "RidgeLinear",
    "LogisticSGD",
    "DualBudgetScheduler",
    "MatrixMemory",
    "SpectralMind",
    "Hypothesis",
    "HypothesisLab",
    "extract_suggestions_json",
    "SharedBasis",
    "TransitionReplay",
    "combined_row_space",
    "residual_novelty",
    "block_change_matrix",
    "cochange_affinity",
    "spectral_segments",
    "object_regions",
    "ridge_solve",
    "detect_period",
    "krylov_horizon",
    "robust_rank",
    "encode_grid",
    "grid_summary",
    "block_of",
    "block_center",
    "activity_mask",
]
