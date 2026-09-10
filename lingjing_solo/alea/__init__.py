"""ALEA 代数学习–进化内核（独立研究包）。

理论：docs/ALEA_理论白皮书_v1.md
数学：Strang《线性代数》第6版 Ch1–Ch10
"""
from .cov_pca import CovPCA
from .cr_rules import CRRule, CRRuleBank
from .gram_evolve import GramVolume
from .lu_identify import HypothesisEliminator, LinearHypothesis
from .mind import AleaMind, grid_embed
from .piecewise_dnn import ActionPiecewiseHead, PiecewiseDNN
from .verify_wm import VerifyWM

__all__ = [
    "AleaMind",
    "grid_embed",
    "CRRule",
    "CRRuleBank",
    "HypothesisEliminator",
    "LinearHypothesis",
    "CovPCA",
    "GramVolume",
    "PiecewiseDNN",
    "ActionPiecewiseHead",
    "VerifyWM",
]
