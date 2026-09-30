"""L6 CEAX Transfer：跨关/跨游戏可验证因果技能库。

Observation → Hypotheses → WorldModel → Counterfactuals
→ Planning → Verification → Rule Extraction → Transfer
"""
from .counterfactual import CounterfactualPlanner
from .families import FamilyTransfer, infer_game_family
from .hypotheses import CompetingHypotheses, HypothesisKind, StructuredHypothesis
from .layer import TransferLayer
from .ceax_controller import CeaxController
from .spectral_layer import SpectralCeaxController
from .neural_layer import NeuralCeaxController
from .alea_controller import AleaCeaxController
from .unified_controller import UnifiedCeaxController
from .rule_primitives import RuleComposer, RulePrimitive
from .rule_transfer import CROSS_GAME_FAMILIES, RuleTransfer
from .signatures import ObjectSig, StateSignature, build_signature
from .skill_library import SkillLibrary
from .world_model import CounterfactualWorldModel

__all__ = [
    "TransferLayer",
    "CeaxController",
    "SpectralCeaxController",
    "NeuralCeaxController",
    "AleaCeaxController",
    "UnifiedCeaxController",
    "SkillLibrary",
    "RuleTransfer",
    "RuleComposer",
    "RulePrimitive",
    "CounterfactualPlanner",
    "CounterfactualWorldModel",
    "CompetingHypotheses",
    "HypothesisKind",
    "StructuredHypothesis",
    "StateSignature",
    "ObjectSig",
    "build_signature",
    "FamilyTransfer",
    "infer_game_family",
    "CROSS_GAME_FAMILIES",
]
