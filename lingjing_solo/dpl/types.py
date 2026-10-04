"""DPL 数据结构定义（附录 B/C/E/F 对齐）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DimensionInfo:
    spatial: float = 2.0
    temporal: bool = False
    effective_dim: int = 2
    is_height: bool = False


@dataclass
class D1FlowLine:
    gradient: np.ndarray = None
    divergence: np.ndarray = None
    direction: np.ndarray = None
    gradient_vector: np.ndarray = None
    available: bool = True


@dataclass
class D2Surface:
    hessian: Optional[np.ndarray] = None
    vorticity: np.ndarray = None
    curvature: np.ndarray = None
    vorticity_is_vector: bool = False
    available: bool = True


@dataclass
class D3Volume:
    density: np.ndarray = None
    euler_char: int = 0
    ricci_scalar: Optional[np.ndarray] = None
    available: bool = False


@dataclass
class D4Spacetime:
    evolution: np.ndarray = None
    causal_link: np.ndarray = None
    is_fake_time: bool = False
    available: bool = False


@dataclass
class ProjectedState:
    D1: D1FlowLine = None
    D2: D2Surface = None
    D3: D3Volume = None
    D4: D4Spacetime = None
    dim_info: DimensionInfo = None
    input_shape: tuple = ()
    env_version: int = 0
    tick: int = 0


@dataclass
class UniversalConstants:
    c_light: float = 1.0
    h_planck: float = 1e-3
    delta_s_min: float = 1.0
    time_arrow: float = 0.0
    laws_set: List[str] = field(default_factory=list)
    conservation_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    K_max: float = 1.0
    omega_max: float = 1.0
    epsilon_gate: float = 1e-3
    delta_overlap: float = 0.05
    D: float = 1.0
    I_0: float = 1.0
    l_0: float = 1.0
    clear_region_percentile: float = 80.0
    chi_ref: float = 2.0
    gravity_threshold_ratio: float = 0.1
    periodicity_threshold: float = 0.3
    provenance: Dict[str, str] = field(default_factory=dict)


@dataclass
class Modification:
    type: str = "add_source"
    target: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    info_before: float = 0.0
    info_after: float = 0.0


@dataclass
class GateResult:
    passed: bool = True
    violations: List[str] = field(default_factory=list)
    score: float = 1.0
    action: str = "pass"
    threshold: Optional[Dict[str, float]] = None
    observed: Optional[Dict[str, float]] = None
    evidence: Optional[Dict[str, Any]] = None


@dataclass
class IntuitionRule:
    rule_type: str = ""
    confidence: float = 0.0
    triggered_features: List[str] = field(default_factory=list)


@dataclass
class IntuitionResult:
    rules: List[IntuitionRule] = field(default_factory=list)
    confidence: float = 0.0
    intuition_type: str = "no_intuition"


@dataclass
class GeometricInvariant:
    pattern_type: str = ""
    dim_layer: str = ""
    stability_score: float = 0.0
    evidence_count: int = 0
    source_family: str = "unknown"
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class GateCounterexample:
    gate_type: str = ""
    violation: float = 0.0
    raw_dim: float = 0.0
    effective_dim: int = 0
    sample_hash: str = ""
    timestamp: int = 0


@dataclass
class IntuitionHypothesis:
    rule_type: str = ""
    confidence: float = 0.0
    triggered_features: List[str] = field(default_factory=list)
    holdout_delta: float = 0.0
