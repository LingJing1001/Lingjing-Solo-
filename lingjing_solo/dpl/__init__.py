"""DPL 维度投影层（Dimension Projection Layer）。

五接口：detect_dimension / calibrate / project / check / infer
对齐附录 B-H 规范，纯函数无副作用。
"""
from .types import (
    DimensionInfo, D1FlowLine, D2Surface, D3Volume, D4Spacetime,
    ProjectedState, UniversalConstants, Modification, GateResult,
    IntuitionRule, IntuitionResult,
    GeometricInvariant, GateCounterexample, IntuitionHypothesis,
)

__all__ = [
    "DimensionInfo", "D1FlowLine", "D2Surface", "D3Volume", "D4Spacetime",
    "ProjectedState", "UniversalConstants", "Modification", "GateResult",
    "IntuitionRule", "IntuitionResult",
    "GeometricInvariant", "GateCounterexample", "IntuitionHypothesis",
]
