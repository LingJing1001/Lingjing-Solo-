"""灵境引擎 V14.0 — core 包（L0-L2 物理底座）"""
from .field import Field, FieldConfig
from .laplacian import (
    laplacian_7point,
    build_bubble_laplacian_csr,
    bubble_mask_spherical,
)
from .scenario import Scenario

__all__ = [
    "Field",
    "FieldConfig",
    "laplacian_7point",
    "build_bubble_laplacian_csr",
    "bubble_mask_spherical",
    "Scenario",
]
