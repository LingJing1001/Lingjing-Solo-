from .encoder import PerceptionEncoder
from .ar25_encoder import Ar25Encoder
from .observation import (
    ActionRecord,
    HypothesisContext,
    NormalizedObservation,
    normalize_observation,
    observation_from_mapping,
)
from .segmentation import (
    ARC_COLOR_CHARS,
    segment_frame,
    segment_layer,
)
from .stability import (
    ANIM_RATIO,
    WARMUP,
    LevelStability,
    interior_point,
    object_hud_score,
    rerank_click_candidates,
)
# Generic cell-lattice recovery lives in ``core`` (pure numpy, no heavy imports)
# so the closed-book solvers can use it without pulling this package in; it is
# re-exported here because "where is the board grid?" is a perception question.
from ..core.lattice import (
    LatticeSpec,
    detect_cell_lattice,
    infer_background,
    lattice_sweep,
    to_cell_grid,
)

__all__ = [
    "ANIM_RATIO",
    "ARC_COLOR_CHARS",
    "ActionRecord",
    "Ar25Encoder",
    "HypothesisContext",
    "LatticeSpec",
    "LevelStability",
    "NormalizedObservation",
    "PerceptionEncoder",
    "WARMUP",
    "detect_cell_lattice",
    "infer_background",
    "interior_point",
    "lattice_sweep",
    "normalize_observation",
    "object_hud_score",
    "observation_from_mapping",
    "rerank_click_candidates",
    "segment_frame",
    "segment_layer",
    "to_cell_grid",
]
