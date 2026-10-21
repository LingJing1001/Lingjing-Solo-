from .encoder import PerceptionEncoder
from .ar25_encoder import Ar25Encoder
from .input_contract import InputContractError, R2PerceptionError, validate_input
from .observation import (
    ActionRecord,
    HypothesisContext,
    NormalizedObservation,
    normalize_observation,
    observation_from_mapping,
)
from .projection_graph import (
    DeltaResult,
    build_projection_graph,
    compute_delta,
    extract_candidate_nodes,
    hash_graph,
    propose_edges,
    refine_graph,
    track_objects,
)
from .dpl_adapter import GateReport, apply_geometry_gates, compute_dpl_features, projected_state_to_features
from .world_model import ProjectionWorldModel, object_changes, query_objects
from .projection_graph_types import (
    DPLFeatures,
    FieldSnapshot,
    GameObject,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
)

__all__ = [
    "ActionRecord",
    "Ar25Encoder",
    "DeltaResult",
    "DPLFeatures",
    "FieldSnapshot",
    "GateReport",
    "GameObject",
    "GraphEdge",
    "GraphNode",
    "GraphSnapshot",
    "HypothesisContext",
    "InputContractError",
    "NormalizedObservation",
    "PerceptionEncoder",
    "ProjectionWorldModel",
    "R2PerceptionError",
    "apply_geometry_gates",
    "build_projection_graph",
    "compute_delta",
    "compute_dpl_features",
    "extract_candidate_nodes",
    "hash_graph",
    "normalize_observation",
    "object_changes",
    "observation_from_mapping",
    "projected_state_to_features",
    "propose_edges",
    "query_objects",
    "refine_graph",
    "track_objects",
    "validate_input",
]
