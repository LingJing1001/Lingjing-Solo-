"""DPL-to-projection-graph adapter and P1 geometry gates.

This module exposes only scalar summaries of the existing DPL implementation;
raw arrays remain available through ``ProjectedState`` for callers that need
them. Gate reports are conservative and explicitly label unavailable/fake data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..dpl.projection import project
from ..dpl.types import ProjectedState, UniversalConstants
from .projection_graph_types import DPLFeatures, GraphEdge, GraphSnapshot, GameObject


@dataclass(frozen=True)
class GateReport:
    passed: bool
    violations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    objects: tuple[GameObject, ...] = ()
    graph: GraphSnapshot | None = None
    dpl: DPLFeatures = DPLFeatures()


def _finite_mean(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(np.abs(finite))) if finite.size else None


def _finite_max(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.max(np.abs(finite))) if finite.size else None


def projected_state_to_features(state: ProjectedState) -> DPLFeatures:
    """Adapt existing D1-D4 results without turning unavailable into zero."""
    d1, d2, d3, d4 = state.D1, state.D2, state.D3, state.D4
    values = {
        "gradient_mag": _finite_mean(d1.gradient) if d1 and d1.available else None,
        "divergence": _finite_mean(d1.divergence) if d1 and d1.available else None,
        "curvature": _finite_mean(d2.curvature) if d2 and d2.available else None,
        "vorticity": _finite_mean(d2.vorticity) if d2 and d2.available else None,
        "density": _finite_mean(d3.density) if d3 and d3.available else None,
        "euler_char": float(d3.euler_char) if d3 and d3.available else None,
        "time_arrow": float(1.0 if d4 and d4.available and d4.causal_link is not None else 0.0) if d4 and d4.available else None,
        "evolution_rate": _finite_mean(np.diff(d4.evolution, axis=0)) if d4 and d4.available and d4.evolution is not None else None,
    }
    validity = tuple(sorted((name, value is not None) for name, value in values.items()))
    return DPLFeatures(
        **values,
        available=any(value is not None for value in values.values()),
        validity=validity,
        is_fake_time=bool(d4.is_fake_time) if d4 else False,
    )


def compute_dpl_features(frame: Any, constants: UniversalConstants | None = None, *, has_temporal: bool = False) -> tuple[ProjectedState, DPLFeatures]:
    """Run the existing DPL projection and return its typed graph adapter."""
    dpl_constants = constants or UniversalConstants()
    array = np.asarray(frame)
    if array.ndim == 3:
        array = array.mean(axis=2)
    state = project(array.astype(np.float32, copy=False), has_temporal, dpl_constants)
    return state, projected_state_to_features(state)


def _connected_components(graph: GraphSnapshot) -> int:
    nodes = {node.node_id for node in graph.nodes}
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for edge in graph.edges:
        if edge.src in adjacency and edge.dst in adjacency:
            adjacency[edge.src].add(edge.dst)
            adjacency[edge.dst].add(edge.src)
    count = 0
    while adjacency:
        count += 1
        pending = [next(iter(adjacency))]
        visited = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            pending.extend(adjacency.get(node, set()) - visited)
        for node in visited:
            adjacency.pop(node, None)
    return count


def apply_geometry_gates(
    graph: GraphSnapshot,
    objects: Sequence[GameObject],
    dpl: DPLFeatures,
    constants: UniversalConstants | None = None,
) -> GateReport:
    """Apply the minimum P1 gates and return an auditable adjusted snapshot."""
    limits = constants or UniversalConstants()
    violations: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, Any] = {}

    # Information-conservation proxy: finite degree/weight accounting only.
    total_weight = float(sum(edge.weight for edge in graph.edges))
    degrees = {node.node_id: 0 for node in graph.nodes}
    for edge in graph.edges:
        if edge.src in degrees:
            degrees[edge.src] += 1
        if edge.dst in degrees:
            degrees[edge.dst] += 1
    finite_proxy = bool(np.isfinite(total_weight) and all(value >= 0 for value in degrees.values()))
    evidence["information_conservation_proxy"] = {"edge_weight_total": total_weight, "degrees": degrees, "finite": finite_proxy}
    if not finite_proxy:
        violations.append("information_conservation_proxy")

    curvature = dpl.curvature if dpl.curvature is not None else None
    if curvature is None:
        warnings.append("curvature_unavailable")
    elif abs(curvature) > limits.K_max:
        violations.append("curvature_bound")
        evidence["curvature"] = {"value": curvature, "K_max": limits.K_max}

    vorticity = dpl.vorticity if dpl.vorticity is not None else None
    if vorticity is None:
        warnings.append("vorticity_unavailable")
    elif abs(vorticity) > limits.omega_max:
        warnings.append("vorticity_anomaly")
        evidence["vorticity"] = {"value": vorticity, "omega_max": limits.omega_max}

    expected_euler = int(round(dpl.euler_char)) if dpl.euler_char is not None else None
    actual_euler = len(graph.nodes) - _connected_components(graph) + sum(1 for edge in graph.edges if edge.src == edge.dst)
    evidence["topology"] = {"expected_euler": expected_euler, "graph_euler_proxy": actual_euler}
    if expected_euler is None:
        warnings.append("topology_unavailable")
    elif abs(expected_euler - actual_euler) > 1:
        violations.append("topology_legality")

    unknown_types = sorted({edge.type for edge in graph.edges} - {"spatial", "color", "motion", "geometric", "temporal"})
    if unknown_types:
        violations.append("cross_dimension_isolation")
        evidence["unknown_edge_types"] = unknown_types

    validity = dict(dpl.validity)
    unavailable = sorted(name for name in ("gradient_mag", "divergence", "curvature", "vorticity", "density", "euler_char", "time_arrow", "evolution_rate") if not validity.get(name, False))
    if unavailable:
        warnings.append("dpl_unavailable:" + ",".join(unavailable))
    if dpl.is_fake_time:
        warnings.append("d4_fake_time")
        evidence["d4"] = {"is_fake_time": True, "time_arrow": dpl.time_arrow}

    invalid_ids = {obj.object_id for obj in objects if curvature is not None and abs(curvature) > limits.K_max}
    adjusted_objects = tuple(replace(obj, state="invalid_geometry") if obj.object_id in invalid_ids else obj for obj in objects)
    adjusted_edges = tuple(
        replace(edge, weight=edge.weight * 0.25, evidence=edge.evidence + ("curvature_bound",))
        if edge.src in invalid_ids or edge.dst in invalid_ids else edge
        for edge in graph.edges
    )
    adjusted_graph = replace(graph, edges=adjusted_edges, global_hash="")
    return GateReport(not violations, tuple(violations), tuple(warnings), evidence, adjusted_objects, adjusted_graph, dpl)


def build_gated_projection_graph(frame: Any, action: Any, *, graph_builder: Any, constants: Mapping[str, Any] | None = None, dpl_constants: UniversalConstants | None = None, **kwargs: Any) -> tuple[list[GameObject], GraphSnapshot, GateReport]:
    """Convenience adapter used by the stage 4 world model."""
    objects, graph = graph_builder(frame, action, constants=constants, **kwargs)
    _, dpl = compute_dpl_features(frame, dpl_constants)
    report = apply_geometry_gates(graph, objects, dpl, dpl_constants)
    return list(report.objects), report.graph or graph, report
