"""Versioned R2 projection-graph contracts for the stage 0/1 rollout.

This module is deliberately independent from the legacy ``core.types`` objects:
legacy R2 remains the rollback target while this schema is introduced behind a
feature flag.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, cast


R2_GRAPH_SCHEMA_VERSION = "r2-graph.v1"
R2_FIELD_SCHEMA_VERSION = "r2-field.v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class DPLFeatures:
    gradient_mag: float | None = None
    divergence: float | None = None
    curvature: float | None = None
    vorticity: float | None = None
    density: float | None = None
    euler_char: float | None = None
    time_arrow: float | None = None
    evolution_rate: float | None = None
    available: bool = False


@dataclass(frozen=True)
class GameObject:
    object_id: str
    bbox: tuple[int, int, int, int]
    mask: tuple[tuple[bool, ...], ...] = ()
    rgb_histogram: tuple[float, ...] = ()
    dpl: DPLFeatures = DPLFeatures()
    graph_embedding: tuple[float, ...] = ()
    first_frame: int = 0
    last_frame: int = 0
    state: str = "visible"
    velocity: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if len(self.bbox) != 4 or self.bbox[0] > self.bbox[2] or self.bbox[1] > self.bbox[3]:
            raise ValueError("bbox must be (min_x, min_y, max_x, max_y)")
        if self.first_frame > self.last_frame:
            raise ValueError("first_frame must not exceed last_frame")
        if any(value < 0 or value > 1 for value in self.rgb_histogram):
            raise ValueError("rgb_histogram values must be normalized to [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "bbox": list(self.bbox),
            "mask": [list(row) for row in self.mask],
            "rgb_histogram": list(self.rgb_histogram),
            "dpl": self.dpl.__dict__,
            "graph_embedding": list(self.graph_embedding),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "state": self.state,
            "velocity": list(self.velocity),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GameObject":
        return cls(
            object_id=str(value["object_id"]),
            bbox=cast(tuple[int, int, int, int], tuple(int(x) for x in value["bbox"])),
            mask=tuple(tuple(bool(x) for x in row) for row in value.get("mask", ())),
            rgb_histogram=tuple(float(x) for x in value.get("rgb_histogram", ())),
            dpl=DPLFeatures(**dict(value.get("dpl", {}))),
            graph_embedding=tuple(float(x) for x in value.get("graph_embedding", ())),
            first_frame=int(value.get("first_frame", 0)),
            last_frame=int(value.get("last_frame", 0)),
            state=str(value.get("state", "visible")),
            velocity=cast(tuple[float, float], tuple(float(x) for x in value.get("velocity", (0.0, 0.0)))),
        )


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    object_id: str
    features: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "object_id": self.object_id, "features": list(self.features)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphNode":
        return cls(str(value["node_id"]), str(value["object_id"]), tuple(float(x) for x in value.get("features", ())))


@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    weight: float
    type: str
    evidence: tuple[str, ...] = ()

    @property
    def edge_type(self) -> str:
        """Compatibility alias for early local callers."""
        return self.type

    def __post_init__(self) -> None:
        if not self.src or not self.dst or not self.type:
            raise ValueError("graph edge endpoints and type must be non-empty")
        if self.weight < 0:
            raise ValueError("graph edge weight must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "weight": self.weight, "type": self.type, "evidence": list(self.evidence)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphEdge":
        return cls(str(value["src"]), str(value["dst"]), float(value["weight"]), str(value.get("type", value.get("edge_type", ""))), tuple(str(x) for x in value.get("evidence", ())))


@dataclass(frozen=True)
class GraphSnapshot:
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    level: int = 0
    timestamp: float | None = None
    schema_version: str = R2_GRAPH_SCHEMA_VERSION
    global_hash: str = ""

    def __post_init__(self) -> None:
        if self.level < 0 or (self.timestamp is not None and self.timestamp < 0):
            raise ValueError("level and timestamp must be non-negative")
        payload = {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "level": self.level,
            "timestamp": self.timestamp,
        }
        expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if self.global_hash and self.global_hash != expected:
            raise ValueError("global_hash does not match canonical graph content")
        object.__setattr__(self, "global_hash", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "level": self.level,
            "timestamp": self.timestamp,
            "global_hash": self.global_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphSnapshot":
        return cls(
            nodes=tuple(GraphNode.from_dict(x) for x in value.get("nodes", ())),
            edges=tuple(GraphEdge.from_dict(x) for x in value.get("edges", ())),
            level=int(value.get("level", 0)),
            timestamp=None if value.get("timestamp") is None else float(value["timestamp"]),
            schema_version=str(value.get("schema_version", R2_GRAPH_SCHEMA_VERSION)),
            global_hash=str(value.get("global_hash", "")),
        )


@dataclass(frozen=True)
class FieldSnapshot:
    schema_version: str = R2_FIELD_SCHEMA_VERSION
    frame_index: int = 0
    field: Mapping[str, Any] | None = None
    phi: Mapping[str, Any] | None = None
    time: float | None = None
    source_backend: str = "unknown"
    visibility: str = "full"
    objects: tuple[GameObject, ...] = ()
    graph: GraphSnapshot = GraphSnapshot()
    object_changes: tuple[str, ...] = ()
    phi_summary: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.frame_index < 0 or (self.time is not None and self.time < 0):
            raise ValueError("frame_index and time must be non-negative")
        if self.visibility not in {"full", "partial", "unknown"}:
            raise ValueError("visibility must be full, partial, or unknown")
        if isinstance(self.phi_summary, Mapping):
            normalized_phi = tuple(sorted((str(k), float(v)) for k, v in self.phi_summary.items()))
            object.__setattr__(self, "phi_summary", normalized_phi)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_index": self.frame_index,
            "field": self.field,
            "phi": self.phi,
            "time": self.time,
            "source_backend": self.source_backend,
            "visibility": self.visibility,
            "objects": [obj.to_dict() for obj in self.objects],
            "graph": self.graph.to_dict(),
            "object_changes": list(self.object_changes),
            "phi_summary": dict(self.phi_summary),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FieldSnapshot":
        phi = value.get("phi_summary", {})
        if isinstance(phi, Mapping):
            phi_items = tuple(sorted((str(k), float(v)) for k, v in phi.items()))
        else:
            phi_items = tuple((str(k), float(v)) for k, v in phi)
        return cls(
            schema_version=str(value.get("schema_version", R2_FIELD_SCHEMA_VERSION)),
            frame_index=int(value.get("frame_index", 0)),
            field=value.get("field"),
            phi=value.get("phi"),
            time=None if value.get("time") is None else float(value["time"]),
            source_backend=str(value.get("source_backend", "unknown")),
            visibility=str(value.get("visibility", "full")),
            objects=tuple(GameObject.from_dict(x) for x in value.get("objects", ())),
            graph=GraphSnapshot.from_dict(value.get("graph", {})),
            object_changes=tuple(str(x) for x in value.get("object_changes", ())),
            phi_summary=phi_items,
        )
