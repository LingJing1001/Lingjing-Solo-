"""Deterministic, rule-based R2 projection-graph builder (stage 2).

The builder is intentionally numpy-only and remains independent from legacy R2
planning. It creates a small closed loop: validate -> delta -> components ->
edges -> tracking -> immutable graph snapshot. DPL values are accepted as
metadata/prior input but are not silently replaced with fake measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import hypot
from typing import Any, Mapping, Sequence, cast

import numpy as np

from .input_contract import validate_input
from .projection_graph_types import GraphEdge, GraphNode, GraphSnapshot, GameObject


EDGE_TYPES = ("spatial", "color", "motion", "geometric", "temporal")


@dataclass(frozen=True)
class DeltaResult:
    """Explicit frame-delta result; first-frame absence is never hidden."""

    magnitude: np.ndarray
    is_first_frame: bool
    reason: str

    @property
    def array(self) -> np.ndarray:
        return self.magnitude

    @property
    def shape(self) -> tuple[int, ...]:
        return self.magnitude.shape

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.magnitude, dtype=dtype)


def _as_float_frame(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    return array.astype(np.float32, copy=False).mean(axis=2)


def compute_delta(frame: Any, prev_frame: Any | None) -> DeltaResult:
    """Compute absolute grayscale delta with an explicit first-frame policy."""
    current = _as_float_frame(frame)
    if prev_frame is None:
        return DeltaResult(np.zeros_like(current, dtype=np.float32), True, "no_previous_frame")
    previous = _as_float_frame(prev_frame)
    if previous.shape != current.shape:
        raise ValueError("prev_frame shape must match frame shape")
    return DeltaResult(np.abs(current - previous), False, "absolute_grayscale_difference")


def _binary_mask(frame: Any, delta: DeltaResult | np.ndarray, threshold: float) -> np.ndarray:
    gray = _as_float_frame(frame)
    if isinstance(delta, DeltaResult):
        if delta.is_first_frame:
            return gray > threshold
        magnitude = delta.magnitude
    else:
        magnitude = np.asarray(delta)
    return magnitude > threshold


def _components(mask: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Return deterministic 4-connected component masks and exclusive bboxes."""
    if mask.ndim != 2:
        raise ValueError("component mask must be 2-D")
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if pixels:
                ys = [p[0] for p in pixels]
                xs = [p[1] for p in pixels]
                local = np.zeros((max(ys) - min(ys) + 1, max(xs) - min(xs) + 1), dtype=bool)
                for py, px in pixels:
                    local[py - min(ys), px - min(xs)] = True
                result.append((local, (min(xs), min(ys), max(xs) + 1, max(ys) + 1)))
    return result


def _histogram(frame: Any, bbox: tuple[int, int, int, int], bins: int = 8) -> tuple[float, ...]:
    array = np.asarray(frame)
    x0, y0, x1, y1 = bbox
    crop = array[y0:y1, x0:x1]
    if crop.ndim == 2:
        values = crop.reshape(-1)
    else:
        values = crop.reshape(-1, crop.shape[-1]).mean(axis=1)
    hist, _ = np.histogram(values, bins=bins, range=(0, 255))
    total = int(hist.sum())
    return tuple((hist / total).tolist()) if total else tuple(0.0 for _ in range(bins))


def extract_candidate_nodes(frame: Any, delta: DeltaResult, *, threshold: float = 1.0, min_area: int = 1, frame_index: int = 0) -> list[GameObject]:
    """Extract deterministic connected-component candidates from a frame/delta."""
    mask = _binary_mask(frame, delta, threshold)
    candidates: list[GameObject] = []
    for index, (local_mask, bbox) in enumerate(_components(mask)):
        if int(local_mask.sum()) < min_area:
            continue
        object_id = f"candidate-{index:04d}-{bbox[0]}-{bbox[1]}"
        candidates.append(GameObject(
            object_id=object_id,
            bbox=bbox,
            mask=tuple(tuple(bool(v) for v in row) for row in local_mask),
            rgb_histogram=_histogram(frame, bbox),
            first_frame=frame_index,
            last_frame=frame_index,
            state="visible",
        ))
    return candidates


def _center(obj: GameObject) -> tuple[float, float]:
    x0, y0, x1, y1 = obj.bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _iou(left: GameObject, right: GameObject) -> float:
    ax0, ay0, ax1, ay1 = left.bbox
    bx0, by0, bx1, by1 = right.bbox
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _color_similarity(left: GameObject, right: GameObject) -> float:
    size = max(len(left.rgb_histogram), len(right.rgb_histogram))
    if not size:
        return 0.0
    a = np.pad(np.asarray(left.rgb_histogram), (0, size - len(left.rgb_histogram)))
    b = np.pad(np.asarray(right.rgb_histogram), (0, size - len(right.rgb_histogram)))
    return float(1.0 - min(1.0, np.abs(a - b).sum() / 2.0))


def track_objects(candidates: Sequence[GameObject], prev_objects: Sequence[GameObject] | None, *, frame_index: int = 0, max_distance: float = 32.0) -> list[GameObject]:
    """Match candidates using IoU, color/geometry similarity, then adjacency proxy."""
    previous = list(prev_objects or ())
    available = set(range(len(previous)))
    tracked: list[GameObject] = []
    for candidate in candidates:
        scored: list[tuple[float, int]] = []
        cx, cy = _center(candidate)
        for index in sorted(available):
            old = previous[index]
            ox, oy = _center(old)
            distance = hypot(cx - ox, cy - oy)
            if distance > max_distance:
                continue
            # Staged matching: overlap first, then color and geometric proximity.
            score = 0.55 * _iou(candidate, old) + 0.30 * _color_similarity(candidate, old) + 0.15 * max(0.0, 1.0 - distance / max_distance)
            scored.append((score, index))
        if scored:
            _, index = max(scored, key=lambda item: (item[0], -item[1]))
            old = previous[index]
            available.remove(index)
            tracked.append(GameObject(**{**candidate.__dict__, "object_id": old.object_id, "first_frame": old.first_frame, "last_frame": frame_index, "state": "visible"}))
        else:
            tracked.append(GameObject(**{**candidate.__dict__, "last_frame": frame_index}))
    for index in sorted(available):
        old = previous[index]
        tracked.append(GameObject(**{**old.__dict__, "last_frame": frame_index, "state": "missing"}))
    return sorted(tracked, key=lambda obj: obj.object_id)


def propose_edges(objects: Sequence[GameObject], prev_objects: Sequence[GameObject] | None = None, *, max_distance: float = 64.0) -> list[GraphEdge]:
    """Propose the five typed edge families using deterministic rule weights."""
    edges: list[GraphEdge] = []
    items = sorted(objects, key=lambda obj: obj.object_id)
    for left_index, left in enumerate(items):
        for right in items[left_index + 1:]:
            distance = hypot(_center(left)[0] - _center(right)[0], _center(left)[1] - _center(right)[1])
            if distance <= max_distance:
                spatial = max(0.0, 1.0 - distance / max_distance)
                evidence = ("center_distance",)
                edges.append(GraphEdge(left.object_id, right.object_id, spatial, "spatial", evidence))
                edges.append(GraphEdge(left.object_id, right.object_id, _color_similarity(left, right), "color", ("histogram_l1",)))
                edges.append(GraphEdge(left.object_id, right.object_id, _iou(left, right), "geometric", ("bbox_iou",)))
                motion = max(0.0, 1.0 - hypot(*left.velocity) - hypot(*right.velocity))
                edges.append(GraphEdge(left.object_id, right.object_id, motion, "motion", ("velocity_norm",)))
    previous = {obj.object_id: obj for obj in (prev_objects or ())}
    for obj in items:
        if obj.object_id in previous:
            edges.append(GraphEdge(obj.object_id, obj.object_id, 1.0, "temporal", ("stable_object_id",)))
    return sorted(edges, key=lambda edge: (edge.src, edge.dst, edge.type))


def refine_graph(objects: Sequence[GameObject], edges: Sequence[GraphEdge], *, min_weight: float = 0.0) -> tuple[list[GameObject], list[GraphEdge]]:
    """Apply rule weights and retain evidence-bearing edges; no silent merge/split."""
    refined = [edge for edge in edges if edge.weight >= min_weight]
    return sorted(objects, key=lambda obj: obj.object_id), sorted(refined, key=lambda edge: (edge.src, edge.dst, edge.type))


def hash_graph(nodes: Sequence[GraphNode] | GraphSnapshot, edges: Sequence[GraphEdge] | None = None, *, level: int = 0, timestamp: float | None = None) -> str:
    """Return a deterministic hash over sorted graph content."""
    if isinstance(nodes, GraphSnapshot):
        payload = nodes.to_dict()
        payload["nodes"] = sorted(payload["nodes"], key=lambda item: (item["node_id"], item["object_id"]))
        payload["edges"] = sorted(payload["edges"], key=lambda item: (item["src"], item["dst"], item["type"]))
        payload.pop("global_hash", None)
    else:
        payload = {
            "schema_version": "r2-graph.v1",
            "nodes": [node.to_dict() for node in sorted(nodes, key=lambda item: (item.node_id, item.object_id))],
            "edges": [edge.to_dict() for edge in sorted(edges or (), key=lambda item: (item.src, item.dst, item.type))],
            "level": level,
            "timestamp": timestamp,
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def build_projection_graph(frame: Any, action: Any, prev_frame: Any | None = None, prev_objects: Sequence[GameObject] | None = None, dpl_prior: Any | None = None, constants: Mapping[str, Any] | None = None) -> tuple[list[GameObject], GraphSnapshot]:
    """Build one complete deterministic projection graph snapshot."""
    constants = dict(constants or {})
    min_size = cast(tuple[int, int], tuple(constants.get("min_size", (1, 1))))
    validated = validate_input(frame, action, min_size=min_size, max_pixels=int(constants.get("max_pixels", 4_000_000)))
    delta = compute_delta(validated, prev_frame)
    candidates = extract_candidate_nodes(validated, delta, threshold=float(constants.get("delta_threshold", 1.0)), min_area=int(constants.get("min_area", 1)), frame_index=int(constants.get("frame_index", 0)))
    objects = track_objects(candidates, prev_objects, frame_index=int(constants.get("frame_index", 0)), max_distance=float(constants.get("track_max_distance", 32.0)))
    edges = propose_edges(objects, prev_objects, max_distance=float(constants.get("edge_max_distance", 64.0)))
    objects, edges = refine_graph(objects, edges, min_weight=float(constants.get("edge_min_weight", 0.0)))
    nodes = [GraphNode(node_id=obj.object_id, object_id=obj.object_id, features=tuple(obj.rgb_histogram)) for obj in objects]
    snapshot = GraphSnapshot(nodes=tuple(nodes), edges=tuple(edges), level=int(constants.get("level", 0)), timestamp=constants.get("timestamp"))
    return objects, snapshot
