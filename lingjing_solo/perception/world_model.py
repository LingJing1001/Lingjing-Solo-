"""Φ/world-model adapter for frame-level projection graph snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.phi import compute_bubble, compute_phi
from ..dpl.types import UniversalConstants
from .dpl_adapter import GateReport, build_gated_projection_graph
from .projection_graph import build_projection_graph
from .projection_graph_types import FieldSnapshot, GameObject, GraphSnapshot


def _gray(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    return array.mean(axis=2) if array.ndim == 3 else array


def _overlap(left: GameObject, right: GameObject) -> float:
    ax0, ay0, ax1, ay1 = left.bbox
    bx0, by0, bx1, by1 = right.bbox
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    old_area = max(1, ax1 - ax0) * max(1, ay1 - ay0)
    return area / old_area


def object_changes(previous: Sequence[GameObject], current: Sequence[GameObject]) -> tuple[str, ...]:
    """Describe appearance, disappearance, movement, occlusion, merge and split."""
    old = {obj.object_id: obj for obj in previous}
    new = {obj.object_id: obj for obj in current}
    changes: list[str] = []
    for object_id in sorted(set(new) - set(old)):
        changes.append(f"appeared:{object_id}")
    for object_id in sorted(set(old) - set(new)):
        changes.append(f"disappeared:{object_id}")
    for object_id in sorted(set(old) & set(new)):
        if new[object_id].state == "missing":
            changes.append(f"occluded:{object_id}")
        elif old[object_id].bbox != new[object_id].bbox:
            changes.append(f"moved:{object_id}")

    old_values, new_values = list(old.values()), list(new.values())
    for current_obj in new_values:
        matches = [previous_obj for previous_obj in old_values if _overlap(previous_obj, current_obj) >= 0.25]
        if len(matches) > 1:
            changes.append("merge:" + current_obj.object_id)
    for previous_obj in old_values:
        matches = [current_obj for current_obj in new_values if _overlap(previous_obj, current_obj) >= 0.25]
        if len(matches) > 1:
            changes.append("split:" + previous_obj.object_id)
    return tuple(sorted(set(changes)))


def query_objects(snapshot: FieldSnapshot, bbox: tuple[int, int, int, int] | None = None, state: str | None = None, object_id: str | None = None) -> list[GameObject]:
    """Stable R3/R4/R5 query API over one immutable field snapshot."""
    def in_bbox(obj: GameObject) -> bool:
        if bbox is None:
            return True
        ax0, ay0, ax1, ay1 = obj.bbox
        bx0, by0, bx1, by1 = bbox
        return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)

    return [
        obj for obj in snapshot.objects
        if (object_id is None or obj.object_id == object_id)
        and (state is None or obj.state == state)
        and in_bbox(obj)
    ]


@dataclass
class ProjectionWorldModel:
    """Mutable holder whose outputs are immutable, versioned FieldSnapshots."""

    snapshot: FieldSnapshot | None = None
    previous_frame: np.ndarray | None = None
    tick: int = 0

    def update(
        self,
        frame: Any,
        action: Any,
        *,
        constants: Mapping[str, Any] | None = None,
        dpl_constants: UniversalConstants | None = None,
        source_backend: str = "projection_graph",
        visibility: str = "full",
    ) -> tuple[FieldSnapshot, GateReport]:
        config = dict(constants or {})
        config.setdefault("frame_index", self.tick)
        previous_objects = self.snapshot.objects if self.snapshot else ()
        current_frame = np.asarray(frame).copy()
        objects, graph, report = build_gated_projection_graph(
            frame,
            action,
            graph_builder=build_projection_graph,
            constants=config,
            dpl_constants=dpl_constants,
            prev_frame=self.previous_frame,
            prev_objects=previous_objects,
        )
        gray = _gray(current_frame)
        phi = compute_phi(gray)
        bubble = compute_bubble(None if self.previous_frame is None else _gray(self.previous_frame), gray)
        changes = object_changes(previous_objects, objects)
        field = {"mean": phi.mean, "grad_energy": phi.grad_energy, "peak_block": list(phi.peak_block)}
        phi_summary = {"mean": phi.mean, "grad_energy": phi.grad_energy, "curvature_proxy": min(1.0, phi.grad_energy / 8.0)}
        snapshot = FieldSnapshot(
            frame_index=self.tick,
            field=field,
            phi=phi_summary,
            time=float(config.get("timestamp", self.tick)),
            source_backend=source_backend,
            visibility=visibility,
            objects=tuple(objects),
            graph=graph,
            object_changes=changes,
            phi_summary=tuple(sorted(phi_summary.items())),
        )
        self.snapshot = snapshot
        self.previous_frame = current_frame
        self.tick += 1
        return snapshot, report

    def query_objects(self, bbox: tuple[int, int, int, int] | None = None, state: str | None = None, object_id: str | None = None) -> list[GameObject]:
        if self.snapshot is None:
            return []
        return query_objects(self.snapshot, bbox=bbox, state=state, object_id=object_id)
