from __future__ import annotations

import numpy as np

from lingjing_solo.core.state_semantics import (
    PersistentStateStore,
    StateTracker,
    TransitionRecord,
    TransitionTable,
    compute_state_key,
)
from lingjing_solo.dpl.types import ProjectedState
from lingjing_solo.perception.dpl_adapter import apply_geometry_gates, projected_state_to_features
from lingjing_solo.perception.projection_graph import build_projection_graph
from lingjing_solo.perception.projection_graph_types import GraphNode, GraphSnapshot
from lingjing_solo.perception.world_model import ProjectionWorldModel, query_objects


def _frame() -> np.ndarray:
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[1:3, 1:3] = 255
    frame[5:7, 5:7] = 128
    return frame


def test_stage3_unavailable_dpl_is_explicit_not_zero_substitution():
    features = projected_state_to_features(ProjectedState())
    assert features.available is False
    assert dict(features.validity)["curvature"] is False
    assert features.curvature is None


def test_stage3_existing_dpl_projection_adapts_with_fake_time_marker():
    from lingjing_solo.perception.dpl_adapter import compute_dpl_features

    state, features = compute_dpl_features(_frame())
    assert state.D1.available is True
    assert features.available is True
    assert features.is_fake_time is True
    assert dict(features.validity)["gradient_mag"] is True


def test_stage3_geometry_gate_records_fake_time_and_edge_evidence():
    objects, graph = build_projection_graph(_frame(), "UP", constants={"frame_index": 1})
    report = apply_geometry_gates(graph, objects, projected_state_to_features(ProjectedState()))
    assert "d4_fake_time" not in report.warnings
    assert any(item.startswith("dpl_unavailable:") for item in report.warnings)
    assert report.evidence["information_conservation_proxy"]["finite"] is True


def test_stage4_world_model_persists_snapshot_and_supports_stable_queries():
    model = ProjectionWorldModel()
    snapshot, report = model.update(_frame(), "UP")
    assert snapshot.frame_index == 0
    assert snapshot.schema_version == "r2-field.v1"
    assert snapshot.objects
    assert query_objects(snapshot)
    assert model.query_objects(object_id=snapshot.objects[0].object_id)
    assert report.graph is not None


def test_stage5_state_key_is_order_stable_and_restart_validated(tmp_path):
    graph = GraphSnapshot(nodes=(GraphNode("b", "b"), GraphNode("a", "a")))
    key1 = compute_state_key(graph, 1, False, ["UP", "DOWN"], {"score": 2, "mode": "x"})
    key2 = compute_state_key(graph, 1, False, ["UP", "DOWN"], {"mode": "x", "score": 2})
    assert key1 == key2

    tracker = StateTracker(loop_window=2)
    tracker.frontier.append(key1)
    tracker.dynamic_context["score"] = 2
    tracker.observe(key1)
    tracker.record_transition(key1, "UP", key1, tick=1)
    path = tmp_path / "state.json"
    PersistentStateStore().save(path, tracker)
    restored = PersistentStateStore().load(path)
    assert restored.frontier == [key1]
    assert restored.transitions.get(key1) == TransitionRecord(key1, "UP", key1, 1, False)


def test_stage5_fifo_eviction_removes_transition_index():
    table = TransitionTable(max_entries=1)
    table.add(TransitionRecord("a", "UP", "b"))
    table.add(TransitionRecord("b", "DOWN", "c"))
    assert table.get("a") is None
    assert "a" not in table.transition_index
    assert table.get("b") is not None
