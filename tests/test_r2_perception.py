from __future__ import annotations

import numpy as np
import pytest

from lingjing_solo.perception.input_contract import InputContractError, validate_input
from lingjing_solo.perception.projection_graph_types import (
    DPLFeatures,
    FieldSnapshot,
    GameObject,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
)


def test_stage0_projection_graph_flag_is_off_by_default(monkeypatch):
    from lingjing_solo.core.config import SoloConfig, projection_graph_enabled

    monkeypatch.delenv("R2_PROJECTION_GRAPH", raising=False)
    assert SoloConfig().r2_projection_graph is False
    assert projection_graph_enabled(SoloConfig()) is False


def test_stage0_environment_flag_can_enable_explicitly(monkeypatch):
    from lingjing_solo.core.config import SoloConfig, projection_graph_enabled

    monkeypatch.setenv("R2_PROJECTION_GRAPH", "on")
    assert projection_graph_enabled(SoloConfig()) is True
    monkeypatch.setenv("R2_PROJECTION_GRAPH", "off")
    assert projection_graph_enabled(SoloConfig(r2_projection_graph=True)) is False


def test_stage1_input_contract_accepts_rgb_and_canonical_action():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    normalized = validate_input(frame, "UP")
    assert normalized.shape == frame.shape
    assert normalized.dtype == frame.dtype
    assert np.array_equal(normalized, frame)


@pytest.mark.parametrize(
    "frame, action",
    [
        (None, "ACTION1"),
        (np.zeros((0, 2), dtype=np.uint8), "ACTION1"),
        (np.zeros((2,), dtype=np.uint8), "ACTION1"),
        (np.zeros((2, 2, 2), dtype=np.uint8), "ACTION1"),
        (np.zeros((2, 2), dtype=object), "ACTION1"),
        (np.full((2, 2), 256, dtype=np.uint16), "ACTION1"),
        (np.zeros((2, 2), dtype=np.uint8), "UNKNOWN"),
    ],
)
def test_stage1_input_contract_rejects_invalid_input(frame, action):
    with pytest.raises(InputContractError):
        validate_input(frame, action)


def test_stage1_graph_snapshot_round_trip_is_deterministic():
    obj = GameObject(
        object_id="obj-1",
        bbox=(0, 1, 2, 3),
        mask=((True, False), (False, True)),
        rgb_histogram=(0.5, 0.25, 0.25),
        dpl=DPLFeatures(gradient_mag=1.0, available=True),
        first_frame=2,
        last_frame=3,
        state="moving",
        velocity=(1.0, 0.0),
    )
    snapshot = GraphSnapshot(
        nodes=(GraphNode(node_id="node-1", object_id="obj-1", features=(1.0, 2.0)),),
        edges=(GraphEdge(src="node-1", dst="node-1", weight=0.5, type="temporal"),),
        level=1,
        timestamp=3.0,
    )
    field = FieldSnapshot(
        schema_version="r2-field.v1",
        frame_index=3,
        field={"version": 1},
        phi={"mean": 0.5},
        time=3.0,
        source_backend="offline",
        visibility="full",
        objects=(obj,),
        graph=snapshot,
        object_changes=("moved:obj-1",),
        phi_summary={"mean": 0.5},
    )
    restored = FieldSnapshot.from_dict(field.to_dict())
    assert restored == field
    assert restored.graph.global_hash == field.graph.global_hash


def test_stage2_first_frame_is_explicit_and_not_fake_zero_delta():
    from lingjing_solo.perception.projection_graph import compute_delta

    result = compute_delta(np.ones((3, 3), dtype=np.uint8), None)
    assert result.is_first_frame is True
    assert result.reason == "no_previous_frame"
    assert np.all(result.magnitude == 0)


def test_stage2_builds_candidates_edges_and_stable_tracking():
    from lingjing_solo.perception.projection_graph import build_projection_graph

    first = np.zeros((8, 8), dtype=np.uint8)
    first[1:3, 1:3] = 255
    first[5:7, 5:7] = 128
    objects1, graph1 = build_projection_graph(first, "UP", constants={"frame_index": 1})
    second = first.copy()
    second[1:3, 2:4] = 255
    objects2, graph2 = build_projection_graph(second, "UP", first, objects1, constants={"frame_index": 2})

    assert len(objects1) == 2
    assert len(graph1.nodes) == 2
    assert all(edge.type in {"spatial", "color", "motion", "geometric", "temporal"} for edge in graph2.edges)
    assert {obj.object_id for obj in objects1} & {obj.object_id for obj in objects2}
    assert graph2.global_hash


def test_stage2_graph_hash_is_order_independent():
    from lingjing_solo.perception.projection_graph import hash_graph

    nodes = (GraphNode("b", "b", (2.0,)), GraphNode("a", "a", (1.0,)))
    edges = (GraphEdge("b", "a", 0.2, "color"), GraphEdge("a", "b", 0.8, "spatial"))
    assert hash_graph(nodes, edges) == hash_graph(tuple(reversed(nodes)), tuple(reversed(edges)))
