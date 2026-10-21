from __future__ import annotations

import json

import numpy as np

from lingjing_solo.perception.deployment import (
    DeploymentPolicy,
    OfflineReplayRunner,
    ReplayFrame,
    ReplayRecord,
    compare_shadow,
)
from lingjing_solo.perception.exploration import (
    ActionCandidate,
    FrontierManager,
    Priority,
    PriorityWeights,
    boundary_recheck,
    detect_ui_mask,
    prioritize_actions,
    select_action,
)


def _frame() -> np.ndarray:
    frame = np.zeros((20, 20), dtype=np.uint8)
    frame[2:5, 2:5] = 255
    frame[10:13, 10:13] = 128
    return frame


def test_stage6_mask_excludes_fixed_ui_and_rechecks_boundary_objects():
    mask = detect_ui_mask(_frame(), top_ratio=0.1, bottom_ratio=0.1)
    assert mask.top_rows == 2
    assert mask.bottom_rows == 2
    masked = mask.apply(_frame())
    assert np.all(masked[:2] == 0)
    assert np.all(masked[-2:] == 0)
    assert masked[3, 3] == 255
    assert boundary_recheck([type("Obj", (), {"object_id": "edge", "bbox": (0, 5, 2, 7)})()], (20, 20)) == ("edge",)


def test_stage6_priority_is_configurable_and_frontier_is_bfs():
    candidates = [
        ActionCandidate("LOW", Priority.P3, 0.9, 0.0, 0.0),
        ActionCandidate("HIGH", Priority.P1, 0.7, 0.9, 0.8),
    ]
    ranked = prioritize_actions(candidates, PriorityWeights(0.2, 0.7, 0.1))
    assert ranked[0].action == "HIGH"

    frontier = FrontierManager()
    frontier.mark_untested("goal", ["RIGHT"])
    frontier.update_state("start", "MOVE", "middle")
    frontier.update_state("middle", "MOVE", "goal")
    assert frontier.nearest_frontier("start") == "goal"
    assert select_action("start", candidates, frontier, dpl_gate_passed=True) is not None
    assert select_action("start", candidates, frontier, dpl_gate_passed=False) is None


def test_stage7_offline_replay_logs_required_fields_without_executing_action(tmp_path):
    path = tmp_path / "replay.jsonl"
    records = OfflineReplayRunner(run_id="run-1").run(
        [ReplayFrame(_frame(), action="UP", episode_id="ep-1")], path
    )
    assert len(records) == 1
    record = records[0]
    assert record.transition_result["executed"] is False
    assert record.graph_hash
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert {"run_id", "episode_id", "frame_hash", "graph_hash", "visibility", "latency_ms", "memory_kb"} <= payload.keys()


def test_stage7_partial_visibility_and_invalid_graph_fail_closed():
    policy = DeploymentPolicy(mode="canary", enabled_levels=frozenset({0}))
    assert policy.allows(0, "ep", "full", graph_valid=True) is True
    assert policy.allows(0, "ep", "partial", graph_valid=True) is False
    comparison = compare_shadow("UP", "DOWN", "", graph_valid=False, visibility="partial")
    assert comparison.fail_closed is True
    assert comparison.new_action is None


def test_stage7_replay_schema_rejects_partial_visibility_without_reason():
    record = ReplayRecord("r", "e", 0, "f", "g", None, 0, 0, False, "partial", {}, (), {})
    try:
        record.validate()
    except ValueError as exc:
        assert "partial visibility" in str(exc)
    else:
        raise AssertionError("partial visibility must fail closed")
