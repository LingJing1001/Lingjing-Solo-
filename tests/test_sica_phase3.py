import json

import pytest

from lingjing_solo.sica import (
    CandidateModification,
    CandidateRegistry,
    EvolutionController,
    PerformanceMonitor,
    RollbackManager,
    SnapshotStore,
)


def candidate(cid, *, depth=0):
    return CandidateModification(
        cid, "policy", {"action": "ACTION1"}, recursion_depth=depth,
        evidence_refs=("e1", "e2", "e3"), evidence_count=3,
        distinct_games=2, complexity_delta=1.0, expected_gain=1.0,
    )


def test_r7_snapshot_is_atomic_and_restorable(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    registry = CandidateRegistry(snapshot_store=store)
    registry.submit(candidate("c1"), tick=1)
    manifest = json.loads((tmp_path / "snapshots" / "manifest.jsonl").read_text().splitlines()[0])
    restored = store.restore(manifest["path"])
    assert restored == {"accepted": []}
    assert manifest["snapshot_id"] == "c1"


def test_r8_performance_drop_restores_previous_snapshot(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = store.create("stable", {"version": "stable"})
    manager = RollbackManager(store, PerformanceMonitor())
    applied = []
    decision = manager.evaluate_and_rollback(
        snapshot, 100.0, 90.0, restore=applied.append
    )
    assert decision.degraded is True
    assert applied == [{"version": "stable"}]
    assert manager.last_restored_snapshot == "stable"


def test_r8_non_regression_does_not_restore(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = store.create("stable", {"version": "stable"})
    manager = RollbackManager(store, PerformanceMonitor())
    applied = []
    decision = manager.evaluate_and_rollback(snapshot, 100.0, 100.0, restore=applied.append)
    assert decision.degraded is False
    assert applied == []


def test_r4_cooldown_quota_and_meta_depth():
    controller = EvolutionController(cooldown_ticks=3, exploration_quota=1,
                                      quota_window=5, max_meta_depth=1)
    controller.admit(tick=0, recursion_depth=1, is_exploration=True)
    controller.commit(tick=0, is_exploration=True)
    with pytest.raises(PermissionError, match="cooldown"):
        controller.admit(tick=1, recursion_depth=0, is_exploration=False)
    with pytest.raises(PermissionError, match="quota"):
        controller.admit(tick=3, recursion_depth=0, is_exploration=True)
    with pytest.raises(PermissionError, match="depth"):
        controller.admit(tick=3, recursion_depth=2, is_exploration=False)
    assert controller.exploration_remaining == 0


def test_candidate_registry_enforces_phase3_controls():
    registry = CandidateRegistry(controls=EvolutionController(cooldown_ticks=2, max_meta_depth=1))
    registry.submit(candidate("c1"), tick=5)
    with pytest.raises(PermissionError, match="cooldown"):
        registry.submit(candidate("c2"), tick=6)
    with pytest.raises(PermissionError, match="depth"):
        registry.submit(candidate("c3", depth=2), tick=8)
