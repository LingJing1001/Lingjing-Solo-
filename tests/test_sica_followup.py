from __future__ import annotations

import json

import pytest

from lingjing_solo.sica.evolution_controls import (
    ActionExplorationPolicy,
    PerformanceMonitor,
    RollbackManager,
    SnapshotStore,
)
from lingjing_solo.sica.holdout import HoldoutGate
from lingjing_solo.sica.observability import CrossGameValidator, RuleScope
from lingjing_solo.sica.rule_lifecycle import Evidence, RuleRegistry
from lingjing_solo.sica.tabu_store import TabuStore


def _entry(signature: str = "env-v1") -> dict:
    return {
        "game_id": "g1", "level": 1, "action_sequence": ["ACTION1"],
        "env_feedback": {"state": "PLAYING"}, "counter_delta": {"score": 1},
        "frame_hash_before": "before", "frame_hash_after": "after",
        "timestamp": "2026-09-27T00:00:00Z", "source": "env_executor",
        "env_signature": signature,
    }


def test_tabu_read_detects_tampering_and_write_is_durable(tmp_path):
    store = TabuStore(tmp_path / "tabu.jsonl")
    capability = store.register_writer(signature="env-v1")
    written = store.write(_entry(), capability=capability)
    assert store.read()[0]["entry_hash"] == written["entry_hash"]
    path = tmp_path / "tabu.jsonl"
    value = json.loads(path.read_text())
    value["counter_delta"] = {"score": 99}
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.read()


def test_holdout_report_and_minimum_delta():
    gate = HoldoutGate(min_delta=0.1)
    result = gate.evaluate([1, 2], lambda _: {"score": 1.0}, lambda _: {"score": 1.05})
    assert not result.passed
    report = gate.report("candidate-1", result, details={"seed": 7})
    assert report.candidate_id == "candidate-1"
    assert report.details == {"seed": 7}


def test_r9_independence_includes_initial_state_and_level():
    registry = RuleRegistry(evidence_threshold=2)
    registry.propose("r1", {"kind": "probe"})
    registry.add_evidence(Evidence("e1", "r1", "env", "g", "ep", True, 1, "s1", 1))
    rule = registry.add_evidence(Evidence("e2", "r1", "env", "g", "ep", True, 2, "s2", 1))
    assert rule.state.value == "active"
    assert rule.independent_evidence_count == 2


def test_r10_specialized_requires_nonempty_observation_and_positive_result():
    validator = CrossGameValidator()
    assert not validator.validate(
        rule_id="r", scope=RuleScope.SPECIALIZED, declared_family="grid", observed_families=[]
    ).passed
    assert validator.validate(
        rule_id="r", scope=RuleScope.SPECIALIZED, declared_family="grid",
        observed_families=["grid"], positive_families=["grid"],
    ).passed


def test_snapshot_integrity_window_and_cooldown(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = store.create("stable", {"policy": "v1"})
    assert store.restore(snapshot) == {"policy": "v1"}
    assert store.manifest()[0].snapshot_id == "stable"
    monitor = PerformanceMonitor()
    manager = RollbackManager(store, monitor, window_size=3, cooldown_ticks=5)
    assert not manager.observe_window(3.0, tick=1)
    assert not manager.observe_window(2.0, tick=2)
    assert manager.observe_window(1.0, tick=3)
    restored = []
    decision = manager.evaluate_and_rollback(snapshot, 2.0, 1.0, restore=restored.append, tick=3)
    assert decision.degraded and restored == [{"policy": "v1"}]
    assert manager.cooldown_until == 8
    assert not manager.observe_window(0.5, tick=4)


def test_action_exploration_respects_budget():
    policy = ActionExplorationPolicy(ratio=1.0, min_remaining_budget=2)
    assert not policy.should_explore(remaining_budget=1)
    assert policy.should_explore(remaining_budget=2)
