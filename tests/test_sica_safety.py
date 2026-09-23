from __future__ import annotations

import json

import pytest

from lingjing_solo.sica import (
    CandidateModification,
    CandidateRegistry,
    ImmutablePathError,
    HoldoutGate,
    SafetyGate,
    TabuStore,
    pre_commit_check,
)


def test_immutable_core_rejects_protected_and_traversal_paths():
    with pytest.raises(ImmutablePathError):
        pre_commit_check(["safety_gate.py"])
    with pytest.raises(ImmutablePathError):
        pre_commit_check(["../safety_gate.py"])
    with pytest.raises(ImmutablePathError):
        pre_commit_check(["/tmp/safety_gate.py"])
    pre_commit_check(["lingjing_solo/new_rule.py"])


def test_tabu_store_requires_executor_capability_and_persists_anchor(tmp_path):
    store = TabuStore(tmp_path / "tabu.jsonl")
    entry = {
        "game_id": "g1", "level": 0, "action_sequence": ["ACTION1"],
        "env_feedback": {"state": "NOT_FINISHED"}, "counter_delta": 1,
        "frame_hash_before": "before", "frame_hash_after": "after",
        "timestamp": 1, "source": "env_executor", "env_signature": "sig",
    }
    with pytest.raises(PermissionError):
        store.write(entry)
    capability = store.register_writer(signature="sig")
    saved = store.write(entry, capability=capability)
    assert saved["source"] == "env_executor"
    assert "env_signature" not in saved
    assert len(saved["entry_hash"]) == 64
    assert store.read()[0]["game_id"] == "g1"
    assert json.loads((tmp_path / "tabu.jsonl").read_text())


def test_tabu_store_rejects_model_source_and_bad_signature(tmp_path):
    store = TabuStore(tmp_path / "tabu.jsonl")
    capability = store.register_writer(signature="sig")
    base = {
        "game_id": "g1", "level": 0, "action_sequence": [], "env_feedback": {},
        "counter_delta": 0, "frame_hash_before": "b", "frame_hash_after": "a",
        "timestamp": 1, "source": "model", "env_signature": "sig",
    }
    with pytest.raises(PermissionError):
        store.write(base, capability=capability)
    base["source"] = "env_executor"
    base["env_signature"] = "wrong"
    with pytest.raises(ValueError):
        store.write(base, capability=capability)


def _candidate(**kwargs):
    values = dict(
        candidate_id="c1", target="rule_set", content={"rule": "x"},
        affected_paths=("lingjing_solo/rules.py",), evidence_refs=("e1", "e2", "e3"),
        evidence_count=3, distinct_games=2, complexity_delta=1.0, expected_gain=0.2,
    )
    values.update(kwargs)
    return CandidateModification(**values)


def test_safety_gate_passes_valid_candidate_and_rejects_failures():
    gate = SafetyGate()
    result = gate.evaluate(_candidate())
    assert result.passed is True
    assert gate.require(_candidate()).candidate_id == "c1"

    rejected = gate.evaluate(_candidate(affected_paths=("safety_gate.py",)))
    assert rejected.passed is False
    assert rejected.checks["immutable_core"] is False

    rejected = gate.evaluate(_candidate(evidence_count=2, evidence_refs=("e1", "e2")))
    assert rejected.checks["evidence_threshold"] is False

    rejected = gate.evaluate(_candidate(recursion_depth=3))
    assert rejected.checks["recursion_depth"] is False

    rejected = gate.evaluate(_candidate(expected_gain=0.01))
    assert rejected.checks["complexity_ratio"] is False


def test_candidate_registry_is_the_merge_boundary():
    registry = CandidateRegistry()
    accepted = registry.submit(_candidate())
    assert registry.get(accepted.candidate_id) == accepted
    with pytest.raises(ValueError):
        registry.submit(accepted)
    with pytest.raises(PermissionError):
        registry.submit(_candidate(candidate_id="bad", evidence_count=0, evidence_refs=()))


def test_holdout_gate_requires_same_nonempty_cases_and_rejects_regression():
    gate = HoldoutGate(metric="score")
    cases = ["a", "b"]
    result = gate.evaluate(
        cases,
        baseline=lambda case: {"score": 10.0},
        candidate=lambda case: {"score": 10.5},
    )
    assert result.passed is True
    assert result.delta == pytest.approx(0.5)

    result = gate.evaluate(
        cases,
        baseline=lambda case: {"score": 10.0},
        candidate=lambda case: {"score": 9.0},
    )
    assert result.passed is False
    assert "regresses" in result.reason

    empty = gate.evaluate([], lambda _: {"score": 1}, lambda _: {"score": 1})
    assert empty.passed is False


def test_agent_wires_environment_transition_to_tabu_store(tmp_path):
    import numpy as np
    from lingjing_solo.agent import LingjingSoloAgent
    from lingjing_solo.core import Frame

    store = TabuStore(tmp_path / "tabu.jsonl")
    agent = LingjingSoloAgent(tabu_store=store, env_signature="env-v1")
    before = Frame(np.zeros((64, 64), dtype=np.int64), t=0)
    after = Frame(np.ones((64, 64), dtype=np.int64), t=1)
    agent.field.update(before)
    agent.field.update(after, before, "ACTION1")
    agent._record_tabu_transition(after)
    entries = store.read()
    assert len(entries) == 1
    assert entries[0]["source"] == "env_executor"
    assert entries[0]["frame_hash_before"] != entries[0]["frame_hash_after"]
    agent._record_tabu_transition(after)
    assert len(store.read()) == 1
