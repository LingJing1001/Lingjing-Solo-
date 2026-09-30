import pytest

from lingjing_solo.sica import (
    CandidateModification,
    CandidateRegistry,
    CrossGameValidator,
    Evidence,
    RuleRegistry,
    RuleScope,
    SICAMonitor,
)


def test_r10_general_rules_need_cross_family_validation():
    registry = RuleRegistry()
    rule = registry.propose("general-1", {"action": "ACTION1"}, category="navigation")
    validator = CrossGameValidator()
    assert validator.validate(
        rule_id=rule.rule_id, scope=rule.scope, declared_family=rule.game_family,
        observed_families=["grid", "click"],
    ).passed
    assert not validator.validate(
        rule_id=rule.rule_id, scope=rule.scope, declared_family=rule.game_family,
        observed_families=["grid"],
    ).passed


def test_r10_specialized_rules_cannot_leak_to_other_families():
    registry = RuleRegistry()
    rule = registry.propose("grid-1", {"shape": "square"}, scope=RuleScope.SPECIALIZED,
                            game_family="grid", category="geometry")
    validator = CrossGameValidator()
    assert validator.validate(
        rule_id=rule.rule_id, scope=rule.scope, declared_family=rule.game_family,
        observed_families=["grid"],
    ).passed
    assert not validator.validate(
        rule_id=rule.rule_id, scope=rule.scope, declared_family=rule.game_family,
        observed_families=["grid", "click"],
    ).passed
    with pytest.raises(ValueError):
        registry.propose("bad-general", {}, scope=RuleScope.GENERAL, game_family="grid")
    with pytest.raises(ValueError):
        registry.propose("bad-special", {}, scope=RuleScope.SPECIALIZED)


def test_monitor_displays_required_metrics_and_categories():
    registry = RuleRegistry()
    general = registry.propose("g", {"action": "ACTION1"}, category="navigation")
    registry.propose("s", {"shape": "square"}, scope=RuleScope.SPECIALIZED,
                     game_family="grid", category="geometry")
    for index, family in enumerate(("grid", "click", "symbol")):
        registry.add_evidence(Evidence(f"e{index}", general.rule_id, "source", family,
                                       f"episode-{index}", True, index))
    candidate_registry = CandidateRegistry()
    with pytest.raises(PermissionError):
        candidate_registry.submit(CandidateModification(
            "bad", "policy", {}, evidence_count=0, evidence_refs=(),
        ))
    monitor = SICAMonitor()
    snapshot = monitor.record_system(
        tick=4, potential=monitor.potential(progress=2, novelty=1, risk=0.5),
        performance=0.75, complexity=12.0, rule_registry=registry,
        candidate_registry=candidate_registry,
    )
    assert snapshot.active_rules == 1
    assert snapshot.hypothesis_rules == 1
    assert snapshot.rejected_candidates == 1
    assert snapshot.rule_categories == {"navigation": 1, "geometry": 1}
    assert snapshot.rule_scopes == {"general": 1, "specialized": 1}
    display = monitor.display()
    assert {"potential", "performance", "rules", "complexity",
            "rejected_candidates", "rollback_count", "rule_categories"} <= display.keys()
    assert display["potential"] == pytest.approx(1.75)


def test_monitor_system_counters_are_not_double_counted():
    monitor = SICAMonitor()
    class Counter:
        rejected_count = 2
        rollback_count = 1
    first = monitor.record_system(tick=1, potential=0, performance=0,
                                  complexity=0, candidate_registry=Counter(),
                                  rollback_manager=Counter())
    second = monitor.record_system(tick=2, potential=0, performance=0,
                                   complexity=0, candidate_registry=Counter(),
                                   rollback_manager=Counter())
    assert first.rejected_candidates == second.rejected_candidates == 2
    assert first.rollback_count == second.rollback_count == 1
