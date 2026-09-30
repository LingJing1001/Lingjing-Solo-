import pytest

from lingjing_solo.planning.plan_contract import PlanContractError, build_plan
from lingjing_solo.sica import (
    ComplexityBudget,
    Evidence,
    RuleRegistry,
    RuleState,
    measure_complexity,
)


def _e(rule, ident, *, success=True, source="env", game="g", episode="e"):
    return Evidence(ident, rule, source, game, episode, success, 1)


def test_r3_complexity_is_structural_and_budgeted():
    profile = measure_complexity({"actions": ["A", "B"], "dependencies": ["x"]})
    assert profile.cost > 0
    assert ComplexityBudget(profile.cost).accept(profile)
    with pytest.raises(PlanContractError, match="复杂度超过预算"):
        build_plan(
            planner="generic", input_state_hash="s", candidate_actions=["ACTION1"],
            expected_goal="goal", cost=1, search_budget={"max_nodes": 1},
            validity="candidate", complexity_budget={"max_cost": 1},
        )


def test_r9_hypothesis_requires_three_independent_evidence_records():
    registry = RuleRegistry()
    rule = registry.propose("r1", {"action": "ACTION1"})
    assert rule.state == RuleState.HYPOTHESIS
    registry.add_evidence(_e("r1", "ev-1", source="s1", game="g1", episode="ep1"))
    registry.add_evidence(_e("r1", "ev-2", source="s1", game="g1", episode="ep2"))
    assert rule.state == RuleState.HYPOTHESIS
    with pytest.raises(ValueError):
        registry.add_evidence(_e("r1", "ev-2", source="s2", game="g2", episode="ep2"))
    registry.add_evidence(_e("r1", "ev-3", source="s2", game="g2", episode="ep3"))
    assert rule.state == RuleState.ACTIVE
    assert rule.independent_evidence_count == 3


def test_r6_usage_success_decay_archive_and_eliminate():
    registry = RuleRegistry(archive_after=2, eliminate_after=3, min_success_rate=0.5)
    rule = registry.propose("r2", {"action": "ACTION2"})
    for index in range(3):
        registry.add_evidence(_e("r2", f"ev-{index}", success=False, source=f"s{index}", game=f"g{index}", episode=f"ep{index}"))
    registry.use("r2", tick=0)
    registry.report_outcome("r2", success=False, tick=0)
    assert rule.usage_count == 1
    assert rule.failure_count == 4
    assert rule.confidence == 0
    changed = registry.decay(current_tick=2)
    assert rule in changed
    assert rule.state == RuleState.ARCHIVED
    registry.decay(current_tick=3)
    assert rule.state == RuleState.ELIMINATED
    with pytest.raises(PermissionError):
        registry.add_evidence(_e("r2", "late", source="late", game="late", episode="late"))
