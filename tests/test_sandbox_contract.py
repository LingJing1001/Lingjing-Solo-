"""沙盒执行契约与执行证据（教程 Agent Task2 四层表 / 可复现性五字段表）。

覆盖点：
- exit_code / termination_reason 映射（8 种终止路径）
- ExecutionRecord 的完整证据字段
- state_diff / memory_digest
- SandboxSpec 的 fingerprint 拆分（环境身份 vs 资源预算）与 gaps 自检
- 三种 memory 重置策略
- RunJournal 的统计、回放与"差异可否归因于策略"判定
- Harness 侧的记录落盘
"""

from __future__ import annotations

import pytest

from lingjing_solo.harness.actions import make_action
from lingjing_solo.harness.contract import (
    ACTION_TOOL,
    DEFAULT_SPEC,
    EXIT_CODES,
    LOG_TOOL,
    RESET_POLICIES,
    TERMINATION_REASONS,
    ExecutionRecord,
    RunJournal,
    SandboxSpec,
    ToolSpec,
    apply_reset_policy,
    memory_digest,
    state_diff,
    take_snapshot,
)
from lingjing_solo.harness.loop import FakeEnv, FakeLLM, Harness
from lingjing_solo.harness.sandbox import Sandbox, SandboxContext


def record(code, **ctx):
    return Sandbox().run_record(code, SandboxContext(**ctx))


# --------------------------------------------------------------------------- #
# 1. exit_code / termination_reason 映射
# --------------------------------------------------------------------------- #


def test_exit_code_ok():
    r = record("action('UP')")
    assert r.ok and r.exit_code == 0 and r.termination_reason == "ok"


def test_exit_code_syntax_error():
    r = record("action('UP'")
    assert r.exit_code == 2 and r.termination_reason == "syntax_error"
    assert r.error_kind == "syntax"


def test_exit_code_runtime_error():
    r = record("1/0\naction('SPACE')")
    assert r.exit_code == 1 and r.termination_reason == "runtime_error"


def test_exit_code_timeout():
    r = Sandbox(timeout_s=0.05).run_record(
        "try:\n    while True:\n        pass\nexcept Exception:\n    pass\n"
    )
    assert r.exit_code == 124 and r.termination_reason == "timeout"


def test_exit_code_step_limit():
    r = Sandbox(timeout_s=30.0, max_steps=5_000).run_record("while True:\n    pass\n")
    assert r.exit_code == 125 and r.termination_reason == "step_limit"


def test_exit_code_no_action():
    r = record("x = 1")
    assert r.exit_code == 126 and r.termination_reason == "no_action"


def test_exit_code_code_too_long():
    r = Sandbox(max_code_chars=50).run_record("x=1\n" * 100)
    assert r.exit_code == 127 and r.termination_reason == "code_too_long"


def test_exit_code_tool_not_allowed():
    sb = Sandbox(spec=SandboxSpec(tools=(LOG_TOOL,)))
    r = sb.run_record("action('UP')", SandboxContext())
    assert r.exit_code == 126 and r.termination_reason == "tool_not_allowed"
    assert "not allowed" in (r.observation or "")


def test_exit_codes_are_shell_like():
    assert EXIT_CODES["ok"] == 0
    assert EXIT_CODES["timeout"] == 124  # 与 coreutils timeout 一致
    assert EXIT_CODES["step_limit"] == 125
    assert EXIT_CODES["code_too_long"] == 127
    assert set(EXIT_CODES) == set(TERMINATION_REASONS)


# --------------------------------------------------------------------------- #
# 2. 证据字段
# --------------------------------------------------------------------------- #


def test_observation_ok_format():
    assert record("action('CLICK', 3, 4)").observation == "action=CLICK(3,4)"


def test_observation_failure_carries_reason():
    r = record("action('TELEPORT')")
    assert r.observation.startswith("runtime_error: ")
    assert "unknown action" in r.observation


def test_record_has_tool_args_and_schema_version():
    r = record("action('UP')")
    assert r.tool == "python_exec"
    assert r.args == {"code_chars": len("action('UP')")}
    assert r.schema_version == ACTION_TOOL.schema_version


def test_record_timestamps_are_ordered():
    r = record("action('UP')")
    assert r.started_at > 0 and r.ended_at >= r.started_at
    # 时钟粒度可能粗到 10ms 以上，极快的一步可以是 0.0，但不允许为负
    assert r.duration_s >= 0.0


def test_record_carries_spec_fingerprint():
    spec = SandboxSpec(image="img@sha256:abc")
    r = Sandbox(spec=spec).run_record("action('UP')", SandboxContext())
    assert r.spec_fingerprint == spec.fingerprint()


def test_record_action_is_a_plain_string():
    """证据里只放动作的字符串形式，不放 Action 对象 —— 日志必须可 JSON 化。"""
    r = record("action('DOWN')")
    assert type(r.action) is str and r.action == "DOWN"


def test_unknown_action_name_from_make_action_is_archived():
    with pytest.raises(ValueError):
        make_action("TELEPORT")
    r = record("action('TELEPORT')")
    assert not r.ok


def test_to_jsonl_is_single_line_json():
    import json

    line = record("action('UP')").to_jsonl()
    assert "\n" not in line
    assert json.loads(line)["termination_reason"] == "ok"


# --------------------------------------------------------------------------- #
# 3. state_diff / memory_digest
# --------------------------------------------------------------------------- #


def test_state_diff_classifies_added_removed_changed():
    d = state_diff({"a": 1, "b": 2}, {"b": 3, "c": 4})
    assert d == {"added": ["c"], "removed": ["a"], "changed": ["b"]}


def test_state_diff_empty_when_memory_untouched():
    r = record("action('SPACE')", memory={"keep": 1})
    assert r.state_diff == {"added": [], "removed": [], "changed": []}


def test_state_diff_records_keys_not_values():
    r = record("memory['secret'] = 'hunter2'\naction('SPACE')")
    assert r.state_diff["added"] == ["secret"]
    # 值不进证据（既不泄漏上下文，也保证日志可比）
    assert "hunter2" not in r.to_jsonl()


def test_memory_digest_is_stable_and_key_order_insensitive():
    assert memory_digest({"a": 1, "b": 2}) == memory_digest({"b": 2, "a": 1})


def test_memory_digest_changes_with_value():
    assert memory_digest({"a": 1}) != memory_digest({"a": 2})


def test_memory_digest_survives_unserializable_values():
    class Weird:
        def __repr__(self):
            return "<weird>"

    assert memory_digest({"w": Weird()})  # 不抛异常


# --------------------------------------------------------------------------- #
# 4. 一次执行只跑一次
# --------------------------------------------------------------------------- #


def test_execute_returns_both_views_from_one_run():
    sb = Sandbox()
    ctx = SandboxContext(memory={})
    res, rec = sb.execute("memory['n'] = memory.get('n', 0) + 1\naction('SPACE')", ctx)
    assert res.ok and rec.ok
    assert ctx.memory["n"] == 1  # 只跑了一次
    assert res.action.kind == rec.action


def test_run_still_returns_sandbox_result():
    sb = Sandbox()
    res = sb.run("action('UP')")
    assert res.ok and res.action.kind == "UP"
    assert not hasattr(res, "exit_code")  # 旧接口未被污染


def test_run_record_with_empty_code_is_no_action():
    r = record("")
    assert r.exit_code == 126 and r.termination_reason == "no_action"


# --------------------------------------------------------------------------- #
# 5. SandboxSpec：身份、预算、自检
# --------------------------------------------------------------------------- #


def test_spec_fingerprint_is_stable_across_instances():
    assert SandboxSpec().fingerprint() == SandboxSpec().fingerprint()


def test_spec_fingerprint_sensitive_to_environment_fields():
    base = SandboxSpec().fingerprint()
    assert SandboxSpec(network=True).fingerprint() != base
    assert SandboxSpec(image="other:latest").fingerprint() != base
    assert SandboxSpec(deps_lock={"numpy": "2.1.0"}).fingerprint() != base


def test_budget_fingerprint_separate_from_identity():
    a, b = SandboxSpec(), SandboxSpec(timeout_s=60.0)
    assert a.fingerprint() == b.fingerprint()          # 环境没变
    assert a.budget_fingerprint() != b.budget_fingerprint()  # 预算变了


def test_spec_rejects_unknown_reset_policy():
    with pytest.raises(ValueError):
        SandboxSpec(reset_policy="yolo")


def test_spec_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        SandboxSpec(timeout_s=0)
    with pytest.raises(ValueError):
        SandboxSpec(max_code_chars=0)


def test_tool_signature_rendering():
    assert ACTION_TOOL.signature() == "action(name:str, row:int|None, col:int|None) -> Action"
    assert LOG_TOOL.signature() == "log(msg:Any) -> None"


def test_tool_spec_is_hashable():
    assert len({ACTION_TOOL, ACTION_TOOL}) == 1


def test_gaps_flags_missing_deps_lock_on_default_spec():
    gaps = DEFAULT_SPEC.gaps()
    assert len(gaps) == 1 and "deps_lock" in gaps[0]


def test_gaps_is_empty_for_a_complete_contract():
    spec = SandboxSpec(deps_lock={"numpy": "2.1.0"})
    assert spec.gaps() == []


def test_gaps_flags_permission_boundary():
    spec = SandboxSpec(deps_lock={"numpy": "2.1.0"}, network=True, filesystem="workspace")
    gaps = spec.gaps()
    assert sum("权限边界" in g for g in gaps) == 2


def test_gaps_flags_empty_tool_list():
    spec = SandboxSpec(deps_lock={"numpy": "2.1.0"}, tools=())
    assert any("工具接口" in g for g in spec.gaps())


def test_gaps_flags_missing_log_fields():
    spec = SandboxSpec(deps_lock={"numpy": "2.1.0"}, log_fields=("tool",))
    assert any("日志与回放" in g for g in spec.gaps())


def test_spec_to_dict_includes_both_fingerprints():
    d = SandboxSpec().to_dict()
    assert d["fingerprint"] == SandboxSpec().fingerprint()
    assert d["budget_fingerprint"] == SandboxSpec().budget_fingerprint()


# --------------------------------------------------------------------------- #
# 6. 重置策略
# --------------------------------------------------------------------------- #


def test_reset_policies_enum():
    assert set(RESET_POLICIES) == {"fresh", "snapshot", "persistent"}


def test_apply_reset_persistent_returns_same_object():
    mem = {"a": 1}
    assert apply_reset_policy(mem, "persistent") is mem
    assert mem == {"a": 1}


def test_apply_reset_fresh_clears_in_place():
    mem = {"a": 1}
    apply_reset_policy(mem, "fresh")
    assert mem == {}


def test_apply_reset_snapshot_restores_checkpoint():
    mem = {"a": 1, "junk": 9}
    apply_reset_policy(mem, "snapshot", {"a": 1})
    assert mem == {"a": 1}


def test_apply_reset_snapshot_without_checkpoint_clears():
    mem = {"a": 1}
    apply_reset_policy(mem, "snapshot", None)
    assert mem == {}


def test_apply_reset_rejects_unknown_policy():
    with pytest.raises(ValueError):
        apply_reset_policy({}, "yolo")


def test_take_snapshot_is_a_copy():
    mem = {"a": 1}
    snap = take_snapshot(mem)
    mem["a"] = 2
    assert snap == {"a": 1}


def test_sandbox_uses_spec_budget():
    sb = Sandbox(spec=SandboxSpec(max_code_chars=10, timeout_s=5.0))
    assert sb.max_code_chars == 10 and sb.timeout_s == 5.0


# --------------------------------------------------------------------------- #
# 7. RunJournal
# --------------------------------------------------------------------------- #


def _two_records():
    sb = Sandbox()
    return [
        sb.run_record("action('UP')", SandboxContext()),
        sb.run_record("1/0\naction('SPACE')", SandboxContext()),
        sb.run_record("x = 1", SandboxContext()),
    ]


def test_journal_append_len_and_iteration():
    j = RunJournal()
    recs = _two_records()
    for r in recs:
        j.append(r)
    assert len(j) == 3
    assert list(j)[0] is recs[0]
    assert j.records == recs


def test_journal_failure_reasons_and_success_rate():
    j = RunJournal()
    j.extend(_two_records())
    assert j.failure_reasons() == {"no_action": 1, "runtime_error": 1}
    assert j.success_rate() == pytest.approx(1 / 3)


def test_journal_empty_stats_are_safe():
    j = RunJournal()
    assert j.success_rate() == 0.0 and j.failure_reasons() == {} and j.total_steps == 0


def test_journal_to_jsonl_line_count():
    j = RunJournal()
    j.extend(_two_records())
    assert len(j.to_jsonl().splitlines()) == 3


def test_journal_replay_returns_copies():
    j = RunJournal()
    j.extend(_two_records())
    out = j.replay()
    out[0].termination_reason = "tampered"
    out[0].state_diff["added"].append("bogus")
    assert j.records[0].termination_reason == "ok"
    assert j.records[0].state_diff["added"] == []


def test_journal_diff_reports_zero_mismatch_for_identical_runs():
    a, b = RunJournal(), RunJournal()
    a.extend(_two_records())
    b.extend(_two_records())
    d = a.diff_against(b)
    assert d["same_env"] and d["same_budget"]
    assert d["n_mismatch"] == 0 and d["first_mismatch"] is None
    assert d["attributable_to_policy"] is True


def test_journal_diff_refuses_attribution_across_environments():
    a = RunJournal(spec=SandboxSpec())
    b = RunJournal(spec=SandboxSpec(network=True))
    a.extend(_two_records())
    b.extend(_two_records())
    d = a.diff_against(b)
    assert d["same_env"] is False
    assert d["attributable_to_policy"] is False


def test_journal_diff_finds_first_mismatch():
    a, b = RunJournal(), RunJournal()
    a.append(record("action('UP')"))
    b.append(record("action('DOWN')"))
    d = a.diff_against(b)
    assert d["first_mismatch"] == 0 and d["n_mismatch"] == 1


def test_execution_record_comparable_strips_timing():
    r = record("action('UP')")
    c = r.comparable()
    assert "duration_s" not in c and "started_at" not in c and "ended_at" not in c
    assert c["termination_reason"] == "ok"


# --------------------------------------------------------------------------- #
# 8. Harness 落盘
# --------------------------------------------------------------------------- #


def _scripted(snippets):
    return Harness(llm=FakeLLM(script=snippets))


def test_harness_records_one_per_step():
    res = Harness(llm=FakeLLM()).run_episode(FakeEnv(), max_turns=60)
    assert res.solved
    assert len(res.records) == len(res.steps)
    assert all(r.ok for r in res.records)
    assert res.exit_codes == [0] * len(res.steps)


def test_harness_journal_accumulates_across_episodes():
    j = RunJournal()
    h = Harness(llm=FakeLLM(), journal=j)
    h.run_episode(FakeEnv(), max_turns=60)
    n1 = len(j)
    h.run_episode(FakeEnv(), max_turns=60)
    assert len(j) > n1
    assert j.success_rate() == 1.0


def test_harness_no_code_reply_is_journalled():
    h = _scripted(["I cannot help with that."])
    res = h.run_episode(FakeEnv(), max_turns=2)
    assert len(res.records) == 2
    assert all(r.tool == "extract_code" for r in res.records)
    assert all(r.exit_code == 126 for r in res.records)
    assert res.failed_steps == 2


def test_harness_failure_keeps_the_loop_alive():
    h = _scripted(["```python\n1/0\n```", "```python\naction('DOWN')\n```"])
    res = h.run_episode(FakeEnv(), max_turns=2)
    assert len(res.records) == 2
    assert res.records[0].termination_reason == "runtime_error"
    assert res.records[1].ok


def test_harness_reset_policy_defaults_to_spec():
    assert Harness(llm=FakeLLM()).reset_policy == DEFAULT_SPEC.reset_policy
    assert Harness(llm=FakeLLM(), reset_policy="fresh").reset_policy == "fresh"


def test_harness_fresh_policy_never_carries_memory():
    h = Harness(llm=FakeLLM(), reset_policy="fresh")
    res = h.run_episode(FakeEnv(), max_turns=3)
    # 每步都是新环境 → memory 每步都从空开始 → 每步都只有一个新增键
    assert all(r.state_diff["added"] == ["last"] for r in res.records)


def test_harness_persistent_policy_carries_memory():
    h = Harness(llm=FakeLLM(), reset_policy="persistent")
    res = h.run_episode(FakeEnv(), max_turns=3)
    assert res.records[0].state_diff["added"] == ["last"]
    assert res.records[1].state_diff["added"] == []
    assert res.records[1].state_diff["changed"] == ["last"]


def test_harness_snapshot_policy_rolls_back_failed_step():
    rollback = "```python\nmemory['n'] = memory.get('n', 0) + 1\naction('DOWN')\n```"
    poison = "```python\nmemory['n'] = 99\nraise ValueError('boom')\n```"
    h = Harness(llm=FakeLLM(script=[rollback, poison, rollback]), reset_policy="snapshot")
    res = h.run_episode(FakeEnv(), max_turns=3)

    assert res.records[1].memory_digest == memory_digest({"n": 99})  # 证据保留现场
    assert res.records[2].memory_digest == memory_digest({"n": 2})   # 但状态已回滚到 1 再 +1
    assert res.records[2].state_diff["changed"] == ["n"]


def test_harness_records_share_objects_with_journal():
    j = RunJournal()
    res = Harness(llm=FakeLLM(), journal=j).run_episode(FakeEnv(), max_turns=60)
    assert j.records[0] is res.records[0]


def test_harness_diff_between_two_policies_uses_journal():
    """同一环境、同一模型，只换重置策略 —— 差异才可归因于策略本身。"""
    env_a, env_b = FakeEnv(), FakeEnv()
    ja = RunJournal()
    jb = RunJournal()
    Harness(llm=FakeLLM(), journal=ja, reset_policy="persistent").run_episode(env_a, max_turns=6)
    Harness(llm=FakeLLM(), journal=jb, reset_policy="fresh").run_episode(env_b, max_turns=6)
    d = ja.diff_against(jb)
    assert d["same_env"] is True            # 环境一致 → 可以谈归因
    assert d["attributable_to_policy"] is True
    assert d["n_mismatch"] > 0              # 状态确实不同


def test_execution_record_is_constructible_standalone():
    r = ExecutionRecord(tool="t", args={}, observation="o", exit_code=0)
    assert r.ok is True and r.to_dict()["tool"] == "t"
