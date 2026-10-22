"""Pins solved-level memory's contract, not its wording.

The upstream patch (lalalia9, `arc3-franzen-v19`) only helps because of four
properties, and each one is a way to get it wrong:

1. the block is pinned at the END, so the shared prefix of the system prompt is
   untouched -- otherwise every game loses the prompt cache;
2. a rule is *asked for* and captured verbatim, never paraphrased by us;
3. asking is bounded: a model that ignores the request is not asked forever;
4. older levels keep their rule but drop actions/reasoning, or the block grows
   without bound and the "cache-safe" claim stops being true.

No engine, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc_adaptor"))

from duck.memory import SolvedLevelMemory  # noqa: E402


def _mem(**kw) -> SolvedLevelMemory:
    return SolvedLevelMemory(**kw)


def test_record_renders_facts_and_not_stated_rule():
    m = _mem()
    m.record(0, ["ACTION2", "ACTION4", "ACTION4"], "the box moved right")
    text = m.block()
    assert "=== SOLVED LEVELS" in text
    assert "Level 0 - cleared after 3 actions." in text
    assert "Your rule: not stated" in text
    assert "Last actions before it cleared: ACTION2, ACTION4, ACTION4" in text
    assert "the box moved right" in text
    assert text.rstrip().endswith("=== END SOLVED LEVELS ===")


def test_ask_is_bounded_and_stops_once_answered():
    m = _mem(ask_turns=2)
    m.record(0, ["ACTION1"])
    first = m.begin_turn()
    assert first and "Rule for level 0:" in first
    second = m.begin_turn()
    assert second and "Rule for level 0:" in second
    assert m.begin_turn() is None          # bounded: two turns, then silence

    m2 = _mem()
    m2.record(1, ["ACTION1"])
    assert m2.begin_turn()
    rule = m2.capture("Sure.\nRule for level 1: pressing SPACE toggles the switch "
                      "and the door opens on the next move.")
    assert rule and rule.startswith("pressing SPACE")
    assert m2.begin_turn() is None         # already answered
    assert "Your rule: pressing SPACE" in m2.block()


def test_capture_ignores_other_levels_and_short_rules():
    m = _mem()
    m.record(2, ["ACTION5"])
    assert m.capture("Rule for level 3: this is about a level we never cleared.") is None
    assert m.capture("Rule for level 2: too short") is None
    assert m.get(2).rule == ""


def test_capture_joins_wrapped_rule_lines():
    m = _mem()
    m.record(0, ["ACTION1"])
    text = ("Rule for level 0:\n"
            "the coloured columns fall one cell per DOWN and\n"
            "line up when two of the same colour touch.\n")
    rule = m.capture(text)
    assert rule == ("the coloured columns fall one cell per DOWN and line up when "
                    "two of the same colour touch.")


def test_block_keeps_rules_for_all_levels_but_detail_only_for_newest():
    m = _mem(full_levels=2)
    for lv in range(5):
        m.record(lv, [f"ACTION{lv}"] * 4, f"reasoning {lv}")
        m.get(lv).rule = f"rule {lv}"
    text = m.block()
    for lv in range(5):
        assert f"Your rule: rule {lv}" in text          # every rule survives
    assert "reasoning 0" not in text and "reasoning 2" not in text
    assert "reasoning 3" in text and "reasoning 4" in text
    assert "ACTION4, ACTION4, ACTION4, ACTION4" in text


def test_rewrite_pins_at_end_and_is_idempotent():
    base = "SYSTEM PREAMBLE ...\n"
    m = _mem()
    assert m.rewrite(base) == base                     # nothing recorded, no change
    m.record(0, ["ACTION2"])
    out = m.rewrite(base)
    assert out.startswith(base.rstrip("\n"))           # prefix untouched
    assert out.index("=== SOLVED LEVELS") > len(base)
    assert not m.changed()
    assert m.rewrite(base) == out                      # idempotent


def test_disabled_memory_still_records_but_never_asks_or_pins():
    m = _mem(enabled=False)
    m.record(0, ["ACTION1"], "why")
    assert m.begin_turn() is None
    assert len(m) == 1
    assert m.rewrite("BASE") == "BASE"


def test_action_tail_is_capped_and_counts_the_whole_attempt():
    m = _mem(action_tail=3)
    m.record(0, [f"ACTION1"] * 40)
    rec = m.get(0)
    assert rec.n_actions == 40
    assert len(rec.actions) == 3
    assert "(37 earlier omitted)" in m.block()


def test_stats_reports_rules_captured():
    m = _mem()
    m.record(0, ["ACTION1"])
    m.record(1, ["ACTION2"])
    m.get(1).rule = "press twice"
    assert m.stats() == {"levels": 2, "with_rule": 1, "actions": [1, 1]}
