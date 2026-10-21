# -*- coding: utf-8 -*-
"""Solved-level memory: keep what a cleared level taught, without paying for it twice.

Where this comes from
---------------------
`lalalia9/arc3-franzen-v19` adds exactly one thing to Daniel Franzen's Milestone 2
harness (itself Tufa Labs' Duck): a record of each cleared level, written back into
the system prompt. Their public score with it: 31.46. This module is that idea,
re-implemented against our own loop (`agent.py` / `deferred.py`), not copied --
their version hooks `inference.agent.tool_agent`, which we do not have.

The problem it solves
---------------------
A conversation per game is a single rolling window. Our compactor asks the model to
summarise and then keeps only the summary (`prompts.SUMMARY_REQUEST_PROMPT`). A level
usually fills most of a window, so on the next level the model re-derives mechanics it
already paid for. Nothing about that is a bug in the loop; it is the loop's memory
being one window wide.

What this keeps
---------------
On a clear, the facts we already hold: level number, actions used, the winning
attempt's last N actions, and the tail of the turn that cleared it. On the next two
turns the model is asked for one `Rule for level N:` line. The records are pinned as
one block at the END of the system prompt, so the shared start of the prompt is
untouched and stays cached across games.

Two rules that make it free instead of expensive
------------------------------------------------
1. **Rewrite only when the conversation is being rebuilt anyway.** `pin_policy="drain"`
   (the default, and what the upstream patch does) means the block is written during
   compaction -- the moment the server must re-read the conversation regardless, so
   changing the system prompt costs nothing extra. `pin_policy="level"` writes at each
   level-up instead: strictly better for short local runs where compaction never
   happens, but it invalidates the prefix cache, so it is not the Kaggle setting.
2. **A rule is requested, never invented.** If the model does not answer with a
   `Rule for level N:` line, the record says `not stated`. We do not paraphrase its
   reasoning into one -- the upstream measurement is that the model's own sentence is
   what transfers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADER = ("=== SOLVED LEVELS (facts from this game + your own conclusions; "
          "kept for the whole game) ===")
FOOTER = "=== END SOLVED LEVELS ==="
REUSE_HINT = ("Reuse these where the new level has the same controls; "
              "verify with one probe before relying on them.")

ACTION_TAIL = 30          # winning actions kept per level
REASONING_CHARS = 900     # tail of the turn that cleared it
RULE_CHARS = 600
FULL_LEVELS = 3           # actions+reasoning kept only for the newest few levels
ASK_TURNS = 2

ASK_TEMPLATE = (
    "Before planning: you just cleared level {level}. Answer with one line, exactly in "
    "this form:\n"
    "Rule for level {level}: <what solves it, stated as a mechanic a later level can "
    "reuse -- not the route you took>\n"
    "This line is kept for the rest of the game, so make it about the rules."
)


@dataclass
class LevelRecord:
    level: int
    n_actions: int
    actions: list[str] = field(default_factory=list)
    reasoning: str = ""
    rule: str = ""

    def as_text(self, *, with_detail: bool) -> list[str]:
        lines = [f"Level {self.level} - cleared after {self.n_actions} actions.",
                 f"  Your rule: {self.rule or 'not stated'}"]
        if with_detail:
            skipped = self.n_actions - len(self.actions)
            tail = ", ".join(self.actions) if self.actions else "none recorded"
            lines.append("  Last actions before it cleared"
                         + (f" ({skipped} earlier omitted)" if skipped > 0 else "")
                         + f": {tail}")
            if self.reasoning:
                lines.append("  Your reasoning in the turn that cleared it: "
                             + self.reasoning)
        return lines


class SolvedLevelMemory:
    """Records for cleared levels, plus the block that carries them forward."""

    def __init__(self, *, enabled: bool = True, ask_turns: int = ASK_TURNS,
                 full_levels: int = FULL_LEVELS, action_tail: int = ACTION_TAIL,
                 reasoning_chars: int = REASONING_CHARS, rule_chars: int = RULE_CHARS):
        self.enabled = enabled
        self.ask_turns = ask_turns
        self.full_levels = full_levels
        self.action_tail = action_tail
        self.reasoning_chars = reasoning_chars
        self.rule_chars = rule_chars

        self.levels: list[LevelRecord] = []
        self._asking: tuple[int, int] | None = None   # (level, turns left)
        self._block = ""

    # ------------------------------------------------------------------ record

    def record(self, level: int, actions: list[str], reasoning: str = "") -> LevelRecord:
        """Called once when `level` is cleared. Actions are the winning attempt's."""
        tail = [a for a in actions if a][-self.action_tail:]
        rec = LevelRecord(level=int(level), n_actions=len(actions), actions=tail,
                          reasoning=(reasoning or "")[-self.reasoning_chars:])
        self.levels.append(rec)
        if self.enabled:
            self._asking = (rec.level, self.ask_turns)
        return rec

    def get(self, level: int) -> LevelRecord | None:
        for rec in reversed(self.levels):
            if rec.level == int(level):
                return rec
        return None

    # ------------------------------------------------------------------ asking

    def begin_turn(self) -> str | None:
        """Text to append to this turn's user message, or None.

        Called once per turn. Gives up after `ask_turns` turns so a model that
        ignores the request does not get asked forever.
        """
        if not self.enabled or self._asking is None:
            return None
        level, left = self._asking
        rec = self.get(level)
        if rec is None or rec.rule or left <= 0:
            self._asking = None
            return None
        self._asking = (level, left - 1)
        return ASK_TEMPLATE.format(level=level)

    _RULE_RE = re.compile(r"Rule for level\s*(\d+)\s*:[ \t]*", re.I)

    def capture(self, text: str) -> str | None:
        """Pull a `Rule for level N:` line out of a model message. Returns it."""
        if not text:
            return None
        found = None
        for m in self._RULE_RE.finditer(text):
            level = int(m.group(1))
            rec = self.get(level)
            if rec is None or rec.rule:
                continue
            body = text[m.end():]
            first = body.split("\n", 1)[0].strip()
            if len(first) >= 40:
                rule = first
            else:
                parts = [first] if first else []
                for line in body.split("\n")[1:6]:
                    if not line.strip():
                        if parts:
                            break
                        continue
                    parts.append(line.strip())
                rule = " ".join(parts)
            rule = " ".join(rule.split())[:self.rule_chars]
            if len(rule) < 20:
                continue
            rec.rule = rule
            found = rule
            if self._asking and self._asking[0] == level:
                self._asking = None
        return found

    # ------------------------------------------------------------------ block

    def block(self) -> str:
        if not self.enabled or not self.levels:
            return ""
        lines = [HEADER]
        detail_from = max(0, len(self.levels) - self.full_levels)
        for i, rec in enumerate(self.levels):
            lines.extend(rec.as_text(with_detail=i >= detail_from))
        lines.append(REUSE_HINT)
        lines.append(FOOTER)
        return "\n".join(lines)

    def changed(self) -> bool:
        return self.block() != self._block

    def rewrite(self, base_system: str) -> str:
        """`base_system` with the block pinned at its end. Idempotent."""
        block = self.block()
        self._block = block
        if not block:
            return base_system
        return base_system.rstrip("\n") + "\n\n" + block

    # ------------------------------------------------------------------ misc

    def __len__(self) -> int:
        return len(self.levels)

    def stats(self) -> dict:
        return {
            "levels": len(self.levels),
            "with_rule": sum(1 for r in self.levels if r.rule),
            "actions": [len(r.actions) for r in self.levels],
        }
