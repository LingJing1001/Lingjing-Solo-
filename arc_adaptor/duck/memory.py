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

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

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

    def to_dict(self) -> dict:
        """Disk format. `actions` is the (already tailed) list; we keep it verbatim."""
        return {
            "level": int(self.level),
            "n_actions": int(self.n_actions),
            "actions": list(self.actions),
            "reasoning": str(self.reasoning),
            "rule": str(self.rule),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LevelRecord":
        return cls(
            level=int(d.get("level", 0)),
            n_actions=int(d.get("n_actions", 0)),
            actions=[str(a) for a in d.get("actions", []) if a][:ACTION_TAIL],
            reasoning=str(d.get("reasoning", ""))[-REASONING_CHARS:],
            rule=str(d.get("rule", ""))[:RULE_CHARS],
        )


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

    # ------------------------------------------------------------ cross-game
    #
    # SolvedLevelMemory 的原生范围是 "this game"（HEADER 字面写明）。这一段把
    # 范围扩到"跨进程的同 game_id"：每局结束把通关规则落盘，下一局开始时读回。
    # 这是补 lingjing_solo transfer 链路未接入 my_agent 的真实缺口，不是替代
    # 它——是先有"局间隔离"的最小可工作版本，避免一次性引入整个 transfer 子系统。
    #
    # 范围说明：仅"同 game_id"跨进程持久化，不跨游戏系列。sb26 学到的规则不会
    # 自动带到 sb27（那是 lingjing_solo transfer 的"同族迁移"，更重）。

    def to_payload(self) -> dict:
        """All cleared levels, ready to write to disk."""
        return {
            "version": 1,
            "game_id": getattr(self, "_game_id", None),
            "levels": [r.to_dict() for r in self.levels],
        }

    def merge_records(self, records: list[LevelRecord], *, overwrite: bool = False) -> int:
        """Fold in records loaded from disk.

        `overwrite=True` replaces any same-level record (later run wins); the
        default only fills levels we don't already have, so an in-progress game
        never loses what it just cleared. Returns the number of records added.
        """
        if not records:
            return 0
        have = {r.level for r in self.levels}
        added = 0
        for rec in records:
            if rec.level in have and not overwrite:
                continue
            if overwrite and rec.level in have:
                for i, r in enumerate(self.levels):
                    if r.level == rec.level:
                        self.levels[i] = rec
                        break
            else:
                self.levels.append(rec)
                have.add(rec.level)
                added += 1
        self._block = ""  # force rewrite() next time
        return added

    @staticmethod
    def _path_for(dir_: str | os.PathLike, game_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(game_id))
        return Path(dir_) / f"{safe}.json"

    def load_from_dir(self, dir_: str | os.PathLike, game_id: str) -> int:
        """Read previous-run rules for `game_id`. Returns records loaded.

        Silent on missing file / parse error: cross-game is best-effort, never
        blocks the run. Caller should set `_game_id` so save_to_dir can mirror.
        """
        self._game_id = str(game_id)
        try:
            path = self._path_for(dir_, game_id)
            if not path.is_file():
                return 0
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
            records = [LevelRecord.from_dict(d) for d in data.get("levels", [])]
            return self.merge_records(records, overwrite=False)
        except Exception:  # noqa: BLE001 - never block the run on bad disk state
            return 0

    def save_to_dir(self, dir_: str | os.PathLike, game_id: str) -> bool:
        """Atomically persist cleared levels for `game_id`. Best-effort."""
        if not self.levels:
            return False
        try:
            path = self._path_for(dir_, game_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self.to_payload(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, path)  # atomic on the same filesystem
            return True
        except Exception:  # noqa: BLE001 - disk full / permission: don't crash the run
            return False


# ============================================================
# FailedRoutesMemory —— 死路教训（与 SolvedLevelMemory 平行结构）
# ============================================================
#
# 设计意图
# ------------
# SolvedLevelMemory 记"成功的规则"（通关后让模型写 Rule for level N）。
# FailedRoutesMemory 记"失败的路线"——某关死亡时最后 N 步动作序列 + 死因。
# 两块独立、并行，都钉在 system prompt 末尾跨关跨局复用。
#
# 为什么不直接复用 SolvedLevelMemory
# -----------------------------------
# 成功和失败的记忆语义不同：成功只记通关瞬间一次（rule 是模型自己写的），
# 失败可能反复出现（同关同一死因记一条，不同死路才记多条）。两者去重规则
# 不同，强行合一只会让代码混乱。
#
# 记什么
# --------
# - level：死在哪一关
# - last_actions：死亡前最后 N 步（默认 15，够让模型识别"这是死路"）
# - reason：game_over / timeout / stuck（"stuck" 留给卡死模式，目前只用 game_over）
# - n_actions_total：本关一共走了多少步（判断是否是"反复横跳"模式）
#
# 不记什么
# --------
# - 不记整关全部动作（payload 太大，没必要）
# - 不记中间推理尾（失败时推理多半是错的，记了反而误导下次）


FAIL_ACTION_TAIL = 15          # 死亡前最后 N 步动作
FAIL_MAX_PER_LEVEL = 3         # 同关同一死因最多记 3 条，避免 prompt 膨胀
FAIL_REASON_CHARS = 200

FAIL_HEADER = ("=== KNOWN DEAD-ENDS (routes that killed an attempt; "
               "do not replay them) ===")
FAIL_FOOTER = "=== END KNOWN DEAD-ENDS ==="
FAIL_AVOID_HINT = ("When the board reaches one of these states, change your "
                   "approach -- do not take the same action sequence.")

FAIL_ASK_TEMPLATE = (
    "Before planning: attempt on level {level} died after {n} actions. Answer "
    "with one line, exactly in this form:\n"
    "Dead-end for level {level}: <the wrong move or wrong assumption that "
    "killed it, stated as a fact a later attempt should avoid>\n"
    "This line is kept for the rest of the game, so make it about the route."
)


@dataclass
class FailedRoute:
    level: int
    last_actions: list[str] = field(default_factory=list)
    reason: str = "game_over"
    n_actions_total: int = 0
    avoid: str = ""   # 模型自己写的"避坑线"，类似 LevelRecord.rule

    def to_dict(self) -> dict:
        return {
            "level": int(self.level),
            "last_actions": list(self.last_actions),
            "reason": str(self.reason),
            "n_actions_total": int(self.n_actions_total),
            "avoid": str(self.avoid),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FailedRoute":
        return cls(
            level=int(d.get("level", 0)),
            last_actions=[str(a) for a in d.get("last_actions", []) if a][:FAIL_ACTION_TAIL],
            reason=str(d.get("reason", "game_over"))[:FAIL_REASON_CHARS],
            n_actions_total=int(d.get("n_actions_total", 0)),
            avoid=str(d.get("avoid", ""))[:RULE_CHARS],
        )

    def as_text(self) -> list[str]:
        tail = ", ".join(self.last_actions) if self.last_actions else "none"
        lines = [f"Level {self.level} - {self.reason} after "
                 f"{self.n_actions_total} actions."]
        lines.append(f"  Last {len(self.last_actions)} actions before death: {tail}")
        if self.avoid:
            lines.append(f"  Avoid: {self.avoid}")
        return lines


class FailedRoutesMemory:
    """Dead routes recorded across attempts. Same shape as SolvedLevelMemory."""

    def __init__(self, *, enabled: bool = True,
                 action_tail: int = FAIL_ACTION_TAIL,
                 max_per_level: int = FAIL_MAX_PER_LEVEL,
                 ask_turns: int = 1):
        self.enabled = enabled
        self.action_tail = action_tail
        self.max_per_level = max_per_level
        self.ask_turns = ask_turns

        self.routes: list[FailedRoute] = []
        self._asking: tuple[int, int] | None = None   # (level, turns left)
        self._block = ""
        self._game_id: str | None = None

    # ------------------------------------------------------------------ record

    def record(self, level: int, level_actions: list[str],
               reason: str = "game_over") -> FailedRoute | None:
        """Called when an attempt on `level` ends in a terminal failure.

        Returns the new record if added, or None if it was a duplicate of an
        existing entry (same level + same last-action-tail) -- the upstream
        rule: do not let the prompt bloat with N copies of the same death.
        """
        if not self.enabled:
            return None
        tail = [a for a in level_actions if a][-self.action_tail:]
        # dedup: same level + same last-3-actions signature = same dead end
        sig = tuple(tail[-3:]) if len(tail) >= 3 else tuple(tail)
        for r in self.routes:
            if r.level != int(level):
                continue
            other_tail = r.last_actions
            other_sig = tuple(other_tail[-3:]) if len(other_tail) >= 3 else tuple(other_tail)
            if other_sig == sig and r.reason == reason:
                return None  # duplicate
        per_level = sum(1 for r in self.routes if r.level == int(level))
        if per_level >= self.max_per_level:
            # evict the oldest entry for this level to make room
            for i, r in enumerate(self.routes):
                if r.level == int(level):
                    self.routes.pop(i)
                    break
        rec = FailedRoute(level=int(level), last_actions=tail, reason=reason,
                          n_actions_total=len(level_actions))
        self.routes.append(rec)
        if self.enabled:
            self._asking = (rec.level, self.ask_turns)
        self._block = ""  # force rewrite() next time
        return rec

    def get_routes_for_level(self, level: int) -> list[FailedRoute]:
        return [r for r in self.routes if r.level == int(level)]

    # ------------------------------------------------------------------ asking

    def begin_turn(self) -> str | None:
        """Same idea as SolvedLevelMemory.begin_turn -- ask for an 'avoid' line."""
        if not self.enabled or self._asking is None:
            return None
        level, left = self._asking
        recs = self.get_routes_for_level(level)
        if not recs:
            self._asking = None
            return None
        rec = recs[-1]
        if rec.avoid or left <= 0:
            self._asking = None
            return None
        self._asking = (level, left - 1)
        return FAIL_ASK_TEMPLATE.format(level=level, n=rec.n_actions_total)

    _AVOID_RE = re.compile(r"Dead-end for level\s*(\d+)\s*:[ \t]*", re.I)

    def capture(self, text: str) -> str | None:
        """Pull a `Dead-end for level N:` line out of a model message."""
        if not text:
            return None
        found = None
        for m in self._AVOID_RE.finditer(text):
            level = int(m.group(1))
            recs = self.get_routes_for_level(level)
            if not recs or recs[-1].avoid:
                continue
            body = text[m.end():]
            first = body.split("\n", 1)[0].strip()
            if len(first) >= 40:
                avoid = first
            else:
                parts = [first] if first else []
                for line in body.split("\n")[1:6]:
                    if not line.strip():
                        if parts:
                            break
                        continue
                    parts.append(line.strip())
                avoid = " ".join(parts)
            avoid = " ".join(avoid.split())[:RULE_CHARS]
            if len(avoid) < 20:
                continue
            recs[-1].avoid = avoid
            found = avoid
            if self._asking and self._asking[0] == level:
                self._asking = None
        return found

    # ------------------------------------------------------------------ block

    def block(self) -> str:
        if not self.enabled or not self.routes:
            return ""
        lines = [FAIL_HEADER]
        for r in self.routes:
            lines.extend(r.as_text())
        lines.append(FAIL_AVOID_HINT)
        lines.append(FAIL_FOOTER)
        return "\n".join(lines)

    def changed(self) -> bool:
        return self.block() != self._block

    def __len__(self) -> int:
        return len(self.routes)

    def rewrite(self, base_system: str) -> str:
        """`base_system` with this block pinned at its end. Idempotent.

        Strips any previous instance of this block (between HEADER and FOOTER)
        from `base_system` before appending the current one -- so calling
        rewrite() twice with the same base returns the same string. This
        matters because `_refresh_system` calls us whenever `changed()`, and
        the block may have grown since the last call.
        """
        block = self.block()
        self._block = block
        if not block:
            return base_system
        # Strip any prior dead-end block from the base before re-pinning.
        cleaned = self._strip_block(base_system)
        return cleaned.rstrip("\n") + "\n\n" + block

    def _strip_block(self, text: str) -> str:
        """Remove a prior KNOWN DEAD-ENDS block from `text`, if present."""
        if FAIL_HEADER not in text or FAIL_FOOTER not in text:
            return text
        start = text.find(FAIL_HEADER)
        end = text.find(FAIL_FOOTER, start) + len(FAIL_FOOTER)
        # also eat the leading "\n\n" that pinned the block
        while start > 0 and text[start - 1] == "\n":
            start -= 1
        return text[:start].rstrip() + text[end:]

    def stats(self) -> dict:
        return {
            "routes": len(self.routes),
            "with_avoid": sum(1 for r in self.routes if r.avoid),
            "per_level": {int(lv): sum(1 for r in self.routes if r.level == int(lv))
                          for lv in sorted({r.level for r in self.routes})},
        }

    # ---------------------------------------------------------- cross-game

    def to_payload(self) -> dict:
        return {
            "version": 1,
            "game_id": self._game_id,
            "routes": [r.to_dict() for r in self.routes],
        }

    def merge_routes(self, records: list[FailedRoute], *, overwrite: bool = False) -> int:
        """Same merge semantics as SolvedLevelMemory: disk only fills, never
        clobbers in-game records unless overwrite=True."""
        if not records:
            return 0
        added = 0
        for rec in records:
            # dedup using same signature as record()
            sig = tuple(rec.last_actions[-3:]) if len(rec.last_actions) >= 3 \
                else tuple(rec.last_actions)
            dup = False
            for r in self.routes:
                if r.level != rec.level:
                    continue
                other = r.last_actions
                other_sig = tuple(other[-3:]) if len(other) >= 3 else tuple(other)
                if other_sig == sig and r.reason == rec.reason:
                    if overwrite:
                        r.avoid = rec.avoid or r.avoid
                    dup = True
                    break
            if dup:
                continue
            self.routes.append(rec)
            added += 1
        self._block = ""
        return added

    @staticmethod
    def _path_for(dir_: str | os.PathLike, game_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(game_id))
        return Path(dir_) / f"{safe}.failures.json"

    def load_from_dir(self, dir_: str | os.PathLike, game_id: str) -> int:
        self._game_id = str(game_id)
        try:
            path = self._path_for(dir_, game_id)
            if not path.is_file():
                return 0
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
            records = [FailedRoute.from_dict(d) for d in data.get("routes", [])]
            return self.merge_routes(records, overwrite=False)
        except Exception:  # noqa: BLE001
            return 0

    def save_to_dir(self, dir_: str | os.PathLike, game_id: str) -> bool:
        if not self.routes:
            return False
        try:
            path = self._path_for(dir_, game_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self.to_payload(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, path)
            return True
        except Exception:  # noqa: BLE001
            return False


