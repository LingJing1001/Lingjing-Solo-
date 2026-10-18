"""Episode mind for whole-game Qwen play.

Offline, stdlib only. Borrowed from Nova cognition ideas without importing
the IDE stack:

- compact older steps into a digest, keep a short tail
- layered briefing (explore / stuck / deep)
- dead-action memory and causal edges (action -> effect)
- competing hypotheses with a simple Bayesian update
- carry confirmed rules into the next level
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Edge:
    effect: str
    hits: int = 0
    misses: int = 0

    @property
    def confidence(self) -> float:
        total = self.hits + self.misses
        if total <= 0:
            return 0.0
        return self.hits / total


@dataclass
class _Hyp:
    name: str
    log_odds: float = 0.0

    @property
    def prob(self) -> float:
        # logistic, clamped
        x = max(-8.0, min(8.0, self.log_odds))
        return 1.0 / (1.0 + pow(2.718281828, -x))


@dataclass
class EpisodeMind:
    tail_limit: int = 8
    digest_after: int = 16
    stuck_note_at: int = 24
    reset_at: int = 80
    max_resets: int = 2

    steps: list[str] = field(default_factory=list)
    digest: str = ""
    dead: set[str] = field(default_factory=set)
    edges: dict[str, _Edge] = field(default_factory=dict)
    hyps: dict[str, _Hyp] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)
    last_sig: str = ""
    last_hash: str = ""
    last_levels: int = -1
    same_level: int = -1
    same_level_steps: int = 0
    stuck_resets: int = 0
    level_rules: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.hyps:
            for name in ("push", "click_object", "align", "collect", "toggle"):
                self.hyps[name] = _Hyp(name)

    def observe(self, *, levels: int, grid_hash: str) -> None:
        """Record the outcome of the previous action against the new frame."""
        if self.same_level != levels:
            if self.same_level >= 0 and self.level_rules:
                for rule in self.level_rules:
                    if rule not in self.rules:
                        self.rules.append(rule)
                self.rules = self.rules[-6:]
            self.level_rules = []
            self.same_level = levels
            self.same_level_steps = 0
            self.dead.clear()
        else:
            self.same_level_steps += 1

        if self.last_sig:
            changed = levels != self.last_levels or grid_hash != self.last_hash
            effect = "changed" if changed else "no_effect"
            self.steps.append(f"{self.last_sig} -> {effect} L{levels}")
            edge = self.edges.setdefault(self.last_sig, _Edge(effect))
            if changed:
                edge.effect = "changed"
                edge.hits += 1
                self.dead.discard(self.last_sig)
                self._bump("push", 0.4)
                self._bump("click_object", 0.2 if self.last_sig.startswith("ACTION6") else -0.1)
            else:
                edge.effect = "no_effect"
                edge.misses += 1
                self.dead.add(self.last_sig)
                self._bump("toggle", 0.15)
            if edge.confidence >= 0.75 and edge.hits >= 2 and effect == "changed":
                rule = f"{self.last_sig} often changes the grid"
                if rule not in self.level_rules:
                    self.level_rules.append(rule)
            self._compact()
        self.last_hash = grid_hash
        self.last_levels = levels

    def want_reset(self) -> bool:
        return self.same_level_steps >= self.reset_at and self.stuck_resets < self.max_resets

    def mark_reset(self, levels: int) -> None:
        self.stuck_resets += 1
        self.same_level_steps = 0
        self.dead.clear()
        self.last_sig = "RESET"
        self.steps.append(f"RESET stuck L{levels}")
        self._compact()

    def note_action(self, sig: str, levels: int, step: int) -> None:
        self.last_sig = sig
        self.steps.append(f"do {sig} L{levels} step={step}")
        self._compact()

    def reject(self, sig: str) -> None:
        self.steps.append(f"reject {sig}")
        self._compact()

    def briefing(self, *, levels: int) -> tuple[str, str]:
        """Return (history, note) sized by how stuck the level is."""
        stuck = self.same_level_steps
        if stuck >= self.stuck_note_at:
            tier = "deep"
        elif stuck >= 8:
            tier = "standard"
        else:
            tier = "explore"
        tail_n = {"explore": 4, "standard": 8, "deep": 8}[tier]
        history_parts = []
        if self.digest and tier != "explore":
            history_parts.append("digest: " + self.digest)
        if self.rules and tier == "deep":
            history_parts.append("carried_rules: " + " | ".join(self.rules[-4:]))
        if self.level_rules and tier != "explore":
            history_parts.append("level_rules: " + " | ".join(self.level_rules[-3:]))
        if tier == "deep":
            top = sorted(self.hyps.values(), key=lambda h: h.prob, reverse=True)[:3]
            history_parts.append(
                "hypotheses: " + ", ".join(f"{h.name}={h.prob:.2f}" for h in top)
            )
            strong = [
                f"{sig}:{edge.effect}@{edge.confidence:.2f}"
                for sig, edge in self.edges.items()
                if edge.hits + edge.misses >= 2
            ][:6]
            if strong:
                history_parts.append("causal: " + ", ".join(strong))
        history_parts.extend(self.steps[-tail_n:])
        history = "\n".join(history_parts) or "(start)"

        note = f"tier={tier} L={levels}"
        if stuck >= self.stuck_note_at:
            note += f" no level progress for {stuck} steps; change mechanism"
            lead = max(self.hyps.values(), key=lambda h: h.prob)
            if lead.prob >= 0.62:
                note += f" exploit={lead.name}"
            else:
                note += " explore; confidence low"
        if self.dead:
            note += " forbidden=" + ",".join(sorted(self.dead)[-8:])
        return history, note

    def _bump(self, name: str, delta: float) -> None:
        hyp = self.hyps.setdefault(name, _Hyp(name))
        hyp.log_odds = max(-6.0, min(6.0, hyp.log_odds + delta))

    def _compact(self) -> None:
        if len(self.steps) <= self.digest_after:
            return
        overflow = self.steps[:-self.tail_limit]
        self.steps = self.steps[-self.tail_limit :]
        bits = []
        for line in overflow[-12:]:
            if "->" in line or line.startswith("RESET") or line.startswith("do "):
                bits.append(line)
        chunk = "; ".join(bits)[:500]
        if not chunk:
            return
        self.digest = (self.digest + " | " + chunk).strip(" |")
        if len(self.digest) > 700:
            self.digest = self.digest[-700:]
