"""结构化竞争假设：用 kind 枚举，禁止靠名字子串匹配。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class HypothesisKind(str, Enum):
    BLOCKED_TRANSITION = "blocked_transition"
    MOVEMENT = "movement"
    TOGGLE_ON_CONTACT = "toggle_on_contact"
    CLICK_EFFECT = "click_effect"
    LEVEL_PROGRESS = "level_progress"
    NOOP = "noop"
    UNKNOWN = "unknown"


@dataclass
class StructuredHypothesis:
    kind: HypothesisKind
    confidence: float = 0.5
    evidence: List[Any] = field(default_factory=list)
    contradictions: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.kind.value


class CompetingHypotheses:
    def __init__(self):
        self.items: Dict[str, StructuredHypothesis] = {}

    def add(self, kind: HypothesisKind, **meta) -> StructuredHypothesis:
        key = kind.value
        if key not in self.items:
            self.items[key] = StructuredHypothesis(kind=kind, meta=dict(meta))
        elif meta:
            self.items[key].meta.update(meta)
        return self.items[key]

    def support(self, kind: HypothesisKind, evidence=None, amount: float = 0.12, **meta):
        h = self.add(kind, **meta)
        h.confidence = min(0.99, h.confidence + amount)
        if evidence is not None:
            h.evidence.append(evidence)
        return h

    def contradict(self, kind: HypothesisKind, evidence=None, amount: float = 0.20):
        h = self.add(kind)
        h.confidence = max(0.01, h.confidence - amount)
        if evidence is not None:
            h.contradictions.append(evidence)
        return h

    def ranked(self) -> List[StructuredHypothesis]:
        return sorted(self.items.values(), key=lambda h: h.confidence, reverse=True)

    def get(self, kind: HypothesisKind) -> Optional[StructuredHypothesis]:
        return self.items.get(kind.value)
