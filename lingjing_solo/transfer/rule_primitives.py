"""可组合规则原语（CEAX RulePrimitive / RuleComposer 硬化版）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .hypotheses import HypothesisKind


@dataclass(frozen=True)
class RulePrimitive:
    kind: HypothesisKind
    trigger: str
    effect: str
    confidence: float
    family: str = "generic"

    @property
    def name(self) -> str:
        return f"{self.kind.value}:{self.trigger}->{self.effect}"


# 机制族：跨游戏迁移只按族，不按坐标
FAMILY_BY_KIND = {
    HypothesisKind.MOVEMENT: "navigation",
    HypothesisKind.BLOCKED_TRANSITION: "gate_block",
    HypothesisKind.TOGGLE_ON_CONTACT: "switch_gate",
    HypothesisKind.CLICK_EFFECT: "click_interact",
    HypothesisKind.LEVEL_PROGRESS: "progress",
    HypothesisKind.NOOP: "noop",
    HypothesisKind.UNKNOWN: "generic",
}


class RuleComposer:
    """把高置信原子规则组成宏技能表示。"""

    def __init__(self, min_confidence: float = 0.75):
        self.min_confidence = min_confidence

    def compose(self, primitives: Sequence[RulePrimitive]) -> List[RulePrimitive]:
        kept = [p for p in primitives if p.confidence >= self.min_confidence]
        out = list(kept)
        # 经典组合：接触切换 ∘ 阻挡 → 开关门宏技能
        kinds = {p.kind for p in kept}
        if (
            HypothesisKind.TOGGLE_ON_CONTACT in kinds
            and HypothesisKind.BLOCKED_TRANSITION in kinds
        ):
            conf = min(
                p.confidence
                for p in kept
                if p.kind
                in (HypothesisKind.TOGGLE_ON_CONTACT, HypothesisKind.BLOCKED_TRANSITION)
            )
            out.append(
                RulePrimitive(
                    kind=HypothesisKind.TOGGLE_ON_CONTACT,
                    trigger="toggle_on_contact",
                    effect="unblocks_when_toggled",
                    confidence=conf,
                    family="switch_gate",
                )
            )
        if (
            HypothesisKind.CLICK_EFFECT in kinds
            and HypothesisKind.LEVEL_PROGRESS in kinds
        ):
            conf = min(
                p.confidence
                for p in kept
                if p.kind in (HypothesisKind.CLICK_EFFECT, HypothesisKind.LEVEL_PROGRESS)
            )
            out.append(
                RulePrimitive(
                    kind=HypothesisKind.CLICK_EFFECT,
                    trigger="click_effect",
                    effect="may_progress_level",
                    confidence=conf,
                    family="click_interact",
                )
            )
        return out

    def from_hypotheses(self, hypotheses: Iterable) -> List[RulePrimitive]:
        prims: List[RulePrimitive] = []
        for h in hypotheses:
            kind = getattr(h, "kind", None)
            if kind is None:
                continue
            if isinstance(kind, str):
                try:
                    kind = HypothesisKind(kind)
                except ValueError:
                    kind = HypothesisKind.UNKNOWN
            conf = float(getattr(h, "confidence", 0.5))
            family = FAMILY_BY_KIND.get(kind, "generic")
            meta = getattr(h, "meta", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            prims.append(
                RulePrimitive(
                    kind=kind,
                    trigger=kind.value,
                    effect=str(meta.get("effect", kind.value)),
                    confidence=conf,
                    family=family,
                )
            )
        return self.compose(prims)
