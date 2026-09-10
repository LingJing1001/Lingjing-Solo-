"""从结构化假设 / Φ 转移证据抽取可迁移技能（禁止名字子串匹配）。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .hypotheses import HypothesisKind, StructuredHypothesis
from .rule_primitives import FAMILY_BY_KIND, RuleComposer, RulePrimitive


# 可跨游戏导入的机制族（坐标级脚本禁止）
CROSS_GAME_FAMILIES = frozenset(
    {"navigation", "gate_block", "switch_gate", "click_interact", "progress"}
)


class RuleTransfer:
    def __init__(self, min_confidence: float = 0.75):
        self.min_confidence = min_confidence
        self.composer = RuleComposer(min_confidence=min_confidence)

    def extract(self, hypotheses: Iterable) -> Dict[str, dict]:
        """结构化抽取：按 HypothesisKind，不用 'trigger' in name。"""
        skills: Dict[str, dict] = {}
        ranked = list(hypotheses)
        for h in ranked:
            kind = self._kind_of(h)
            conf = float(getattr(h, "confidence", 0.0))
            if conf < self.min_confidence or kind is None:
                continue
            family = FAMILY_BY_KIND.get(kind, "generic")
            skill_id = kind.value
            skills[skill_id] = {
                "representation": self._repr(kind),
                "confidence": conf,
                "kind": kind.value,
                "family": family,
                "cross_game": family in CROSS_GAME_FAMILIES,
            }

        # 组合宏技能
        prims = self.composer.from_hypotheses(ranked)
        for p in prims:
            if p.confidence < self.min_confidence:
                continue
            if p.effect in ("unblocks_when_toggled", "may_progress_level"):
                sid = f"compose:{p.trigger}->{p.effect}"
                skills[sid] = {
                    "representation": f"IF {p.trigger} THEN {p.effect}",
                    "confidence": p.confidence,
                    "kind": p.kind.value,
                    "family": p.family,
                    "cross_game": p.family in CROSS_GAME_FAMILIES,
                    "composed": True,
                }
        return skills

    def extract_from_field(self, field) -> Dict[str, dict]:
        """从 Φ 转移表归纳结构化假设再抽取（无 CEAX 字符串匹配）。"""
        hyps = self.induce_from_transitions(field)
        return self.extract(hyps)

    def induce_from_transitions(self, field) -> List[StructuredHypothesis]:
        """证据驱动：像素 Δ、同态后继、点击动作、关卡进展。"""
        from .hypotheses import CompetingHypotheses

        store = CompetingHypotheses()
        if field is None:
            return []

        idx = getattr(field, "transition_index", {}) or {}
        for (shash, action), transitions in list(idx.items()):
            if not transitions:
                continue
            action_s = str(action)
            total = len(transitions)
            same = sum(1 for t in transitions if t.state_after == t.state_before)
            changed = total - same
            avg_delta = sum(getattr(t, "delta_pixels", 0) for t in transitions) / total
            any_progress = any(getattr(t, "progressed", False) for t in transitions)

            if same / total >= 0.7 and avg_delta <= 0:
                if "ACTION6" in action_s.upper() or action_s.upper() in ("SPACE", "CLICK"):
                    store.support(
                        HypothesisKind.NOOP,
                        evidence=(shash, action_s),
                        amount=0.08,
                    )
                else:
                    store.support(
                        HypothesisKind.BLOCKED_TRANSITION,
                        evidence=(shash, action_s, "no_change"),
                        amount=0.10,
                    )
            if changed / total >= 0.55 and avg_delta > 0:
                store.support(
                    HypothesisKind.MOVEMENT,
                    evidence=(shash, action_s, avg_delta),
                    amount=0.10,
                )
            # 接触切换代理：同动作有时变有时不变，且像素变化中等
            successors = {t.state_after for t in transitions}
            if len(successors) >= 2 and avg_delta > 0:
                store.support(
                    HypothesisKind.TOGGLE_ON_CONTACT,
                    evidence=(shash, action_s, len(successors)),
                    amount=0.14,
                )
            if ("ACTION6" in action_s.upper() or action_s.upper() in ("SPACE", "CLICK")) and avg_delta > 0:
                store.support(
                    HypothesisKind.CLICK_EFFECT,
                    evidence=(shash, action_s, avg_delta),
                    amount=0.12,
                )
            if any_progress:
                store.support(
                    HypothesisKind.LEVEL_PROGRESS,
                    evidence=(shash, action_s),
                    amount=0.15,
                )

        # 也吸收 field.rules 中带 [kind] 标记的结论
        for r in getattr(field, "rules", []) or []:
            conclusion = str(getattr(r, "conclusion", ""))
            conf = float(getattr(r, "confidence", 0.5))
            kind = self._parse_bracket_kind(conclusion)
            if kind is not None and conf >= 0.4:
                store.support(kind, evidence=getattr(r, "premise", ""), amount=0.05)

        return store.ranked()

    @staticmethod
    def _kind_of(h) -> Optional[HypothesisKind]:
        kind = getattr(h, "kind", None)
        if isinstance(kind, HypothesisKind):
            return kind
        if isinstance(kind, str):
            try:
                return HypothesisKind(kind)
            except ValueError:
                return None
        name = str(getattr(h, "name", "") or "")
        try:
            return HypothesisKind(name)
        except ValueError:
            return None

    @staticmethod
    def _parse_bracket_kind(conclusion: str) -> Optional[HypothesisKind]:
        # e.g. "[spatial_change] -> abc" from explorer
        mapping = {
            "noop": HypothesisKind.NOOP,
            "spatial_change": HypothesisKind.MOVEMENT,
            "object_move": HypothesisKind.MOVEMENT,
            "color_toggle": HypothesisKind.TOGGLE_ON_CONTACT,
            "level_progress": HypothesisKind.LEVEL_PROGRESS,
            "win_signal": HypothesisKind.LEVEL_PROGRESS,
        }
        if "[" in conclusion and "]" in conclusion:
            tag = conclusion.split("[", 1)[1].split("]", 1)[0].strip()
            return mapping.get(tag)
        return None

    @staticmethod
    def _repr(kind: HypothesisKind) -> str:
        return {
            HypothesisKind.BLOCKED_TRANSITION: "IF blocked THEN transition yields no state change",
            HypothesisKind.MOVEMENT: "IF move action THEN spatial/object change",
            HypothesisKind.TOGGLE_ON_CONTACT: "IF contact/interaction THEN latent gate/switch toggles",
            HypothesisKind.CLICK_EFFECT: "IF click/ACTION6 THEN pixel or object effect",
            HypothesisKind.LEVEL_PROGRESS: "IF key interaction THEN levels_completed increases",
            HypothesisKind.NOOP: "IF action THEN no observable change",
            HypothesisKind.UNKNOWN: "unknown regularity",
        }.get(kind, kind.value)
