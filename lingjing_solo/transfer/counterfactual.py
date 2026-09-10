"""反事实规划：未知后继高 epistemic；可与 explore 分数融合。"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


class CounterfactualPlanner:
    def __init__(self, epistemic_weight: float = 1.0, exploit_weight: float = 0.05):
        self.epistemic_weight = epistemic_weight
        self.exploit_weight = exploit_weight

    def rank(
        self,
        state_sig,
        actions: Sequence[str],
        model,
        *,
        goal=None,
        explore_scores: Optional[dict] = None,
    ) -> List[Tuple[float, str, object]]:
        rows = []
        explore_scores = explore_scores or {}
        for a in actions:
            predicted = model.predict(state_sig, a)
            epistemic = self.epistemic_weight if predicted is None else 0.0
            goal_value = self.exploit_weight if predicted is not None else 0.0
            explore = float(explore_scores.get(a, 0.0))
            score = epistemic + goal_value + 0.25 * explore
            rows.append((score, a, predicted))
        return sorted(rows, reverse=True)

    def fuse_with_explore(
        self,
        state_sig,
        scored_actions: Sequence[Tuple[str, float]],
        model,
        *,
        cf_blend: float = 0.55,
    ) -> List[Tuple[str, float]]:
        """把 CEAX 反事实分与灵境 info-gain 分加权融合。"""
        if not scored_actions:
            return []
        actions = [a for a, _ in scored_actions]
        explore_map = {a: float(s) for a, s in scored_actions}
        ranked = self.rank(state_sig, actions, model, explore_scores=explore_map)
        cf_map = {a: float(s) for s, a, _ in ranked}
        fused = []
        for a, eg in scored_actions:
            cf = cf_map.get(a, 0.0)
            fused.append((a, (1.0 - cf_blend) * eg + cf_blend * cf))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused
