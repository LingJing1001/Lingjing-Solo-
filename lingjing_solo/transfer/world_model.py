"""反事实世界模型：优先桥接 Φ 场 predict_graph，本地转移作回退。"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class CounterfactualWorldModel:
    """(state_sig, action) → next_sig 记忆；可挂 WorldModelField.predict。"""

    def __init__(self, field=None):
        self.field = field
        self.transitions: Dict[Tuple[Any, str], List[Any]] = defaultdict(list)
        self._rule_predictors: List[Any] = []  # Phase3: composed rule callables

    def bind_field(self, field) -> None:
        self.field = field

    def observe(self, state_sig, action: str, next_sig) -> None:
        key = (state_sig, str(action))
        self.transitions[key].append(next_sig)

    def predict(self, state_sig, action: str):
        action = str(action)
        # 1) Φ 场已观测边（哈希级）
        if self.field is not None and hasattr(self.field, "predict"):
            shash = None
            if isinstance(state_sig, str):
                shash = state_sig
            elif hasattr(state_sig, "grid_hash") and state_sig.grid_hash:
                shash = state_sig.grid_hash
            if shash:
                nxt = self.field.predict(shash, action)
                if nxt is not None:
                    return nxt
        # 2) 本地签名转移
        values = self.transitions.get((state_sig, action), [])
        if values:
            return values[-1]
        # 3) 规则组合预测（Phase 3）
        for pred in self._rule_predictors:
            try:
                out = pred(state_sig, action)
            except Exception:
                out = None
            if out is not None:
                return out
        return None

    def uncertainty(self, state_sig, action: str) -> float:
        return 1.0 if self.predict(state_sig, action) is None else 0.0

    def counterfactuals(self, state_sig, actions) -> Dict[str, Any]:
        return {a: self.predict(state_sig, a) for a in actions}

    def register_rule_predictor(self, fn) -> None:
        self._rule_predictors.append(fn)

    def clear_rule_predictors(self) -> None:
        self._rule_predictors.clear()
