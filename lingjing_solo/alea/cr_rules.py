"""Ch1 · A=CR 秩-1 规则原子库。

每条规则 = 外积 c r^T：在上下文方向 r 上激活，产生效应方向 c。
这是 ScriptBank / 点击技能 / 键盘走廊的统一数学身份。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v) + 1e-8)
    return (v / n).astype(np.float32)


@dataclass
class CRRule:
    name: str
    c: np.ndarray  # effect direction
    r: np.ndarray  # context direction
    action: str = "ACTION6"
    confidence: float = 0.5
    trials: int = 0
    verifies: int = 0
    meta: Dict = field(default_factory=dict)

    def predict_delta(self, ctx: np.ndarray) -> np.ndarray:
        """Δ̂ = c · ⟨r, ctx⟩  (rank-1 action)."""
        act = float(np.dot(_unit(self.r), _unit(ctx)))
        return (self.c * act).astype(np.float32)

    def as_matrix(self) -> np.ndarray:
        return np.outer(self.c, self.r).astype(np.float32)


class CRRuleBank:
    """A ≈ C R  — columns of C are effect atoms, rows of R are contexts."""

    def __init__(self, dim: int = 64, max_rules: int = 48):
        self.dim = int(dim)
        self.max_rules = int(max_rules)
        self.rules: List[CRRule] = []

    def propose_from_transition(
        self,
        ctx: np.ndarray,
        delta: np.ndarray,
        action: str,
        name: Optional[str] = None,
    ) -> CRRule:
        r = _unit(ctx)
        c = _unit(delta)
        # scale c by ||delta|| so prediction magnitude matches
        mag = float(np.linalg.norm(delta))
        c = (c * mag).astype(np.float32)
        rule = CRRule(
            name=name or f"cr_{len(self.rules)}_{action}",
            c=c,
            r=r,
            action=str(action).upper(),
            confidence=0.4,
        )
        return rule

    def add(self, rule: CRRule) -> None:
        # merge if nearly parallel
        for old in self.rules:
            if old.action != rule.action:
                continue
            sim_r = abs(float(np.dot(_unit(old.r), _unit(rule.r))))
            sim_c = abs(float(np.dot(_unit(old.c), _unit(rule.c))))
            if sim_r > 0.92 and sim_c > 0.85:
                old.c = _unit(0.7 * old.c + 0.3 * rule.c) * (
                    0.7 * float(np.linalg.norm(old.c)) + 0.3 * float(np.linalg.norm(rule.c))
                )
                old.r = _unit(0.7 * old.r + 0.3 * rule.r)
                old.trials += 1
                old.confidence = min(0.99, old.confidence + 0.05)
                return
        self.rules.append(rule)
        if len(self.rules) > self.max_rules:
            self.rules.sort(key=lambda x: x.confidence * (1 + x.verifies), reverse=True)
            self.rules = self.rules[: self.max_rules]

    def reconstruct(self, ctx: np.ndarray, action: Optional[str] = None) -> np.ndarray:
        """Σ_i c_i ⟨r_i, ctx⟩ over matching actions — A=CR applied to context."""
        out = np.zeros(self.dim, dtype=np.float32)
        ctx_u = _unit(ctx)
        for rule in self.rules:
            if action is not None and rule.action != str(action).upper():
                continue
            if len(rule.c) != self.dim:
                continue
            out = out + rule.predict_delta(ctx_u)
        return out

    def matrix_CR(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.rules:
            return (
                np.zeros((self.dim, 0), np.float32),
                np.zeros((0, self.dim), np.float32),
            )
        C = np.stack([r.c[: self.dim] for r in self.rules], axis=1)
        R = np.stack([r.r[: self.dim] for r in self.rules], axis=0)
        return C.astype(np.float32), R.astype(np.float32)

    def snapshot(self) -> dict:
        return {
            "n_rules": len(self.rules),
            "mean_conf": float(np.mean([r.confidence for r in self.rules])) if self.rules else 0.0,
            "verifies": int(sum(r.verifies for r in self.rules)),
            "top": [
                {"name": r.name, "action": r.action, "conf": round(r.confidence, 3), "ver": r.verifies}
                for r in sorted(self.rules, key=lambda x: -x.confidence)[:5]
            ],
        }
