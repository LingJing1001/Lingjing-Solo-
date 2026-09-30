"""可执行世界模型验证门（2026 前沿 Verification 的代数实现）。

规则只有在历史转移上残差够小才入库 / 升置信度。
"""
from __future__ import annotations

from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

from .cr_rules import CRRule, CRRuleBank, _unit


class VerifyWM:
    """Replay-verify CR rules against observed (ctx, action, delta) transitions."""

    def __init__(self, dim: int = 64, tol: float = 0.45, history: int = 128):
        self.dim = int(dim)
        self.tol = float(tol)
        self.bank = CRRuleBank(dim=dim)
        self.history: Deque[Tuple[np.ndarray, str, np.ndarray]] = deque(maxlen=history)
        self.stats = {
            "proposed": 0,
            "accepted": 0,
            "rejected": 0,
            "replays": 0,
            "pass_rate": 0.0,
        }

    def _pad(self, v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=np.float32).reshape(-1)
        out = np.zeros(self.dim, dtype=np.float32)
        out[: min(self.dim, x.shape[0])] = x[: self.dim]
        return out

    def residual(self, rule: CRRule, ctx: np.ndarray, delta: np.ndarray) -> float:
        pred = rule.predict_delta(self._pad(ctx))
        d = self._pad(delta)
        # cosine distance + relative L2
        pn = float(np.linalg.norm(pred) + 1e-8)
        dn = float(np.linalg.norm(d) + 1e-8)
        cos = float(np.dot(pred, d) / (pn * dn))
        rel = float(np.linalg.norm(pred - d) / (dn + pn))
        # map to [0,1] residual-like score (0=perfect)
        return float(np.clip(0.5 * (1.0 - cos) + 0.5 * rel, 0.0, 1.0))

    def record(self, ctx: np.ndarray, action: str, delta: np.ndarray) -> None:
        self.history.append((self._pad(ctx), str(action).upper(), self._pad(delta)))

    def verify_rule(self, rule: CRRule, min_hits: int = 1) -> Tuple[bool, float]:
        """Average residual on matching-action history."""
        res = []
        for ctx, act, delta in self.history:
            if act != rule.action:
                continue
            if float(np.linalg.norm(delta)) < 1e-6:
                continue
            res.append(self.residual(rule, ctx, delta))
        self.stats["replays"] += 1
        if len(res) < min_hits:
            # allow bootstrap on the proposing transition only
            return False, 1.0
        mean_r = float(np.mean(res))
        return mean_r <= self.tol, mean_r

    def propose_and_gate(
        self,
        ctx: np.ndarray,
        action: str,
        delta: np.ndarray,
        *,
        force_accept_first: bool = True,
    ) -> Optional[CRRule]:
        """Create CR atom; accept only if verifies (or first-hit bootstrap)."""
        self.stats["proposed"] += 1
        self.record(ctx, action, delta)
        if float(np.linalg.norm(self._pad(delta))) < 1e-5:
            self.stats["rejected"] += 1
            return None
        rule = self.bank.propose_from_transition(ctx, delta, action)
        ok, mean_r = self.verify_rule(rule, min_hits=1)
        if not ok and force_accept_first and len(self.history) <= 2:
            # bootstrap: accept proposing sample, mark low confidence
            rule.confidence = 0.35
            ok = True
            mean_r = self.residual(rule, ctx, delta)
        if ok and mean_r <= self.tol * 1.25:
            rule.verifies += 1
            rule.trials += 1
            rule.confidence = float(np.clip(1.0 - mean_r, 0.2, 0.95))
            self.bank.add(rule)
            self.stats["accepted"] += 1
            self._refresh_pass_rate()
            return rule
        self.stats["rejected"] += 1
        self._refresh_pass_rate()
        return None

    def reinforce(self, action: str, ctx: np.ndarray, delta: np.ndarray) -> None:
        """Re-check existing rules; bump or decay confidence."""
        self.record(ctx, action, delta)
        for rule in self.bank.rules:
            if rule.action != str(action).upper():
                continue
            r = self.residual(rule, ctx, delta)
            rule.trials += 1
            if r <= self.tol:
                rule.verifies += 1
                rule.confidence = min(0.99, rule.confidence + 0.04)
            else:
                rule.confidence = max(0.05, rule.confidence - 0.06)
        self._refresh_pass_rate()

    def _refresh_pass_rate(self) -> None:
        a, r = self.stats["accepted"], self.stats["rejected"]
        self.stats["pass_rate"] = float(a / max(1, a + r))

    def score_action_context(self, ctx: np.ndarray, action: str) -> float:
        """Higher if verified rules predict large effect for this context."""
        pred = self.bank.reconstruct(ctx, action=action)
        conf = 0.0
        for rule in self.bank.rules:
            if rule.action == str(action).upper():
                conf = max(conf, rule.confidence)
        return float(np.linalg.norm(pred) * (0.5 + 0.5 * conf))

    def snapshot(self) -> dict:
        return {"stats": dict(self.stats), "bank": self.bank.snapshot()}
