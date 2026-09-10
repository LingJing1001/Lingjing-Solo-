"""Ch2 · 消元式假设辨识：矛盾假设被消去，不可辨识时要求升秩实验。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LinearHypothesis:
    hid: str
    direction: np.ndarray  # claimed effect axis
    action: str
    support: float = 0.5
    status: str = "open"  # open | supported | contradicted | unidentified
    notes: List[str] = field(default_factory=list)


class HypothesisEliminator:
    """Elimination on stacked constraints E h ≈ y (Strang Ch2)."""

    def __init__(self, dim: int = 64):
        self.dim = int(dim)
        self.hyps: Dict[str, LinearHypothesis] = {}
        self._E: List[np.ndarray] = []
        self._y: List[float] = []

    def propose(self, hid: str, direction: np.ndarray, action: str) -> LinearHypothesis:
        d = np.asarray(direction, dtype=np.float32).reshape(-1)
        dd = np.zeros(self.dim, dtype=np.float32)
        dd[: min(self.dim, d.shape[0])] = d[: self.dim]
        n = float(np.linalg.norm(dd) + 1e-8)
        dd /= n
        h = LinearHypothesis(hid=hid, direction=dd, action=str(action).upper())
        self.hyps[hid] = h
        return h

    def observe(self, delta: np.ndarray, progressed: bool, action: str) -> Dict[str, str]:
        """Update each open hypothesis; return status map."""
        d = np.asarray(delta, dtype=np.float32).reshape(-1)
        dd = np.zeros(self.dim, dtype=np.float32)
        dd[: min(self.dim, d.shape[0])] = d[: self.dim]
        mag = float(np.linalg.norm(dd) + 1e-8)
        self._E.append(dd / mag)
        self._y.append(1.0 if progressed or mag > 0.05 else 0.0)
        if len(self._E) > 200:
            self._E = self._E[-200:]
            self._y = self._y[-200:]

        out = {}
        for hid, h in self.hyps.items():
            if h.status == "contradicted":
                out[hid] = h.status
                continue
            if h.action != str(action).upper() and action:
                continue
            align = abs(float(np.dot(h.direction, dd / mag)))
            if mag < 1e-6 and not progressed:
                # null observation: weak evidence against "always-effect" claims
                h.support = max(0.05, h.support - 0.03)
                h.notes.append("noop")
            elif align > 0.55:
                h.support = min(0.99, h.support + 0.12 + (0.1 if progressed else 0.0))
                h.status = "supported" if h.support >= 0.7 else "open"
                h.notes.append(f"align={align:.2f}")
            elif align < 0.15 and mag > 0.1:
                h.support = max(0.05, h.support - 0.15)
                if h.support < 0.25:
                    h.status = "contradicted"
                    h.notes.append("eliminated")
            out[hid] = h.status
        return out

    def need_rank_experiment(self) -> bool:
        """Elimination failure analogue: many open hyps, low numerical rank of E."""
        if len(self._E) < 4:
            return True
        M = np.stack(self._E[-32:], axis=0)
        s = np.linalg.svd(M, compute_uv=False)
        rank = int((s > 0.05 * s[0]).sum()) if s.size else 0
        open_n = sum(1 for h in self.hyps.values() if h.status == "open")
        return open_n >= 2 and rank < min(4, M.shape[0] // 2 + 1)

    def snapshot(self) -> dict:
        return {
            "n": len(self.hyps),
            "open": sum(1 for h in self.hyps.values() if h.status == "open"),
            "supported": sum(1 for h in self.hyps.values() if h.status == "supported"),
            "contradicted": sum(1 for h in self.hyps.values() if h.status == "contradicted"),
            "need_rank_exp": self.need_rank_experiment(),
        }
