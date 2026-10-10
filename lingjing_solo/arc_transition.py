"""V14.2-backed transition adapter for the ARC action boundary.

The adapter keeps ARC's GameAction semantics intact while making every chosen
action pass through the V14.2 Field transition:

    ARC observation -> source term -> Field.apply_sources_and_evolve
    chosen GameAction -> action source term -> Field.apply_sources_and_evolve

The adapter is deliberately side-effect-free with respect to the ARC engine:
it never mutates a GameAction or calls env.step. The caller returns the same
validated action to ARC after the physical transition has been recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core.field import Field, FieldConfig


@dataclass(frozen=True)
class TransitionDiagnostics:
    tick: int
    action_name: str
    total_before: float
    total_after: float
    source_total: float
    conservation_residual: float


class V14ArcTransition:
    """Bridge discrete ARC frames/actions into the V14.2 field contract."""

    def __init__(self, diffusion: float = 0.05) -> None:
        self.diffusion = float(diffusion)
        self.field: Optional[Field] = None
        self._last_grid: Optional[np.ndarray] = None
        self._tick = 0
        self.last_diagnostics: Optional[TransitionDiagnostics] = None

    @property
    def gradient(self) -> Optional[np.ndarray]:
        """Return normalized 2-D field-gradient magnitude for CEAX scoring."""
        if self.field is None:
            return None
        phi = np.asarray(self.field.phi[..., 0], dtype=np.float64)
        if phi.size == 0:
            return None
        gy, gx = np.gradient(phi)
        mag = np.hypot(gx, gy)
        peak = float(np.max(mag))
        if peak <= 1e-12:
            return np.zeros_like(mag)
        return mag / peak

    @property
    def tick(self) -> int:
        return self._tick

    def reset(self) -> None:
        self.field = None
        self._last_grid = None
        self._tick = 0
        self.last_diagnostics = None

    def observe(self, grid: Any) -> None:
        """Advance V14.2 with the observed ARC grid as a conservative source."""
        arr = self._normalise_grid(grid)
        if arr is None:
            return
        self._ensure_field(arr.shape)
        source = arr if self._last_grid is None else arr - self._last_grid
        self._advance(source, "OBSERVATION")
        self._last_grid = arr

    def transition(self, action: Any, grid: Any = None) -> Any:
        """Record one chosen ARC action, then return it unchanged to ARC."""
        if getattr(action, "name", None) == "RESET":
            self.reset()
            return action
        arr = self._normalise_grid(grid)
        if arr is not None:
            self._ensure_field(arr.shape)
        elif self.field is None:
            return action
        source = np.zeros(self.field.cfg.shape, dtype=np.float64)
        self._encode_action(source, action)
        self._advance(source, getattr(action, "name", type(action).__name__))
        return action

    def _ensure_field(self, shape: tuple[int, int]) -> None:
        wanted = (int(shape[0]), int(shape[1]), 1)
        if self.field is None or self.field.phi.shape != wanted:
            self.field = Field(
                FieldConfig(shape=wanted, h=1.0, D=self.diffusion, scheme="explicit")
            )
            self._last_grid = None

    @staticmethod
    def _normalise_grid(grid: Any) -> Optional[np.ndarray]:
        if grid is None:
            return None
        arr = np.asarray(grid)
        if arr.ndim != 2 or arr.size == 0:
            return None
        arr = arr.astype(np.float64, copy=False)
        scale = max(float(np.max(np.abs(arr))), 1.0)
        return (arr / scale)[:, :, None]

    def _advance(self, source: np.ndarray, action_name: str) -> None:
        assert self.field is not None
        before = self.field.total()
        self.field.apply_sources_and_evolve(source, dt=1.0)
        after = self.field.total()
        expected = float(np.sum(source))
        residual = (after - before) - expected
        self._tick += 1
        self.last_diagnostics = TransitionDiagnostics(
            tick=self._tick,
            action_name=action_name,
            total_before=before,
            total_after=after,
            source_total=expected,
            conservation_residual=residual,
        )

    @staticmethod
    def _encode_action(source: np.ndarray, action: Any) -> None:
        """Encode action intent as one bounded source impulse.

        Mouse actions use their supplied display coordinates. Keyboard actions
        use a deterministic direction-to-cell projection; RESET is handled
        before this method.
        """
        name = str(getattr(action, "name", ""))
        h, w, _ = source.shape
        data = getattr(action, "data", None) or {}
        if isinstance(data, dict) and "x" in data and "y" in data:
            x = int(round(float(data["x"]) * (h - 1) / 64.0))
            y = int(round(float(data["y"]) * (w - 1) / 64.0))
        else:
            center = (h // 2, w // 2)
            offsets = {
                "ACTION1": (-1, 0),
                "ACTION2": (1, 0),
                "ACTION3": (0, -1),
                "ACTION4": (0, 1),
            }
            dx, dy = offsets.get(name, (0, 0))
            x, y = center[0] + dx, center[1] + dy
        x = min(max(x, 0), h - 1)
        y = min(max(y, 0), w - 1)
        source[x, y, 0] += 1.0


__all__ = ["TransitionDiagnostics", "V14ArcTransition"]
