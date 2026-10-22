"""Fail-closed input boundary for the R2 projection graph."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.actions import ID_TO_CANONICAL, canonicalize


class R2PerceptionError(ValueError):
    """Structured, fail-closed error for malformed R2 perception inputs."""

    code = "R2_PERCEPTION_ERROR"


class InputContractError(R2PerceptionError):
    """Raised when an R2 frame/action pair violates the input contract."""


_ALLOWED_DTYPES = {np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32), np.dtype(np.float64)}
_ALLOWED_ACTIONS = set(ID_TO_CANONICAL.values()) - {""}
_MAX_PIXELS = 4_000_000


def _canonical_action(action: Any) -> str:
    if isinstance(action, (int, np.integer)) and not isinstance(action, bool):
        try:
            return ID_TO_CANONICAL[int(action)]
        except KeyError as exc:
            raise InputContractError(f"unknown action id: {action!r}") from exc
    if not isinstance(action, str):
        raise InputContractError("action must be a canonical name, alias, or integer id")
    normalized = canonicalize(action)
    if normalized not in _ALLOWED_ACTIONS:
        raise InputContractError(f"unknown action: {action!r}")
    return normalized


def validate_input(
    frame: Any,
    action: Any,
    *,
    min_size: tuple[int, int] = (1, 1),
    max_pixels: int = _MAX_PIXELS,
) -> np.ndarray:
    """Validate an R2 frame/action pair and return an owned frame copy.

    Accepted frames are 2-D grayscale or 3-D ``(H, W, C)`` arrays with
    ``C in {1, 3, 4}``, finite values in the uint8-compatible range [0, 255],
    and one of the explicitly supported numeric dtypes. No reshape,
    default-frame substitution, clipping, or action fallback is performed.
    """
    _canonical_action(action)
    if frame is None:
        raise InputContractError("frame must not be None")
    try:
        array = np.asarray(frame)
    except Exception as exc:  # pragma: no cover - numpy-specific defensive boundary
        raise InputContractError("frame cannot be converted to an array") from exc
    if array.ndim not in (2, 3) or array.size == 0:
        raise InputContractError("frame must be a non-empty 2-D or 3-D array")
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
        raise InputContractError("3-D frame must use channel-last C in {1, 3, 4}")
    if array.dtype not in _ALLOWED_DTYPES:
        raise InputContractError(f"unsupported frame dtype: {array.dtype}")
    height, width = int(array.shape[0]), int(array.shape[1])
    if height < min_size[0] or width < min_size[1]:
        raise InputContractError(f"frame is smaller than minimum size {min_size}")
    if height * width > max_pixels:
        raise InputContractError(f"frame exceeds maximum pixel budget: {max_pixels}")
    if not np.isfinite(array).all():
        raise InputContractError("frame contains NaN or infinite values")
    if np.any(array < 0) or np.any(array > 255):
        raise InputContractError("frame values must be within [0, 255]")
    return array.copy()
