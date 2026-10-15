"""Formal recording-frame -> DPL projection and gate evaluation boundary."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from lingjing_solo.dpl.gates import check
from lingjing_solo.dpl.projection import project
from lingjing_solo.dpl.types import Modification, ProjectedState, UniversalConstants


def frame_to_array(frame: Any) -> np.ndarray:
    """Convert one recording frame to a finite numeric array without inference."""
    try:
        array = np.asarray(frame, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("recording frame must be numeric and rectangular") from exc
    if array.ndim < 2 or array.size == 0:
        raise ValueError("recording frame must be a non-empty 2D-or-higher array")
    array = np.squeeze(array)
    if array.ndim < 2:
        raise ValueError("recording frame becomes one-dimensional after singleton removal")
    if not np.isfinite(array).all():
        raise ValueError("recording frame contains non-finite values")
    return array


def project_recording_frame(frame: Any, *, constants: UniversalConstants | None = None) -> ProjectedState:
    """Project a recording frame through the repository's formal DPL projector."""
    array = frame_to_array(frame)
    constants = constants or UniversalConstants()
    return project(array, has_temporal=False, constants=constants)


def evaluate_recording_frame(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    constants: UniversalConstants | None = None,
) -> dict[str, Any]:
    """Project the settled frame and run the existing DPL gate.

    The action and before frame are included in the evidence identity, while the
    settled ``after.frame`` is the authoritative input to the DPL projection.
    No score or pass value is inferred from the recording metadata.
    """
    if "frame" not in before or "frame" not in after:
        raise ValueError("recording transition requires before.frame and after.frame")
    constants = constants or UniversalConstants()
    projected = project_recording_frame(after["frame"], constants=constants)
    before_levels = float(before.get("levels_completed", 0))
    after_levels = float(after.get("levels_completed", 0))
    modification = Modification(
        type="recording_transition",
        target=str(action.get("name", action.get("id", "unknown"))),
        params={"levels_delta": after_levels - before_levels},
        info_before=before_levels,
        info_after=after_levels,
    )
    gate = check(projected, constants, modification)
    result = {
        "status": "passed" if gate.passed else "failed",
        "passed": bool(gate.passed),
        "score": float(gate.score),
        "action": dict(action),
        "projection": projection_summary(projected, after["frame"]),
        "gate": _jsonable(asdict(gate)),
    }
    return result


def projection_summary(projected: ProjectedState, frame: Any) -> dict[str, Any]:
    """Return bounded, deterministic metadata rather than serializing arrays."""
    array = frame_to_array(frame)
    return {
        "frame_shape": list(array.shape),
        "frame_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "input_shape": list(projected.input_shape),
        "spatial": float(projected.dim_info.spatial),
        "effective_dim": int(projected.dim_info.effective_dim),
        "d1_available": bool(projected.D1.available),
        "d2_available": bool(projected.D2.available),
        "d3_available": bool(projected.D3.available),
        "d4_available": bool(projected.D4.available),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
