"""Versioned, fail-closed evidence contracts for replayable runs."""

from .protocol import (
    EvidenceValidationError,
    ReplayResult,
    build_manifest,
    build_tick,
    build_verification_report,
    replay_recording,
    validate_manifest,
    validate_tick,
    validate_verification_report,
)

from .ar25 import run_ar25_level
from .ls20 import convert_ls20_recording

__all__ = [
    "EvidenceValidationError",
    "ReplayResult",
    "build_manifest",
    "build_tick",
    "build_verification_report",
    "replay_recording",
    "validate_manifest",
    "validate_tick",
    "validate_verification_report",
    "run_ar25_level",
    "convert_ls20_recording",
]
