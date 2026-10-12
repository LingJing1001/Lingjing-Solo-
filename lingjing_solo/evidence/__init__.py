"""Evidence contracts for replayable runs and Sandbox Evidence Plane v1."""

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
from .adapter import AdapterResult, RecordingAdapter, SandboxAdapter
from .dpl_projection import evaluate_recording_frame, frame_to_array, project_recording_frame
from .schema import EpisodeEvent, EventValidationError

__all__ = [
    "EvidenceValidationError", "ReplayResult", "build_manifest", "build_tick", "build_verification_report",
    "replay_recording", "validate_manifest", "validate_tick", "validate_verification_report", "run_ar25_level",
    "convert_ls20_recording", "AdapterResult", "EpisodeEvent", "EventValidationError", "RecordingAdapter", "SandboxAdapter", "frame_to_array", "project_recording_frame", "evaluate_recording_frame",
]
