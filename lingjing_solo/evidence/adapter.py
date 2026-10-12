"""Read-only Sandbox Evidence Plane adapters for existing recordings."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from models.aop.recording_gate import validate_jsonl
from tools.aop.normalize_arc_recording import normalize

from .provenance import recording_hash, replay_hash, sha256_json
from .schema import validate_event_rows

DplEvaluator = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class AdapterResult:
    event_count: int
    recording_rows: int
    output_dir: Path
    replay_hash: str
    input_hash: str
    dpl_status: str


class RecordingAdapter:
    """Convert one recording without modifying it or inventing transitions.

    ``dpl_evaluator`` is deliberately dependency-injected. The adapter owns
    the evidence boundary; a caller supplies the project's real DPL projection
    and gate implementation when the input can be projected into DPL types.
    Without one, the output says ``not_configured`` rather than claiming a gate
    pass.
    """

    def __init__(self, recording: Path, *, environment_id: str, environment_version: str, seed: int, source_commit: str, dpl_evaluator: DplEvaluator | None = None) -> None:
        self.recording = Path(recording)
        self.environment_id = environment_id
        self.environment_version = environment_version
        self.seed = seed
        self.source_commit = source_commit
        self.dpl_evaluator = dpl_evaluator

    def build(self, output_dir: Path) -> AdapterResult:
        if not self.recording.is_file():
            raise ValueError(f"recording does not exist: {self.recording}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_hash = recording_hash(self.recording)
        rep_hash = replay_hash(input_hash, self.environment_id, self.environment_version, self.source_commit)
        with tempfile.TemporaryDirectory(prefix="lingjing-evidence-") as work:
            normalized = Path(work) / "normalized.jsonl"
            normalize(self.recording, normalized)
            validation = validate_jsonl(str(normalized))
            rows: list[dict[str, Any]] = []
            segments: dict[str, int] = {}
            dpl_status = "configured" if self.dpl_evaluator else "not_configured"
            for line in normalized.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                before = row["observation"]
                after = row["data"]
                action = row["requested_action"]
                episode_id = str(row["episode_id"])
                segment = segments.get(episode_id, 0)
                tick = int(row["tick"])
                dpl_result: Mapping[str, Any]
                if self.dpl_evaluator:
                    dpl_result = dict(self.dpl_evaluator(before, after, action))
                    _json_safe(dpl_result)
                else:
                    dpl_result = {"status": "not_configured", "passed": None}
                rows.append({
                    "schema_version": "episode_event.v1", "episode_id": episode_id, "segment": str(segment),
                    "environment_id": self.environment_id, "environment_version": self.environment_version, "seed": self.seed,
                    "tick": tick, "observation_ref": sha256_json(before.get("frame", before.get("grid"))),
                    "action": {**action, "requested": True},
                    "transition": {"before_tick": tick, "after_tick": tick + 1, "settled": True, "state_before": before.get("state"), "state_after": after.get("state")},
                    "geometric_event": {"features": {"levels_completed_before": before.get("levels_completed", 0), "levels_completed_after": after.get("levels_completed", 0), "progressed": float(after.get("levels_completed", 0) > before.get("levels_completed", 0))}},
                    "gate_results": {"recording": {"passed": True, "validator": "models.aop.recording_gate"}, "dpl": dpl_result},
                    "hypothesis": None, "replay_hash": rep_hash,
                    "provenance": {"source": "recording", "source_commit": self.source_commit, "input_hash": input_hash, "recording_path": str(self.recording)},
                    "authority_status": "shadow",
                })
                if action.get("name") == "RESET":
                    segments[episode_id] = segment + 1
        after_hash = recording_hash(self.recording)
        if after_hash != input_hash:
            raise RuntimeError("recording changed during read-only adapter run")
        validate_event_rows(rows)
        event_path = output_dir / "episode_events.v1.jsonl"
        event_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        (output_dir / "validation_report.json").write_text(json.dumps({"recording_rows": validation.rows, "event_rows": len(rows), "episodes": validation.episodes, "actions": validation.actions, "authority_status": "shadow", "dpl_status": dpl_status}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "provenance.json").write_text(json.dumps({"input_hash": input_hash, "replay_hash": rep_hash, "environment_id": self.environment_id, "environment_version": self.environment_version, "seed": self.seed, "source_commit": self.source_commit}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps({"schema_version": "episode_event.v1", "event_file": "episode_events.v1.jsonl", "authority_status": "shadow", "external_ugc": "disabled", "dpl_status": dpl_status, "event_hash": sha256_json(rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return AdapterResult(len(rows), validation.rows, output_dir, rep_hash, input_hash, dpl_status)


class SandboxAdapter(RecordingAdapter):
    """Phase-2 adapter with the repository's formal frame-to-DPL evaluator."""

    def __init__(self, recording: Path, *, environment_id: str, environment_version: str, seed: int, source_commit: str, dpl_evaluator: DplEvaluator | None = None) -> None:
        if dpl_evaluator is None:
            from .dpl_projection import evaluate_recording_frame

            dpl_evaluator = evaluate_recording_frame
        super().__init__(recording, environment_id=environment_id, environment_version=environment_version, seed=seed, source_commit=source_commit, dpl_evaluator=dpl_evaluator)


def _json_safe(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("DPL result contains non-finite numeric value")
    if isinstance(value, dict):
        for item in value.values():
            _json_safe(item)
    elif isinstance(value, list):
        for item in value:
            _json_safe(item)
