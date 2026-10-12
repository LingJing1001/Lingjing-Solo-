"""Read-only conversion from existing recordings to EpisodeEvent v1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from models.aop.recording_gate import validate_jsonl
from tools.aop.normalize_arc_recording import normalize

from .provenance import recording_hash, replay_hash, sha256_json
from .schema import EpisodeEvent, validate_event_rows


@dataclass(frozen=True)
class AdapterResult:
    event_count: int
    recording_rows: int
    output_dir: Path
    replay_hash: str


class RecordingAdapter:
    def __init__(self, recording: Path, *, environment_id: str, environment_version: str, seed: int, source_commit: str) -> None:
        self.recording = Path(recording)
        self.environment_id = environment_id
        self.environment_version = environment_version
        self.seed = seed
        self.source_commit = source_commit

    def build(self, output_dir: Path) -> AdapterResult:
        if not self.recording.is_file():
            raise ValueError(f"recording does not exist: {self.recording}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_hash = recording_hash(self.recording)
        rep_hash = replay_hash(input_hash, self.environment_id, self.environment_version, self.source_commit)
        with tempfile.TemporaryDirectory(prefix="lingjing-evidence-") as work:
            normalized = Path(work) / "normalized.jsonl"
            count = normalize(self.recording, normalized)
            validation = validate_jsonl(str(normalized))
            rows: list[dict[str, Any]] = []
            for line in normalized.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                before = row["observation"]
                after = row["data"]
                action = row["requested_action"]
                tick = int(row["tick"])
                rows.append({
                    "schema_version": "episode_event.v1",
                    "episode_id": str(row["episode_id"]),
                    "segment": "0",
                    "environment_id": self.environment_id,
                    "environment_version": self.environment_version,
                    "seed": self.seed,
                    "tick": tick,
                    "observation_ref": sha256_json(before.get("frame", before.get("grid"))),
                    "action": {**action, "requested": True},
                    "transition": {"before_tick": tick, "after_tick": tick + 1, "settled": True, "state_before": before.get("state"), "state_after": after.get("state")},
                    "geometric_event": {"features": {"levels_completed_before": before.get("levels_completed", 0), "levels_completed_after": after.get("levels_completed", 0), "progressed": float(after.get("levels_completed", 0) > before.get("levels_completed", 0))}},
                    "gate_results": {"recording": {"passed": True, "validator": "models.aop.recording_gate"}},
                    "hypothesis": None,
                    "replay_hash": rep_hash,
                    "provenance": {"source": "recording", "source_commit": self.source_commit, "input_hash": input_hash, "recording_path": str(self.recording)},
                    "authority_status": "shadow",
                })
        validate_event_rows(rows)
        event_path = output_dir / "episode_events.v1.jsonl"
        event_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        (output_dir / "validation_report.json").write_text(json.dumps({"recording_rows": validation.rows, "event_rows": len(rows), "episodes": validation.episodes, "actions": validation.actions, "authority_status": "shadow"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "provenance.json").write_text(json.dumps({"input_hash": input_hash, "replay_hash": rep_hash, "environment_id": self.environment_id, "environment_version": self.environment_version, "seed": self.seed, "source_commit": self.source_commit}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps({"schema_version": "episode_event.v1", "event_file": "episode_events.v1.jsonl", "authority_status": "shadow", "external_ugc": "disabled", "event_hash": sha256_json(rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return AdapterResult(len(rows), validation.rows, output_dir, rep_hash)
