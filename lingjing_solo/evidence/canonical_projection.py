"""Read-only projection from canonical AOP rows to EpisodeEvent v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .provenance import sha256_json
from .schema import EpisodeEvent, EventValidationError, validate_event_rows

SOURCE_SCHEMA_VERSION = "aop-canonical-v2"
TARGET_SCHEMA_VERSION = "episode_event.v1"


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EventValidationError(f"{field} must be a non-empty string")
    return value


def canonical_to_event(
    row: dict[str, Any],
    *,
    environment_id: str,
    environment_version: str,
    seed: int,
    source_commit: str,
) -> dict[str, Any]:
    if row.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported canonical source schema")
    episode_id = _required_text(row.get("episode_id"), "episode_id")
    action = row.get("requested_action")
    if not isinstance(action, dict) or not isinstance(action.get("name"), str) or not action["name"]:
        raise ValueError("canonical row must preserve requested_action.name")
    before = row.get("observation") or row.get("observation_before")
    after = row.get("data") or row.get("observation_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("canonical row must contain observation and data objects")
    delta = row.get("state_delta")
    if not isinstance(delta, dict):
        raise ValueError("canonical row must contain state_delta")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("canonical row provenance is required")
    input_hash = _required_text(provenance.get("sha256"), "provenance.sha256")
    shard_id = _required_text(provenance.get("shard_id"), "provenance.shard_id")
    tick = row.get("tick", row.get("step_id"))
    if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
        raise ValueError("canonical row tick must be a non-negative integer")
    replay_hash = row.get("replay_hash") or sha256_json({"source": input_hash, "episode_id": episode_id, "tick": tick})
    event = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "episode_id": episode_id,
        "environment_id": _required_text(environment_id, "environment_id"),
        "environment_version": _required_text(environment_version, "environment_version"),
        "seed": seed,
        "tick": tick,
        "observation_ref": sha256_json(before.get("frame", before.get("grid", before))),
        "action": {**action, "requested": True},
        "transition": {
            "before_tick": tick,
            "after_tick": tick + 1,
            "settled": True,
            "state_before": before.get("state"),
            "state_after": after.get("state"),
            "before_hash": delta.get("before_hash"),
            "after_hash": delta.get("after_hash"),
        },
        "geometric_event": {
            "features": {
                "action_effective": float(bool(delta.get("action_effective"))),
                "progressed": float(bool(delta.get("progressed"))),
            }
        },
        "gate_results": {"canonical_ingest": {"passed": True, "source_schema": SOURCE_SCHEMA_VERSION}},
        "hypothesis": None,
        "replay_hash": str(replay_hash),
        "provenance": {
            "source": "aop-canonical-v2",
            "source_commit": _required_text(source_commit, "source_commit"),
            "input_hash": input_hash,
            "canonical_shard": shard_id,
        },
        "authority_status": "shadow",
    }
    # EpisodeEvent performs the common finite/settled/action checks.
    return EpisodeEvent.from_dict(event).to_dict()


def project_canonical_rows(
    rows: Iterable[dict[str, Any]],
    *,
    environment_id: str,
    environment_version: str,
    seed: int,
    source_commit: str,
) -> list[dict[str, Any]]:
    projected = [
        canonical_to_event(
            row,
            environment_id=environment_id,
            environment_version=environment_version,
            seed=seed,
            source_commit=source_commit,
        )
        for row in rows
    ]
    validate_event_rows(projected)
    return projected


def project_canonical_jsonl(
    rows: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    environment_id: str,
    environment_version: str,
    seed: int,
    source_commit: str,
) -> dict[str, Any]:
    projected = project_canonical_rows(
        rows,
        environment_id=environment_id,
        environment_version=environment_version,
        seed=seed,
        source_commit=source_commit,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_file = output_dir / "episode_events.v1.jsonl"
    event_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in projected),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "event_file": event_file.name,
        "event_hash": hashlib.sha256(event_file.read_bytes()).hexdigest(),
        "events": len(projected),
        "authority_status": "shadow",
        "training_performed": False,
        "external_ugc": "disabled",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
