#!/usr/bin/env python3
"""Fail-closed import of verified counterfactual labels into canonical AOP format."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.aop.freeze_stage3_dataset import refresh_counterfactual, validate_counterfactual_groups

SCHEMA_VERSION = "aop-canonical-v2"
SOURCE_SCHEMA_VERSION = "aop-counterfactual-shard-v2"


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"provenance manifest missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"provenance manifest invalid: {path}") from exc
    if not isinstance(value, dict) or value.get("provenance_complete") is not True:
        raise ValueError("provenance manifest must declare provenance_complete=true")
    required = ("schema_version", "shard_id", "thread", "source", "created_at")
    missing = [key for key in required if not value.get(key)]
    if missing:
        raise ValueError(f"provenance manifest missing required fields: {', '.join(missing)}")
    if value.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported provenance manifest schema")
    if value.get("collector_exit") != 0:
        raise ValueError("collector_exit must be 0")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(value)
    return rows


def semantic_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    action = str((row.get("requested_action") or {}).get("name", ""))
    payload = json.dumps((row.get("requested_action") or {}).get("payload", {}), sort_keys=True, separators=(",", ":"))
    delta = row.get("state_delta") or {}
    before = str(row.get("replay_state_hash") or delta.get("before_hash") or "")
    after = str(delta.get("after_hash") or "")
    group = str(row.get("counterfactual_group_id") or row.get("episode_id") or "")
    return group, action, payload, before, after


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, str, str]]]:
    if not path.exists():
        return [], set()
    rows = read_jsonl(path)
    return rows, {semantic_key(row) for row in rows}


def ingest_counterfactual_shard(
    source: Path,
    output: Path,
    provenance_manifest: Path | None,
    thread: str,
) -> dict[str, Any]:
    if provenance_manifest is None:
        raise ValueError("provenance manifest is required")
    manifest = load_manifest(provenance_manifest)
    source_digest = sha256(source)
    expected_digest = manifest.get("source_sha256")
    if expected_digest and expected_digest != source_digest:
        raise ValueError("source_sha256 does not match input labels")

    rows = read_jsonl(source)
    if manifest.get("rows") != len(rows):
        raise ValueError("manifest rows does not match input labels")
    declared_labels_digest = manifest.get("labels_sha256")
    if declared_labels_digest and declared_labels_digest != source_digest:
        raise ValueError("labels_sha256 does not match input labels")
    if not rows:
        raise ValueError("counterfactual shard is empty")
    validate_counterfactual_groups(rows, str(manifest.get("shard_id") or source.parent.name))
    existing, seen = _load_existing(output)
    canonical: list[dict[str, Any]] = []
    duplicates = 0
    shard_id = str(manifest.get("shard_id") or source.parent.name)
    for row in rows:
        converted = refresh_counterfactual(row, shard_id, source_digest, str(manifest.get("source_commit") or "") or None)
        converted["schema_version"] = SCHEMA_VERSION
        converted["thread"] = thread
        key = semantic_key(converted)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        canonical.append(converted)

    validate_counterfactual_groups([*existing, *canonical], shard_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in [*existing, *canonical]:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    groups = {str(row.get("counterfactual_group_id")) for row in [*existing, *canonical] if row.get("counterfactual_group_id")}
    output_manifest = (
        output.parent / "manifest.json"
        if output.name == "counterfactual_labels.jsonl"
        else output.with_suffix(".manifest.json")
    )
    output_manifest.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "thread": thread,
                "shard_id": shard_id,
                "created_at": manifest.get("created_at"),
                "source": "remote_counterfactual_v1",
                "source_sha256": source_digest,
                "labels_sha256": sha256(output),
                "collector_exit": 0,
                "rows": len(existing) + len(canonical),
                "rows_read": len(rows),
                "rows_written": len(canonical),
                "rows_total": len(existing) + len(canonical),
                "duplicates": duplicates,
                "groups": len(groups),
                "provenance_complete": True,
                "training_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "rows_read": len(rows),
        "rows_written": len(canonical),
        "rows_total": len(existing) + len(canonical),
        "duplicates": duplicates,
        "groups": len(groups),
        "manifest_path": str(output_manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--thread", choices=("model", "sandbox"), required=True)
    args = parser.parse_args()
    print(json.dumps(ingest_counterfactual_shard(args.input, args.output, args.provenance_manifest, args.thread), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
