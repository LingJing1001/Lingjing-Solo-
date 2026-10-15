"""Deterministic provenance and replay hashing helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def recording_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def replay_hash(input_hash: str, environment_id: str, environment_version: str, source_commit: str) -> str:
    return sha256_json({"input_hash": input_hash, "environment_id": environment_id, "environment_version": environment_version, "source_commit": source_commit})
