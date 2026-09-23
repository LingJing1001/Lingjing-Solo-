"""R1 append-only environment feedback anchor (tabu store).

The model-facing API is read-only in normal use: a write requires an opaque
executor capability returned during writer registration, the current process
identity, and all environment-anchor fields. This is a process-level boundary,
not a replacement for OS ACLs or a separate executor process.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

REQUIRED_FIELDS = (
    "game_id", "level", "action_sequence", "env_feedback", "counter_delta",
    "frame_hash_before", "frame_hash_after", "timestamp", "source",
)


class TabuEntryError(ValueError):
    """Raised for unauthenticated or malformed feedback."""


@dataclass(frozen=True)
class WriterCapability:
    pid: int
    signature: str
    _token: str


class TabuStore:
    def __init__(self, path: str | Path = "tabu_store.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._writer: WriterCapability | None = None

    def register_writer(self, pid: int | None = None, signature: str | None = None) -> WriterCapability:
        """Register the current executor and return its opaque write capability."""
        actual_pid = os.getpid() if pid is None else int(pid)
        if actual_pid != os.getpid():
            raise PermissionError("writer PID must be the current executor process")
        if not signature or not isinstance(signature, str):
            raise TabuEntryError("environment signature must be a non-empty string")
        capability = WriterCapability(actual_pid, signature, secrets.token_urlsafe(32))
        self._writer = capability
        return capability

    def read(self, filter: Callable[[dict[str, Any]], bool] | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TabuEntryError(f"invalid JSON at line {line_number}") from exc
                self._validate_entry(value)
                if filter is None or filter(value):
                    entries.append(value)
        return entries

    def write(self, entry: dict[str, Any], *, capability: WriterCapability | None = None) -> dict[str, Any]:
        if capability is None or self._writer is None:
            raise PermissionError("tabu writes require an executor capability")
        if capability != self._writer or capability.pid != os.getpid():
            raise PermissionError("invalid or stale executor capability")
        self._validate_entry(entry)
        if entry["source"] != "env_executor":
            raise TabuEntryError("tabu source must be env_executor")
        if entry.get("env_signature") != capability.signature:
            raise TabuEntryError("environment signature mismatch")
        payload = dict(entry)
        payload.pop("env_signature", None)
        payload["entry_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        return payload

    @staticmethod
    def _validate_entry(entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            raise TabuEntryError("tabu entry must be an object")
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise TabuEntryError(f"tabu entry missing required fields: {', '.join(missing)}")
        if entry["source"] == "model":
            raise PermissionError("model-originated tabu entries are forbidden")
        if entry["source"] != "env_executor":
            raise TabuEntryError("tabu source must be env_executor")
        if not isinstance(entry["game_id"], str) or not entry["game_id"]:
            raise TabuEntryError("game_id must be non-empty")
        if not isinstance(entry["level"], int) or isinstance(entry["level"], bool) or entry["level"] < 0:
            raise TabuEntryError("level must be a non-negative integer")
        if not isinstance(entry["action_sequence"], list):
            raise TabuEntryError("action_sequence must be a list")
        if not isinstance(entry["frame_hash_before"], str) or not entry["frame_hash_before"]:
            raise TabuEntryError("frame_hash_before must be non-empty")
        if not isinstance(entry["frame_hash_after"], str) or not entry["frame_hash_after"]:
            raise TabuEntryError("frame_hash_after must be non-empty")
        try:
            json.dumps(entry, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TabuEntryError("tabu entry must be JSON-safe") from exc
