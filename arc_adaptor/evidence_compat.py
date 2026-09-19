"""lingjing-evidence-v1 的本地兼容层。

优先使用 feat/ar25-minimal-closed-loop 带来的 `lingjing_solo.evidence.protocol`；
它还没合进来时，用这里逐字段等价的实现 —— 字段名、必填校验、回放语义都照该分支
`lingjing_solo/evidence/protocol.py` 抄（build_manifest/build_tick/
build_verification_report/validate_*/replay_recording）。

合并之后本模块会自动走到 protocol 那份，跑测脚本一行都不用改。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:                                        # 合并后走这里
    from lingjing_solo.evidence.protocol import (  # noqa: F401
        EvidenceValidationError, ReplayResult, SCHEMA_VERSION, build_manifest,
        build_tick, build_verification_report, replay_recording,
        validate_manifest, validate_tick, validate_verification_report,
    )
    BACKEND = "lingjing_solo.evidence.protocol"
except ImportError:                         # 合并前走这里（等价实现）
    BACKEND = "arc_adaptor.evidence_compat(fallback)"

    SCHEMA_VERSION = "lingjing-evidence-v1"

    class EvidenceValidationError(ValueError):
        """证据不合法或不可回放时抛出（fail-closed）。"""

    @dataclass(frozen=True)
    class ReplayResult:
        transitions: tuple
        final_state: str | None
        levels_completed: int
        reset_count: int

    def _require(mapping: dict, fields: Iterable[str], label: str) -> None:
        missing = [f for f in fields if f not in mapping]
        if missing:
            raise EvidenceValidationError(f"{label} missing required fields: {', '.join(missing)}")

    def _json_safe(value: Any, label: str) -> Any:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise EvidenceValidationError(f"{label} must be JSON-safe") from exc
        return value

    def build_manifest(*, run_id: str, game_id: str, branch: str, commit: str,
                       module_versions: dict, evidence_tier: str, mode: str,
                       game_specific: dict | None = None, seed: Any = None,
                       limits: dict | None = None,
                       source_recording: str | None = None) -> dict:
        return validate_manifest({
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "game_id": game_id,
            "agent_build": f"git:{commit}", "project_branch": branch,
            "module_versions": dict(module_versions), "evidence_tier": evidence_tier,
            "mode": mode, "seed": seed,
            "limits": limits or {"max_steps": 64, "max_nodes": 128000, "timeout_s": 30},
            "source_recording": source_recording, "status": "running", "artifacts": [],
            "game_specific": game_specific or {},
        })

    def build_tick(*, run_id: str, episode_id: str, tick: int, frame: Any, state: str,
                   levels_completed: int, legal_actions: list, state_hash: str,
                   requested_action: dict | None = None, settled_frame: bool = True,
                   score: float | None = None, plan_id: str | None = None,
                   decision_id: str | None = None, reflection_id: str | None = None,
                   evidence_refs: list | None = None,
                   game_specific: dict | None = None) -> dict:
        return validate_tick({
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "episode_id": episode_id,
            "tick": tick, "frame": frame, "requested_action": requested_action,
            "settled_frame": settled_frame, "state": state,
            "levels_completed": levels_completed, "score": score,
            "legal_actions": legal_actions, "state_hash": state_hash, "plan_id": plan_id,
            "decision_id": decision_id, "reflection_id": reflection_id,
            "evidence_refs": evidence_refs or [], "game_specific": game_specific or {},
        })

    def build_verification_report(*, run_id: str, tier: str, verdict: str,
                                  criteria: dict, metrics: dict, evidence_refs: list,
                                  limitations: list | None = None,
                                  game_specific: dict | None = None) -> dict:
        return validate_verification_report({
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "verdict": verdict,
            "tier": tier, "criteria": criteria, "metrics": metrics,
            "evidence_refs": evidence_refs, "limitations": limitations or [],
            "game_specific": game_specific or {},
        })

    def validate_manifest(value: dict) -> dict:
        if not isinstance(value, dict):
            raise EvidenceValidationError("manifest must be an object")
        _require(value, ("schema_version", "run_id", "game_id", "agent_build", "project_branch",
                         "module_versions", "evidence_tier", "mode", "limits", "status",
                         "artifacts", "game_specific"), "manifest")
        if value["schema_version"] != SCHEMA_VERSION:
            raise EvidenceValidationError("unsupported schema_version")
        if not all(value.get(k) for k in ("run_id", "game_id", "agent_build", "project_branch")):
            raise EvidenceValidationError("manifest identity fields must be non-empty")
        if not isinstance(value["module_versions"], dict) or not isinstance(value["game_specific"], dict):
            raise EvidenceValidationError("manifest module_versions/game_specific must be objects")
        _json_safe(value, "manifest")
        return value

    def validate_tick(value: dict) -> dict:
        if not isinstance(value, dict):
            raise EvidenceValidationError("tick must be an object")
        _require(value, ("schema_version", "run_id", "episode_id", "tick", "frame",
                         "requested_action", "settled_frame", "state", "levels_completed",
                         "score", "legal_actions", "state_hash", "evidence_refs",
                         "game_specific"), "tick")
        if value["schema_version"] != SCHEMA_VERSION:
            raise EvidenceValidationError("unsupported schema_version")
        if not isinstance(value["tick"], int) or value["tick"] < 0:
            raise EvidenceValidationError("tick must be a non-negative integer")
        if not isinstance(value["legal_actions"], list) or \
                not all(isinstance(x, str) and x for x in value["legal_actions"]):
            raise EvidenceValidationError("legal_actions must be a list of non-empty strings")
        if not isinstance(value["game_specific"], dict) or not isinstance(value["evidence_refs"], list):
            raise EvidenceValidationError("tick extensions/evidence_refs have invalid types")
        _json_safe(value, "tick")
        return value

    def validate_verification_report(value: dict) -> dict:
        if not isinstance(value, dict):
            raise EvidenceValidationError("verification report must be an object")
        _require(value, ("schema_version", "run_id", "verdict", "tier", "criteria", "metrics",
                         "evidence_refs", "limitations", "game_specific"), "verification report")
        if value["schema_version"] != SCHEMA_VERSION or \
                value["verdict"] not in {"PASS", "FAIL", "BLOCKED"}:
            raise EvidenceValidationError("invalid report schema_version or verdict")
        if not isinstance(value["criteria"], dict) or not isinstance(value["metrics"], dict):
            raise EvidenceValidationError("report criteria/metrics must be objects")
        if not isinstance(value["evidence_refs"], list) or not isinstance(value["limitations"], list):
            raise EvidenceValidationError("report evidence_refs/limitations must be lists")
        _json_safe(value, "verification report")
        return value

    def _frame_hash(frame: Any) -> str:
        encoded = json.dumps(frame, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _action(value: Any) -> dict:
        if isinstance(value, str):
            return {"name": value}
        if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"]:
            raise EvidenceValidationError("requested_action must contain a non-empty name")
        payload = value.get("payload", {})
        if not isinstance(payload, dict):
            raise EvidenceValidationError("action payload must be an object")
        for axis in ("x", "y"):
            if axis in payload and (not isinstance(payload[axis], int)
                                    or not 0 <= payload[axis] <= 63):
                raise EvidenceValidationError(f"payload {axis} must be an integer in 0..63")
        _json_safe(value, "action")
        return value

    def replay_recording(path: str | Path, *, legal_actions: list) -> ReplayResult:
        """把 JSONL 逐行读成可审计的转移序列；不合法就抛错，不改源文件。"""
        if not legal_actions:
            raise EvidenceValidationError("legal_actions must not be empty")
        previous: dict | None = None
        transitions: list[dict] = []
        reset_count = 0
        final_state: str | None = None
        levels_completed = 0
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvidenceValidationError(f"invalid JSON at line {line_number}") from exc
                data = record.get("data", record)
                if not isinstance(data, dict) or "frame" not in data:
                    continue
                action_value = data.get("requested_action")
                state = str(data.get("state") or "")
                final_state = state or final_state
                levels_completed = int(data.get("levels_completed") or 0)
                if previous is None:
                    if action_value is not None and _action(action_value)["name"].upper() != "RESET":
                        raise EvidenceValidationError("recording has no baseline")
                    previous = data
                    continue
                action = _action(action_value)
                if action["name"].upper() == "RESET" or state.upper() == "RESET":
                    previous = data
                    reset_count += 1
                    continue
                if action["name"] not in legal_actions:
                    raise EvidenceValidationError(f"illegal action: {action['name']}")
                transitions.append({
                    "before_frame": previous["frame"], "action": action,
                    "after_frame": data["frame"], "before_hash": _frame_hash(previous["frame"]),
                    "after_hash": _frame_hash(data["frame"]), "state": state,
                    "levels_completed": levels_completed,
                })
                previous = data
        if previous is None:
            raise EvidenceValidationError("recording has no baseline")
        if not transitions and reset_count == 0:
            raise EvidenceValidationError("recording has no action transition")
        return ReplayResult(tuple(transitions), final_state, levels_completed, reset_count)
