"""Convert the verified LS20 recording into lingjing-evidence-v1."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..core import Frame, SoloConfig
from ..exploration.action_diff import _normalize_grid, analyze_recording
from ..world_model.field import WorldModelField
from .protocol import build_manifest, build_verification_report


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def convert_ls20_recording(recording_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Replay the known 309-record LS20 artifact through ActionDiff and Field."""
    recording = Path(recording_path).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    deltas = analyze_recording(recording, player_color=1)
    records: list[dict[str, Any]] = []
    with recording.open(encoding="utf-8") as handle:
        records = [json.loads(line).get("data", {}) for line in handle]
    frames = [row for row in records if "frame" in row]
    if not frames:
        raise ValueError("LS20 recording has no frames")

    field = WorldModelField(SoloConfig(grid_size=64))
    previous: Frame | None = None
    for tick, row in enumerate(frames):
        grid = _normalize_grid(np.asarray(row["frame"]))
        current = Frame(grid=grid, t=tick, state=str(row.get("state") or ""),
                        levels_completed=int(row.get("levels_completed") or 0))
        action = row.get("requested_action")
        if isinstance(action, dict):
            action = action.get("name") or action.get("id")
        field.update(current, previous, action=action if previous is not None else None)
        previous = current

    final = frames[-1]
    state = str(final.get("state") or "").upper()
    levels = int(final.get("levels_completed") or 0)
    report = build_verification_report(
        run_id="ls20-9607627b-real-recording",
        tier="recorded-offline",
        verdict="PASS" if state == "WIN" and len(deltas) == 308 and levels == 7 else "FAIL",
        criteria={"recording_present": True, "action_diff_replay": True,
                  "field_replay": True, "terminal_win": state == "WIN"},
        metrics={"records": len(records), "transitions": len(deltas),
                 "field_transitions": len(field.transition_table),
                 "levels_completed": field.levels, "field_version": field.version,
                 "win_hash_count": len(field.win_hashes)},
        evidence_refs=[str(recording)],
        limitations=["recorded offline artifact; not live Scorecard"],
        game_specific={"game": "LS20", "frame_shapes": "1/2/6/17 x 64 x 64",
                       "source_recording_commit": _git(recording.parents[1], "rev-parse", "HEAD")},
    )
    manifest = build_manifest(
        run_id="ls20-9607627b-real-recording", game_id="LS20",
        branch=_git(Path(__file__).resolve().parents[2], "branch", "--show-current"),
        commit=_git(Path(__file__).resolve().parents[2], "rev-parse", "HEAD"),
        module_versions={"action_diff": "r4", "field": "WorldModelField",
                         "evidence": "lingjing-evidence-v1"},
        evidence_tier="recorded-offline", mode="recording-replay", seed=None,
        limits={"records": len(records), "expected_transitions": 308},
        source_recording=str(recording),
        game_specific={"arc_commit": manifest_arc_commit(recording)},
    )
    artifact = {"manifest": manifest, "report": report}
    path = output / "ls20-9607627b.verification.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "manifest": manifest, "report": report}


def manifest_arc_commit(recording: Path) -> str:
    return _git(recording.parents[1], "rev-parse", "HEAD")
