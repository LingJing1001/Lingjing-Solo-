"""AR25 candidate -> offline simulator -> unified evidence artifacts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .protocol import build_manifest, build_tick, build_verification_report, replay_recording

_ACTION_NAMES = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}
_LEGAL_ACTIONS = list(_ACTION_NAMES.values())


def _git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def run_ar25_level(*, level: int, output_dir: str | Path, strategy: str = "auto",
                   t_limit: int = 30) -> dict[str, Any]:
    """Solve one AR25 level, replay it in the offline simulator, and write evidence."""
    from arc_adaptor.ar25_solver import AR25Simulator, __version__, load_levels, solve

    root = Path(__file__).resolve().parents[2]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"ar25-l{level}-offline"
    recording = output / f"{run_id}.jsonl"
    report_path = output / f"{run_id}.report.json"
    levels = load_levels()
    level_data = levels[level - 1]
    actions = solve(level_data, strategy=strategy, t_limit=t_limit, verbose=False)
    if not actions:
        raise RuntimeError(f"AR25 level {level} produced no candidate")

    simulator = AR25Simulator(level_data)
    records: list[dict[str, Any]] = []
    frame = simulator.snapshot()
    records.append(build_tick(
        run_id=run_id, episode_id=f"level-{level}", tick=0, frame=frame,
        requested_action={"name": "RESET"}, settled_frame=True, state="RESET",
        levels_completed=0, legal_actions=_LEGAL_ACTIONS,
        state_hash="offline-initial", game_specific={"level": level},
    ))
    for tick, action_id in enumerate(actions, 1):
        if action_id not in _ACTION_NAMES:
            raise ValueError(f"unsupported AR25 action id: {action_id}")
        simulator.step(action_id)
        state = "WON" if simulator.won else "LOST" if simulator.lost else "RUNNING"
        records.append(build_tick(
            run_id=run_id, episode_id=f"level-{level}", tick=tick,
            frame=simulator.snapshot(), requested_action={"name": _ACTION_NAMES[action_id], "payload": {"id": action_id}},
            settled_frame=True, state=state, levels_completed=1 if simulator.won else 0,
            legal_actions=_LEGAL_ACTIONS, state_hash=f"offline-{tick}",
            game_specific={"level": level, "action_id": action_id},
        ))
    with recording.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    replay = replay_recording(recording, legal_actions=_LEGAL_ACTIONS)
    manifest = build_manifest(
        run_id=run_id, game_id="AR25", branch=_git_value(root, "branch", "--show-current"),
        commit=_git_value(root, "rev-parse", "HEAD"),
        module_versions={"ar25_solver": __version__, "evidence": "lingjing-evidence-v1"},
        evidence_tier="offline", mode="candidate-replay", seed=None,
        limits={"max_steps": level_data["budget"], "timeout_s": t_limit},
        source_recording=str(recording), game_specific={"level": level, "strategy": strategy},
    )
    report = build_verification_report(
        run_id=run_id, tier="offline", verdict="PASS" if simulator.won else "FAIL",
        criteria={"candidate_present": True, "offline_replay": True, "won": simulator.won},
        metrics={"candidate_steps": len(actions), "replayed_transitions": len(replay.transitions),
                 "covered": len(simulator.all_covered_cells() & simulator.targets), "total_targets": len(simulator.targets)},
        evidence_refs=[str(recording)], limitations=["offline simulator; not live Scorecard"],
        game_specific={"level": level, "strategy": strategy},
    )
    report_path.write_text(json.dumps({"manifest": manifest, "report": report}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"recording": str(recording), "report": str(report_path), "manifest": manifest, "report_data": report}
