"""Compare Lingjing EtherealRealm-Solo vs Nova on the same games."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PLAY = ROOT / "scripts" / "play_local.py"
GAMES = "ls20,vc33,ft09"
STEPS = 200


def run(agent_rel: str) -> dict[str, tuple[int, int]]:
    cmd = [
        str(PY), str(PLAY),
        "--game", GAMES,
        "--max-steps", str(STEPS),
        "--agent", agent_rel,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    text = out.stdout + out.stderr
    results: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        if "levels=" in line and "actions=" in line and "state=" in line:
            parts = line.split()
            if len(parts) < 3:
                continue
            gid = parts[0]
            try:
                lv = int(parts[1].split("=")[1])
                act = int(parts[2].split("=")[1])
            except (IndexError, ValueError):
                continue
            results[gid] = (lv, act)
    score = 0.0
    for line in text.splitlines():
        if "Aggregate scorecard score:" in line:
            try:
                score = float(line.split(":")[-1].strip())
            except ValueError:
                pass
    results["_score"] = (int(score * 100), 0)  # hack store
    return results


def main():
    agents = [
        ("Lingjing EtherealRealm-Solo", "agent/my_agent.py"),
        ("Nova", "agent/baselines/nova_arc.py"),
    ]
    print(f"Games={GAMES}  max_steps={STEPS}\n")
    all_res = {}
    for name, path in agents:
        print(f"Running {name}...")
        all_res[name] = run(path)
    print("\n======== COMPARE ========")
    print(f"{'game':8} {'LERS L':>8} {'Nova L':>8}")
    for gid in ["ls20", "vc33", "ft09"]:
        z = all_res["Lingjing EtherealRealm-Solo"].get(gid, (0, 0))[0]
        n = all_res["Nova"].get(gid, (0, 0))[0]
        print(f"{gid:8} {z:8} {n:8}")
    zs = all_res["Lingjing EtherealRealm-Solo"].get("_score", (0, 0))[0] / 100
    ns = all_res["Nova"].get("_score", (0, 0))[0] / 100
    print(f"\nAggregate: LERS={zs}  Nova={ns}")


if __name__ == "__main__":
    main()
