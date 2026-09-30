"""Compare Nova-ARC vs Aether-Prime on sp80/ls20 long runs (offline).

Streams child output live (no capture hang).
Usage:
  .venv/Scripts/python.exe scripts/compare_longrun.py --steps 1000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PLAY = ROOT / "scripts" / "play_local.py"


def run(label: str, agent: Path, games: str, steps: int, log: Path) -> None:
    env = os.environ.copy()
    env["AETHER_LLM"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        PY,
        "-u",
        str(PLAY),
        "--agent",
        str(agent),
        "--game",
        games,
        "--max-steps",
        str(steps),
    ]
    header = f"\n######## {label} ########\n{' '.join(cmd)}\n"
    print(header, flush=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(header)
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        rc = proc.wait()
        footer = f"[exit {rc}] {label}\n"
        print(footer, flush=True)
        fh.write(footer)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--games", default="sp80,ls20")
    args = p.parse_args()
    nova = ROOT / "agent" / "my_agent.py"
    prime = ROOT / "agent" / "baselines" / "aether_prime.py"
    if not prime.exists():
        raise SystemExit(f"Missing baseline {prime}")
    log = ROOT / "eval_compare.txt"
    log.write_text("", encoding="utf-8")
    run("Nova-ARC", nova, args.games, args.steps, log)
    run("Aether-Prime", prime, args.games, args.steps, log)
    print(f"\nWrote {log}", flush=True)


if __name__ == "__main__":
    main()
