"""Watch the Duck v25 competition submission until it scores.

Polls `kaggle competitions submissions` every 10 min. Reports:
  - status transitions (PENDING/RUNNING/COMPLETE/ERROR)
  - publicScore when it lands
On ERROR: downloads the rerun output log for diagnosis.
Usage: python _watch_score_v25.py [expected_ref]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parents[3] / "ARC-AGI-3-Kaggle-Starter" / ".venv" / "Scripts" / "python.exe"
COMP = "arc-prize-2026-arc-agi-3"
OUT = HERE / "_v25_rerun_out"
REPORT = HERE / "_v25_score_report.txt"
MSG_MARK = "duck franzen"


def run(cmd: list[str], timeout: float = 180.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    expected_ref = sys.argv[1] if len(sys.argv) > 1 else ""
    seen_ref = ""
    deadline = time.time() + 14 * 3600  # reruns can take ~9-12h
    last_status = ""
    while time.time() < deadline:
        code, text = run([str(KAGGLE), "-m", "kaggle", "competitions", "submissions", "-c", COMP], timeout=120)
        rows = [l for l in text.splitlines() if MSG_MARK in l.lower()]
        if not rows:
            # fall back: first data row (most recent submission)
            data = [l for l in text.splitlines() if l.strip() and "SubmissionStatus" in l]
            rows = data[:1]
        if rows:
            row = " | ".join(rows[0].split())
            say(f"[{time.strftime('%m-%d %H:%M:%S')}] {row[:220]}")
            low = row.lower()
            ref = row.split()[0] if row.split() else ""
            if expected_ref and ref == expected_ref:
                seen_ref = ref
            status = "COMPLETE" if "submissionstatus.complete" in low else ("ERROR" if "submissionstatus.error" in low else "")
            if status and status != last_status:
                last_status = status
                say("STATUS_CHANGE " + status)
            if status == "COMPLETE":
                say("SCORED — see publicScore above")
                return 0
            if status == "ERROR":
                say("SUBMISSION_ERROR — downloading rerun output for diagnosis")
                run([str(KAGGLE), "-m", "kaggle", "kernels", "output", "liaojiaxuan/lingjing-duck-harness-v1", "-p", str(OUT)], timeout=600)
                for p in sorted(OUT.rglob("*.log")):
                    say(f"rerun log: {p} ({p.stat().st_size} bytes)")
                return 3
        else:
            say(f"[{time.strftime('%m-%d %H:%M:%S')}] (no duck submission row yet)")
        time.sleep(600)
    say("WATCH_TIMEOUT")
    return 2


if __name__ == "__main__":
    sys.exit(main())
