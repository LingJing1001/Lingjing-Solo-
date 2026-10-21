"""Poll Duck kernel and competition-submit when COMPLETE."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "ARC-AGI-3-Kaggle-Starter"
KAGGLE = ROOT / ".venv" / "Scripts" / "kaggle.exe"
KERNEL = "liaojiaxuan/lingjing-duck-harness-v1"
MSG = "duck franzen qwen3.8 conc8 ctx32768 think-off"


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, ((p.stdout or "") + "\n" + (p.stderr or "")).strip()


def status() -> str:
    last = ""
    for _ in range(6):
        _, text = _run([str(KAGGLE), "kernels", "status", KERNEL])
        last = " ".join(text.split())
        if "KernelWorkerStatus." in last and "Traceback" not in last:
            return last
        time.sleep(5)
    return last


def latest_version() -> str | None:
    for _ in range(5):
        code, text = _run([str(KAGGLE), "kernels", "status", KERNEL])
        # status alone has no version; use kernels list / push output. Caller
        # can pass VERSION env; we also try metadata via API-less fallback.
        _ = code, text
        break
    return None


def main() -> int:
    version = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    if not version:
        print("USAGE: _wait_submit_duck.py <versionNumber>", flush=True)
        return 2

    for i in range(1, 301):
        line = status()
        short = line
        if "KernelWorkerStatus." in line:
            short = line[line.find("KernelWorkerStatus.") :][:80]
        print(f"[{i}] {short}", flush=True)
        if "KernelWorkerStatus.COMPLETE" in line:
            for attempt in range(1, 8):
                print("SUBMIT_TRY", attempt, flush=True)
                code, out = _run(
                    [
                        str(KAGGLE),
                        "competitions",
                        "submit",
                        "arc-prize-2026-arc-agi-3",
                        "-k",
                        KERNEL,
                        "-v",
                        version,
                        "-f",
                        "submission.parquet",
                        "-m",
                        MSG,
                    ]
                )
                print("SUBMIT_RC", code, flush=True)
                print(out[:2000], flush=True)
                low = out.lower()
                if "0 submissions remaining" in low and "error" not in low:
                    # Last slot often returns this after a successful enqueue.
                    print("SUBMIT_OK_QUOTA_NOTE", flush=True)
                    return 0
                if code == 0 and (
                    "successfully" in low
                    or "submitted" in low
                    or "your submission" in low
                    or "has been submitted" in low
                ):
                    print("SUBMIT_OK", flush=True)
                    return 0
                if "10054" in out or "connection" in low:
                    time.sleep(6)
                    continue
                return code or 1
            return 1
        if "KernelWorkerStatus.ERROR" in line or "KernelWorkerStatus.CANCEL" in line:
            print("KERNEL_NOT_COMPLETE", short, flush=True)
            return 1
        time.sleep(45)
    print("TIMEOUT", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
