"""Poll Duck kernel v25 status; download log snapshots for smoke-test markers."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parents[3] / "ARC-AGI-3-Kaggle-Starter" / ".venv" / "Scripts" / "python.exe"
KERNEL = "liaojiaxuan/lingjing-duck-harness-v1"
OUT = HERE / "_v25_watch"
POLL_LOG = HERE / "_v25_poll.log"

MARKERS = [
    "LINGJING_TORCH_PROBE",
    "LINGJING_SO_LINK",
    "LINGJING_GDN_TRITON",
    "vLLM server ready",
    "VLLM OPENAI SERVER QWEN SMOKE TEST",
    "Generated:",
    "LINGJING_BUDGET",
    "Loading weights took",
    "Engine core initialization failed",
    "Traceback",
]


def run(cmd: list[str], timeout: float = 120.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def status() -> str:
    code, text = run(
        [str(KAGGLE), "-m", "kaggle", "kernels", "status", KERNEL], timeout=90
    )
    return " ".join(text.split())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 6 * 3600  # watch up to 6h
    last_dl = 0.0
    n = 0
    while time.time() < deadline:
        n += 1
        st = status()
        short = st
        if "KernelWorkerStatus." in st:
            short = st[st.find("KernelWorkerStatus.") :][:90]
        line = f"[{n}] {time.strftime('%H:%M:%S')} {short}"
        print(line, flush=True)
        with POLL_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if "COMPLETE" in st or "ERROR" in st or "CANCEL" in st:
            print("TERMINAL", flush=True)
            break
        # download log every ~4 min to inspect markers while running
        if time.time() - last_dl > 240:
            last_dl = time.time()
            code, text = run(
                [str(KAGGLE), "-m", "kaggle", "kernels", "output", KERNEL, "-p", str(OUT)],
                timeout=300,
            )
            print(f"  dl rc={code} {text[:200]}", flush=True)
            log = OUT / "lingjing-duck-harness-v1.log"
            if log.is_file():
                txt = log.read_text(encoding="utf-8", errors="replace")
                hits = [m for m in MARKERS if m in txt]
                print(f"  markers: {hits}", flush=True)
        time.sleep(60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
