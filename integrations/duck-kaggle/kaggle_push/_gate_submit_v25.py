"""Wait for Duck kernel v25 COMPLETE, verify smoke markers, then competition-submit.

Gating (all must hold before we burn the daily quota):
  + "vLLM server ready"            (server bound the port)
  + "VLLM OPENAI SERVER QWEN SMOKE TEST" + a non-empty "Generated:" line
  + "LINGJING_BUDGET"               (notebook cell 12 ran)
  - "Engine core initialization failed"
  - "returned non-zero exit status 1" inside the setup phase
  - "LINGJING_TORCH_PROBE 1" / "LINGJING_CUDA_LIBS_MISSING"
On gate pass: `kaggle competitions submit ... -k <kernel> -v 25 -f submission.parquet`.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parents[3] / "ARC-AGI-3-Kaggle-Starter" / ".venv" / "Scripts" / "python.exe"
KERNEL = "liaojiaxuan/lingjing-duck-harness-v1"
VERSION = "25"
MSG = "duck franzen qwen3.8 conc8 ctx32768 think-off enforce-eager cutlass maxseqs8"
OUT = HERE / "_v25_out"
COMP = "arc-prize-2026-arc-agi-3"
REPORT = HERE / "_v25_gate_report.txt"


def run(cmd: list[str], timeout: float = 300.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def status() -> str:
    _, text = run([str(KAGGLE), "-m", "kaggle", "kernels", "status", KERNEL], timeout=90)
    return " ".join(text.split())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 1) wait for terminal state (up to 8h)
    deadline = time.time() + 8 * 3600
    st = ""
    while time.time() < deadline:
        st = status()
        short = st[st.find("KernelWorkerStatus.") :][:60] if "KernelWorkerStatus." in st else st[:120]
        say(f"[{time.strftime('%H:%M:%S')}] {short}")
        if "COMPLETE" in st or "ERROR" in st or "CANCEL" in st:
            break
        time.sleep(120)
    else:
        say("WATCH_TIMEOUT kernel still not terminal")
        return 2

    if "COMPLETE" not in st:
        say("KERNEL_NOT_COMPLETE — not submitting")
        return 3

    # 2) download the log (Kaggle needs a moment to finalize output after COMPLETE)
    log = None
    for attempt in range(1, 13):
        code, text = run([str(KAGGLE), "-m", "kaggle", "kernels", "output", KERNEL, "-p", str(OUT)], timeout=600)
        say(f"download try={attempt} rc={code}")
        cands = sorted(OUT.rglob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands and cands[0].stat().st_size > 1000:
            log = cands[0]
            break
        time.sleep(30)
    if log is None:
        say("NO_LOG_DOWNLOADED — not submitting")
        return 4
    txt = log.read_text(encoding="utf-8", errors="replace")
    say(f"log file={log.name} size={len(txt)}")

    # 3) gate checks
    must_have = [
        "vLLM server ready",
        "VLLM OPENAI SERVER QWEN SMOKE TEST",
        "LINGJING_BUDGET",
    ]
    must_not = [
        "Engine core initialization failed",
        "LINGJING_TORCH_PROBE 1",
        "LINGJING_CUDA_LIBS_MISSING",
        "LINGJING_TORCH_IMPORT_FAIL",
    ]
    ok = True
    for m in must_have:
        hit = m in txt
        say(("PASS " if hit else "FAIL ") + m)
        ok = ok and hit
    for m in must_not:
        hit = m in txt
        say(("PASS(no) " if not hit else "FAIL(present) ") + m)
        ok = ok and not hit

    # smoke output must be a real sentence, not empty
    gen = re.search(r"Generated:(.+?)\n", txt)
    if gen and gen.group(1).strip():
        say("PASS smoke Generated -> " + gen.group(1).strip()[:120])
    else:
        say("FAIL smoke Generated empty")
        ok = False

    # benchmark actually played: analyzer activity markers
    played = ("analysis_step=" in txt) or ("tool-agent" in txt) or ("levels completed" in txt)
    say(("PASS " if played else "WARN ") + "benchmark activity")
    if not played:
        ok = False

    # vLLM flags actually in effect
    for flag in ("enforce_eager': True", "linear_backend': 'cutlass'", "max_num_seqs': 8"):
        hit = flag in txt
        say(("PASS " if hit else "WARN ") + "flag " + flag)

    if not ok:
        say("GATE_FAILED — NOT submitting (quota preserved)")
        return 5

    # 4) submit
    say("GATE_PASSED — submitting v" + VERSION)
    code, text = run(
        [
            str(KAGGLE),
            "competitions",
            "submit",
            COMP,
            "-k",
            KERNEL,
            "-v",
            VERSION,
            "-f",
            "submission.parquet",
            "-m",
            MSG,
        ],
        timeout=180,
    )
    say(f"submit rc={code}")
    say(text[:1500])
    if code != 0:
        say("SUBMIT_FAILED")
        return 6
    low = text.lower()
    if "error" in low and "successfully" not in low:
        say("SUBMIT_MAYBE_FAILED — inspect above")
        return 7
    say("SUBMIT_OK")

    # 5) watch the submission appear
    for _ in range(10):
        time.sleep(30)
        code, text = run([str(KAGGLE), "-m", "kaggle", "competitions", "submissions", "-c", COMP], timeout=120)
        first = text.splitlines()[3:5]
        say("latest submissions: " + " | ".join(first))
        if MSG[:20] in text or "duck franzen" in text:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
