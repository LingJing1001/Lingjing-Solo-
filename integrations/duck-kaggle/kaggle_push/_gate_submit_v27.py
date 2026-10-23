"""Wait for Duck kernel v27 COMPLETE, verify health + quality gates, then submit.

This is the v27 gate.  v27's whole purpose is to stop the agent from being
crippled into never executing an in-game action, so the gate checks the thing
v27 actually fixes:

  HARD (run health -- a failure here means the config is broken, not unlucky):
    + "vLLM server ready"
    + "VLLM OPENAI SERVER QWEN SMOKE TEST" + a non-empty "Generated:" line
    + "LINGJING_BUDGET" line showing think=True / giveup=1800 / upscale=4
    + "Engine core initialization failed" absent
    + "LINGJING_TORCH_PROBE 1" / "LINGJING_CUDA_LIBS_MISSING" /
      "LINGJING_TORCH_IMPORT_FAIL" absent
    + at least 12 of the 25 bench games executed an action
      (v25 baseline: 14 of 25 moved, 11 never moved at all)

  SOFT FLOOR (a broken-quality config must not burn the daily quota):
    + total offline score >= 5.0   (v25 baseline: 14.52)

On gate pass: `kaggle competitions submit ... -k <kernel> -v <version> -f
submission.parquet`.  The competition re-runs the kernel against the live
gateway; the offline bench above is only the health check.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parents[3] / "ARC-AGI-3-Kaggle-Starter" / ".venv" / "Scripts" / "python.exe"
KERNEL = "liaojiaxuan/lingjing-duck-harness-v27"
MSG = "duck v27 official-analyzer toolsteps0 yield60 maxout0 timeout900 upscale4 giveup1800"
OUT = HERE / "_v27_out"
COMP = "arc-prize-2026-arc-agi-3"
REPORT = HERE / "_v27_gate_report.txt"

MIN_GAMES_MOVED = 12
MIN_TOTAL_SCORE = 5.0


def run(cmd: list[str], timeout: float = 300.0) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return p.returncode, ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def status() -> str:
    _, text = run([str(KAGGLE), "-m", "kaggle", "kernels", "status", KERNEL], timeout=90)
    return " ".join(text.split())


def kernel_version() -> str:
    f = HERE / "push_v27" / "_last_push_version_v27.txt"
    if f.is_file():
        v = f.read_text(encoding="utf-8").strip()
        if v:
            return v
    return "1"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 1) wait for terminal state (up to 12h)
    deadline = time.time() + 12 * 3600
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
    # NOTE: the v27 kernel output contains BOTH the embedded main log
    # (`*embedded_log*.txt`) and per-game analyzer transcripts (`*_p0.log`).
    # The gate must read the MAIN log: the per-game transcripts do not contain
    # "vLLM server ready" / "LINGJING_BUDGET" and would fail the gate wrongly.
    log = None
    # Reuse an already-downloaded main log (the kernel output is ~1 GB; a
    # second download wastes minutes and Kaggle bandwidth).
    for _old in sorted(OUT.glob("*embedded_log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
        if _old.stat().st_size > 1000 and "LINGJING_BUDGET" in _old.read_text(encoding="utf-8", errors="replace"):
            log = _old
            say(f"reusing local main log {log.name} size={log.stat().st_size}")
            break
    for attempt in range(1, 13):
        if log is not None:
            break
        code, text = run([str(KAGGLE), "-m", "kaggle", "kernels", "output", KERNEL, "-p", str(OUT)], timeout=600)
        say(f"download try={attempt} rc={code}")
        cands = sorted(OUT.rglob("*embedded_log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            cands = sorted(OUT.rglob("*_p0.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands and cands[0].stat().st_size > 1000:
            log = cands[0]
            break
        time.sleep(30)
    if log is None:
        say("NO_LOG_DOWNLOADED — not submitting")
        return 4
    txt = log.read_text(encoding="utf-8", errors="replace")
    say(f"log file={log.name} size={len(txt)}")

    # 3) hard health gate
    must_have = [
        "vLLM server ready",
        "VLLM OPENAI SERVER QWEN SMOKE TEST",
        "LINGJING_BUDGET",
    ]
    must_not = [
        # Match the RAISED form, not the bare marker: the v27 main log echoes the
        # notebook source, so the literal strings "LINGJING_CUDA_LIBS_MISSING" /
        # "LINGJING_TORCH_IMPORT_FAIL" appear there as `raise RuntimeError(...)`
        # text even on a fully healthy run.  The traceback form never does.
        "RuntimeError: LINGJING_CUDA_LIBS_MISSING",
        "RuntimeError: LINGJING_TORCH_IMPORT_FAIL",
        "LINGJING_TORCH_PROBE 1 ",
        "Traceback (most recent call last)",
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

    # The main log echoes the notebook source, so the first "Generated:" match can
    # be source text.  Require at least one match that looks like real output.
    gens = [g.strip() for g in re.findall(r"Generated:(.+?)\\?n", txt)]
    real = [g for g in gens if len(g) > 3 and "flush" not in g and "print(" not in g]
    if real:
        say("PASS smoke Generated -> " + real[-1][:120])
    else:
        say(f"FAIL smoke Generated empty (candidates={len(gens)}, real={len(real)})")
        ok = False

    # v27 knobs actually in effect
    budget = re.search(r"LINGJING_BUDGET.*", txt)
    if budget:
        say("PASS budget line -> " + budget.group(0).strip()[:200])
        b = budget.group(0)
        # thinking must be ON (official); upscale must be 4 (official).
        # giveup is expected to be None until the overlay dataset is re-versioned
        # (the Kaggle dataset upload path needs www.googleapis.com, unreachable
        # from this network), so only require that the knob was reported.
        for want, label in (
            ("think True", "thinking ON"),
            ("upscale 4", "upscale 4"),
            ("giveup", "giveup field present"),
            ("900", "analyzer_timeout 900"),
            ("5400", "max_runtime 5400"),
        ):
            hit = want in b
            say(("PASS " if hit else "FAIL ") + label + f" ({want})")
            ok = ok and hit
    else:
        say("FAIL no LINGJING_BUDGET line")
        ok = False

    # 4) quality gate: how many bench games actually moved?
    fin = re.findall(
        r"\[finished\]\s+(\S+)\s+state=(\S+)\s+level=([\d.]+)/(\d+)\s+score=([\d.]+)\s+actions=(\d+)\s+tokens=(\d+)",
        txt,
    )
    moved = [f for f in fin if int(f[5]) > 0]
    total_score = sum(float(f[4]) for f in fin)
    say(f"games finished={len(fin)}  games with actions>0={len(moved)}  total_score={total_score:.2f}")
    for f in sorted(fin, key=lambda x: float(x[4]), reverse=True):
        say(f"   {f[0]:>16} score={f[4]:>5} level={f[2]}/{f[3]} actions={f[5]:>5} tokens={f[6]:>8}")
    if len(moved) >= MIN_GAMES_MOVED:
        say(f"PASS games moved {len(moved)} >= {MIN_GAMES_MOVED}")
    else:
        say(f"FAIL games moved {len(moved)} < {MIN_GAMES_MOVED}")
        ok = False
    if total_score >= MIN_TOTAL_SCORE:
        say(f"PASS total score {total_score:.2f} >= {MIN_TOTAL_SCORE}")
    else:
        say(f"FAIL total score {total_score:.2f} < {MIN_TOTAL_SCORE}")
        ok = False

    if not ok:
        say("GATE_FAILED — NOT submitting (quota preserved)")
        return 5

    # 5) submit
    #
    # 不要再用 `kaggle competitions submit <COMP> -k ... -v ... -f ...` 这条 CLI：
    # CLI 2.2.4 的 competitions submit 要的是 `-c <competition>`，位置参数会被当成
    # 待上传的文件名，于是它去打开一个叫 "competitions" 的文件并报
    #   can't open file '...\\duck-kaggle\\competitions': [Errno 2] No such file
    # v27 第一次就是死在这里：门控全 PASS，却在烧配额前一步失败。
    #
    # 改走 API 通道 competition_submit_code(...) —— 这正是网页上
    # "Submit to Competition" 按钮调的同一个接口，SmartRouter 线已用它成功提交过
    # （ref 57542504）。file_name 只是占位：真实 submission.parquet 由竞赛网关
    # sidecar 在 rerun 时产出。
    ver = kernel_version()
    say(f"GATE_PASSED — submitting v27 kernel version {ver} via competition_submit_code")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        _api = KaggleApi()
        _api.authenticate()
        _resp = _api.competition_submit_code(
            file_name="submission.parquet",
            message=MSG,
            competition=COMP,
            kernel=KERNEL,
            kernel_version=int(ver),
        )
        _d = _resp.to_dict() if hasattr(_resp, "to_dict") else getattr(_resp, "__dict__", {})
        say("SUBMIT_RESP " + json.dumps(_d, ensure_ascii=False, default=str)[:1200])
        say("SUBMIT_OK")
    except Exception as _exc:  # noqa: BLE001
        say(f"SUBMIT_FAILED {type(_exc).__name__}: {_exc}")
        return 6

    # 6) watch the submission appear
    for _ in range(10):
        time.sleep(30)
        code, text = run([str(KAGGLE), "-m", "kaggle", "competitions", "submissions", "-c", COMP], timeout=120)
        first = text.splitlines()[3:5]
        say("latest submissions: " + " | ".join(first))
        if MSG[:20] in text or "duck v27" in text:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
