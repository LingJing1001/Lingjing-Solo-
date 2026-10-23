"""Duck harness 竞赛提交的**通用**门控 + 提交脚本（取代 _gate_submit_v27.py）。

为什么要有这个文件
------------------
v27 的门控脚本 `_gate_submit_v27.py` 把阈值写死、只认 v27 的 kernel slug，
而且**用 CLI 提交**（`kaggle competitions submit <COMP> ...`）——CLI 2.2.4 的
competitions submit 要的是 `-c <competition>`，位置参数会被当成文件名，于是它去
打开一个叫 "competitions" 的文件：

    can't open file '...\\duck-kaggle\\competitions': [Errno 2] No such file

⇒ v27 第一次门控 GATE_PASSED 却在烧配额前一步失败。这里统一走
`competition_submit_code(...)`（网页 "Submit to Competition" 按钮同款接口，
SmartRouter 线已用它成功提交过，ref 57542504）。

阈值是怎么定的（别再用 40.0 当门控！）
--------------------------------------
官方基线是 **1.6002 / 局**（官方博客原文 "mean score across all public games is
1.6002 +/- 0.4475"，25 局 × 20 passes）⇒ 折算成 25 局总分是 **40.0**。
但 40.0 是**能力上限口径**，不是**可用性口径**：

    v27 实测（Kaggle GPU 真跑 25 局）：总分 18.61，有动作局 19/25，0.744/局
    v25 实测（同一口径，agent 被四处旋钮掐死）：总分 14.52，有动作局 14/25
    官方基线：总分 40.0，1.6002/局

如果把门控下限设成 40.0，**v27 也一起被挡住**——而日提交配额只有 1 次，
拿一个已知能跑、bench 分 18.61 的版本去换 0 分，对「冲榜」是纯负收益。
所以门控按**实测基线**标定：

    FULL  bench（每局 5400/7920 s）：有动作局 ≥ 15，总分 ≥ 12.0
    SHORT bench（每局 900 s）    ：有动作局 ≥ 8， 总分 ≥ 4.0

SHORT 档的数值更松，是因为 900 s/局的短 bench 与 5400 s/局不可比
（同样的 25 局，v27 在 5400 s 下拿 18.61，在 900 s 下必然明显更低）。
**绝对分数只用来挡「agent 一个动作都不出」这种硬坏态；真正的闸门是健康检查。**
两种 bench 的绝对分不同，所以脚本会从日志里读 `SHORT_BENCH per_game_s N` 自动分档。

硬检查（与 bench 模式无关，全部来自官方 Evaluation 页口径）
----------------------------------------------------------
官方评分：**总分 = 110 个隐藏局各自局分（0~100）的平均**，每关 `min(human/agent,1)²`。
⇒ **没玩到的局直接算 0**，覆盖率是分数的直接乘数，这也是 v28 存在的全部理由。

    + kernel COMPLETE
    + "vLLM server ready"
    + "VLLM OPENAI SERVER QWEN SMOKE TEST" 且有非空 "Generated:"
    + "LINGJING_BUDGET" 且 think=True / upscale=4 / analyzer_timeout=900
    + 无 RuntimeError: LINGJING_CUDA_LIBS_MISSING / LINGJING_TORCH_IMPORT_FAIL
    + 无 "LINGJING_TORCH_PROBE 1 " / 无 "Traceback (most recent call last)"
    + bench 跑完（finished == 25）
    + 有动作局数达标、总分达标
    + （可选 --require）v28 专属：LINGJING_MAX_NUM_SEQS ['16']、solver.py patched

用法（starter venv python，有 kaggle 包）:
    python _gate_submit_duck.py --kernel liaojiaxuan/lingjing-duck-harness-v27 \
        --version 1 --submit --watch-minutes 180
    python _gate_submit_duck.py --kernel liaojiaxuan/lingjing-duck-harness-v28 \
        --version 1 --no-submit --require "LINGJING_MAX_NUM_SEQS" \
        --require "solver.py patched to v28"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parents[3] / "ARC-AGI-3-Kaggle-Starter" / ".venv" / "Scripts" / "python.exe"
COMP = "arc-prize-2026-arc-agi-3"

# --- 门控阈值（按 bench 模式分档，见模块 docstring 的推导） ---
THRESHOLDS = {
    "FULL":  {"moved": 15, "total": 12.0},
    "SHORT": {"moved": 8,  "total": 4.0},
}
# 参考基线（只写进报告做对照，不参与判定）
BASELINES = {
    "v27": "总分 18.61 / 有动作 19/25 / 0.744 局（5400 s bench）",
    "v25": "总分 14.52 / 有动作 14/25 / 0.581 局（5400 s bench，agent 被掐死）",
    "official": "总分 40.0 / 1.6002 局（官方 Duck 基线，25 局 × 20 passes）",
}


def run(cmd: list[str], timeout: float = 300.0) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return p.returncode, ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def kernel_status(kernel: str) -> str:
    _, text = run([str(KAGGLE), "-m", "kaggle", "kernels", "status", kernel], timeout=90)
    return " ".join(text.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="owner/slug")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--min-moved", type=int, default=0, help="覆盖默认阈值")
    ap.add_argument("--min-total", type=float, default=0.0, help="覆盖默认阈值")
    ap.add_argument("--require", action="append", default=[],
                    help="日志里必须出现的字符串（可多次）")
    ap.add_argument("--forbid", action="append", default=[],
                    help="日志里不允许出现的字符串（可多次）")
    ap.add_argument("--budget-regex", default="",
                    help="LINGJING_BUDGET 行必须匹配的正则（v28 用 "
                         r"\"LINGJING_BUDGET 16 7920 900 .*giveup 2700\" 来验证四个旋钮真的生效)")
    ap.add_argument("--submit", action="store_true", help="过闸后真的提交")
    ap.add_argument("--no-submit", action="store_true",
                    help="显式声明只体检不提交（默认行为就是只体检）")
    ap.add_argument("--watch-minutes", type=int, default=0)
    ap.add_argument("--message", default="")
    ap.add_argument("--wait-hours", type=float, default=12.0, help="等 kernel 终态的小时数")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else HERE / f"_gate_out_{args.kernel.split('/')[-1]}"
    report_path = Path(args.report) if args.report else HERE / f"_gate_report_{args.kernel.split('/')[-1]}.txt"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    say(f"=== Duck 门控 === kernel={args.kernel} version={args.version}")
    for k, v in BASELINES.items():
        say(f"    基线[{k}]: {v}")

    # ---- 1) 等 kernel 终态 ----
    deadline = time.time() + args.wait_hours * 3600
    st = ""
    while time.time() < deadline:
        st = kernel_status(args.kernel)
        short = st[st.find("KernelWorkerStatus."):][:60] if "KernelWorkerStatus." in st else st[:120]
        say(f"[{time.strftime('%H:%M:%S')}] {short}")
        if "COMPLETE" in st or "ERROR" in st or "CANCEL" in st:
            break
        time.sleep(120)
    else:
        say("WATCH_TIMEOUT kernel 仍未到终态 —— 不提交")
        return 2

    if "COMPLETE" not in st:
        say(f"KERNEL_NOT_COMPLETE({st[:80]}) —— 不提交")
        return 3

    # ---- 2) 拿主日志 ----
    log = None
    for _old in sorted(out_dir.glob("*embedded_log*.txt"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        if _old.stat().st_size > 1000 and "LINGJING_BUDGET" in _old.read_text(encoding="utf-8",
                                                                             errors="replace"):
            log = _old
            say(f"复用本地主日志 {log.name} size={log.stat().st_size}")
            break
    for attempt in range(1, 13):
        if log is not None:
            break
        code, _ = run([str(KAGGLE), "-m", "kaggle", "kernels", "output",
                       args.kernel, "-p", str(out_dir)], timeout=600)
        say(f"下载 try={attempt} rc={code}")
        cands = sorted(out_dir.rglob("*embedded_log*.txt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            cands = sorted(out_dir.rglob("*_p0.log"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if cands and cands[0].stat().st_size > 1000:
            log = cands[0]
            break
        time.sleep(30)
    if log is None:
        say("NO_LOG_DOWNLOADED —— 不提交")
        return 4
    txt = log.read_text(encoding="utf-8", errors="replace")
    say(f"日志 {log.name} size={len(txt)}")

    # ---- 3) bench 模式分档 ----
    mode = "FULL"
    m = re.search(r"SHORT_BENCH.*?per_game_s\s+(\d+)", txt)
    if m:
        mode = "SHORT"
    per_game = int(m.group(1)) if m else None
    thr = dict(THRESHOLDS[mode])
    if args.min_moved:
        thr["moved"] = args.min_moved
    if args.min_total:
        thr["total"] = args.min_total
    say(f"bench 模式 = {mode}（per_game_s={per_game}）阈值 moved>={thr['moved']} total>={thr['total']}")

    # ---- 4) 硬健康检查 ----
    ok = True
    must_have = [
        "vLLM server ready",
        "VLLM OPENAI SERVER QWEN SMOKE TEST",
        "LINGJING_BUDGET",
    ] + list(args.require)
    must_not = [
        "RuntimeError: LINGJING_CUDA_LIBS_MISSING",
        "RuntimeError: LINGJING_TORCH_IMPORT_FAIL",
        "LINGJING_TORCH_PROBE 1 ",
        "Traceback (most recent call last)",
    ] + list(args.forbid)
    for mm in must_have:
        hit = mm in txt
        say(("PASS " if hit else "FAIL ") + mm)
        ok = ok and hit
    for mm in must_not:
        hit = mm in txt
        say(("PASS(no) " if not hit else "FAIL(present) ") + mm)
        ok = ok and not hit

    # smoke 的 "Generated:" 必须是真输出，不能是 notebook 源码回显
    gens = [g.strip() for g in re.findall(r"Generated:(.+?)\\?n", txt)]
    real = [g for g in gens if len(g) > 3 and "flush" not in g and "print(" not in g]
    if real:
        say("PASS smoke Generated -> " + real[-1][:120])
    else:
        say(f"FAIL smoke Generated 为空 (candidates={len(gens)}, real={len(real)})")
        ok = False

    # v28 专属：admit limit 必须是 16
    seq = re.findall(r"LINGJING_MAX_NUM_SEQS\s+(\[[^\]]*\])", txt)
    if seq:
        good = any("'16'" in s for s in seq)
        say(("PASS " if good else "FAIL ") + f"admit limit -> {seq[-1]}")
        ok = ok and good

    # 旋钮
    budget = re.search(r"LINGJING_BUDGET.*", txt)
    if budget:
        say("PASS budget -> " + budget.group(0).strip()[:200])
        b = budget.group(0)
        for want, label in (
            ("think True", "thinking ON"),
            ("upscale 4", "upscale 4"),
            ("giveup", "giveup 字段存在"),
            ("900", "analyzer_timeout 900"),
        ):
            hit = want in b
            say(("PASS " if hit else "FAIL ") + f"{label} ({want})")
            ok = ok and hit
        if "max_runtime" not in b and " 7920 " not in b and " 5400 " not in b:
            say("WARN budget 行里没看到 max_runtime —— 只作提示，不判失败")
    else:
        say("FAIL 没有 LINGJING_BUDGET 行")
        ok = False

    # 旋钮精确值断言（v28 用：并发 16 / 每局 7920 / timeout 900 / giveup 2700 / upscale 4）
    if args.budget_regex:
        if budget:
            hit = bool(re.search(args.budget_regex, budget.group(0)))
            say(("PASS " if hit else "FAIL ") + f"budget 正则 {args.budget_regex!r} "
                f"-> {budget.group(0).strip()[:120]}")
            ok = ok and hit
        else:
            say(f"FAIL budget 正则 {args.budget_regex!r} —— 没有 LINGJING_BUDGET 行")
            ok = False

    # ---- 5) 质量闸门 ----
    fin = re.findall(
        r"\[finished\]\s+(\S+)\s+state=(\S+)\s+level=([\d.]+)/(\d+)\s+score=([\d.]+)\s+actions=(\d+)\s+tokens=(\d+)",
        txt,
    )
    moved = [f for f in fin if int(f[5]) > 0]
    total_score = sum(float(f[4]) for f in fin)
    say(f"局数 finished={len(fin)}  有动作={len(moved)}  总分={total_score:.2f}")
    for f in sorted(fin, key=lambda x: float(x[4]), reverse=True):
        say(f"   {f[0]:>16} score={f[4]:>5} level={f[2]}/{f[3]} actions={f[5]:>5} tokens={f[6]:>8}")
    if len(fin) < 25:
        say(f"FAIL bench 没跑完（finished={len(fin)} < 25）")
        ok = False
    if len(moved) >= thr["moved"]:
        say(f"PASS 有动作局 {len(moved)} >= {thr['moved']}")
    else:
        say(f"FAIL 有动作局 {len(moved)} < {thr['moved']}")
        ok = False
    if total_score >= thr["total"]:
        say(f"PASS 总分 {total_score:.2f} >= {thr['total']}")
    else:
        say(f"FAIL 总分 {total_score:.2f} < {thr['total']}")
        ok = False

    if not ok:
        say("GATE_FAILED —— 不提交（配额保住）")
        return 5

    if not args.submit:
        say("GATE_PASSED —— 按 --no-submit 只体检不提交")
        return 0

    # ---- 6) 提交（API 通道，绝不用 CLI） ----
    msg = args.message or f"duck {args.kernel.split('/')[-1]} v{args.version}"
    say(f"GATE_PASSED —— competition_submit_code kernel={args.kernel} version={args.version}")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        os.environ.setdefault("KAGGLE_USERNAME", "liaojiaxuan")
        tok = Path(os.path.expanduser("~/.kaggle/access_token"))
        if tok.is_file():
            os.environ.setdefault("KAGGLE_KEY", tok.read_text(encoding="utf-8").strip())
        api = KaggleApi()
        api.authenticate()
        resp = api.competition_submit_code(
            file_name="submission.parquet",
            message=msg,
            competition=COMP,
            kernel=args.kernel,
            kernel_version=args.version or None,
        )
        d = resp.to_dict() if hasattr(resp, "to_dict") else getattr(resp, "__dict__", {})
        say("SUBMIT_RESP " + json.dumps(d, ensure_ascii=False, default=str)[:1200])
        say("SUBMIT_OK")
    except Exception as exc:  # noqa: BLE001
        say(f"SUBMIT_FAILED {type(exc).__name__}: {exc}")
        return 6

    # ---- 7) 跟出分 ----
    if args.watch_minutes > 0:
        deadline = time.time() + args.watch_minutes * 60
        while time.time() < deadline:
            time.sleep(60)
            try:
                subs = api.competition_submissions(COMP)
                out = []
                for s in list(subs)[:3]:
                    sd = s.to_dict() if hasattr(s, "to_dict") else getattr(s, "__dict__", {})
                    out.append(sd)
                    say(f"[{time.strftime('%H:%M:%S')}] ref={sd.get('ref')} "
                        f"score={sd.get('publicScore')} status={sd.get('status')} "
                        f"bytes={sd.get('totalBytes')} desc={(sd.get('description') or '')[:60]}")
                for sd in out:
                    if sd.get("description") == msg and sd.get("status") == "COMPLETE":
                        say(f"DONE ref={sd.get('ref')} publicScore={sd.get('publicScore')} "
                            f"totalBytes={sd.get('totalBytes')}")
                        return 0
            except Exception as exc:  # noqa: BLE001
                say(f"poll error {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
