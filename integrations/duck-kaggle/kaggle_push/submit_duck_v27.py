"""提交 Duck v27 kernel 到竞赛，并跟踪出分。

为什么不用 `kaggle competitions submit` CLI
-------------------------------------------
v27 的门控脚本 `_gate_submit_v27.py` 用 CLI 提交，参数顺序写错了：

    kaggle competitions submit <COMP> -k <kernel> -v <ver> -f submission.parquet

CLI 2.2.4 把 `<COMP>` 当成了位置参数（它要的是 `-c <competition>`），于是把
`competitions` 当成待上传的文件名，报

    can't open file '...\\duck-kaggle\\competitions': [Errno 2] No such file or directory

⇒ 门控明明 GATE_PASSED，却在烧配额前一步失败（v27 第一次就死在这里）。

改走 API 通道：`competition_submit_code(file_name=..., message=..., competition=...,
kernel=..., kernel_version=...)` —— 这正是网页上"Submit to Competition"按钮调的同一个
接口，SmartRouter 线已用它成功提交过（ref 57542504）。**零 CLI 语法风险。**

用法（starter venv python，有 kaggle 包）:
    python submit_duck_v27.py --dry-run
    python submit_duck_v27.py --watch-minutes 180
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARTER = HERE.parents[3] / "ARC-AGI-3-Kaggle-Starter"
REPORT = HERE / "_duck_v27_submit_report.txt"

COMPETITION = "arc-prize-2026-arc-agi-3"
KERNEL = "liaojiaxuan/lingjing-duck-harness-v27"
VERSION = 1
MESSAGE = ("duck v27: official analyzer budget (tool_steps=0/yield=60/max_output=0/"
           "thinking=on/timeout=900/upscale=4) + no-progress giveup; "
           "public-25 offline 0.744/game vs official 1.6002")
FILE_NAME = "submission.parquet"


def _say(lines: list[str], msg: str) -> None:
    print(msg, flush=True)
    lines.append(msg)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status_str(st) -> str:
    s = st.get("status") if isinstance(st, dict) else getattr(st, "status", st)
    return str(getattr(s, "name", s) or s)


def _limits(api) -> dict:
    try:
        lim = api.competition_get_submission_limits(COMPETITION)
        return lim.to_dict() if hasattr(lim, "to_dict") else vars(lim)
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _latest_submissions(api, n: int = 5) -> list[dict]:
    try:
        subs = api.competition_submissions(COMPETITION)
    except Exception as exc:  # noqa: BLE001
        return [{"_error": f"{type(exc).__name__}: {exc}"}]
    out = []
    for s in list(subs)[:n]:
        d = s.to_dict() if hasattr(s, "to_dict") else getattr(s, "__dict__", {})
        out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watch-minutes", type=int, default=0,
                    help="提交后轮询 competition_submissions 等出分的分钟数")
    ap.add_argument("--version", type=int, default=VERSION)
    args = ap.parse_args()

    os.environ.setdefault("KAGGLE_USERNAME", "liaojiaxuan")
    tok = Path(os.path.expanduser("~/.kaggle/access_token"))
    if tok.is_file():
        os.environ.setdefault("KAGGLE_KEY", tok.read_text(encoding="utf-8").strip())

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    lines: list[str] = []
    _say(lines, f"=== Duck v27 提交 === kernel={KERNEL} version={args.version}")
    st = api.kernels_status(KERNEL)
    status = _status_str(st)
    lim = _limits(api)
    _say(lines, f"kernel status = {status}")
    _say(lines, f"limits       = {json.dumps(lim, ensure_ascii=False, default=str)}")

    if status != "COMPLETE":
        _say(lines, f"REFUSE: kernel 是 {status}，不是 COMPLETE —— 不能提交")
        return 2

    if args.dry_run:
        _say(lines, f"DRY-RUN: competition_submit_code(file_name={FILE_NAME!r}, "
                    f"kernel={KERNEL!r}, kernel_version={args.version}, "
                    f"competition={COMPETITION!r})")
        return 0

    resp = api.competition_submit_code(
        file_name=FILE_NAME,
        message=MESSAGE,
        competition=COMPETITION,
        kernel=KERNEL,
        kernel_version=args.version or None,
    )
    d = resp.to_dict() if hasattr(resp, "to_dict") else getattr(resp, "__dict__", {})
    _say(lines, f"SUBMIT_RESP {json.dumps(d, ensure_ascii=False, default=str)}")

    if args.watch_minutes > 0:
        deadline = time.time() + args.watch_minutes * 60
        while time.time() < deadline:
            time.sleep(60)
            subs = _latest_submissions(api, 3)
            _say(lines, f"[{time.strftime('%H:%M:%S')}] latest = "
                        + " | ".join(
                            f"ref={s.get('ref')} score={s.get('publicScore')} "
                            f"status={s.get('status')} bytes={s.get('totalBytes')} "
                            f"desc={(s.get('description') or '')[:40]}"
                            for s in subs))
            for s in subs:
                if s.get("description") == MESSAGE and s.get("status") == "COMPLETE":
                    _say(lines, f"DONE ref={s.get('ref')} publicScore={s.get('publicScore')} "
                                f"totalBytes={s.get('totalBytes')}")
                    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
