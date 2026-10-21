"""ARC-AGI-3 冲分仪表盘：一页聚合 kernel 状态 / 提交历史 / 监控日志。

用法:
    python _dashboard.py            # 人类可读摘要
    python _dashboard.py --json     # 机器可读
纯 HTTP 实现（无子进程），避免沙箱限制。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent  # ARC奖2026 - ARC-AGI-3
DOCS = PROJECT / "docs"
KERNEL = "liaojiaxuan/lingjing-duck-harness-v1"
COMP = "arc-prize-2026-arc-agi-3"
CST = timezone(timedelta(hours=8))


def _token() -> str:
    for cand in (Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".kaggle" / "access_token",
                 Path.home() / ".kaggle" / "access_token"):
        if cand.is_file():
            return cand.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def _get(url: str, timeout: int = 60) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={
        "User-Agent": "lingjing-dashboard/1.0",
        "Authorization": "Bearer " + _token(),
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:300]
    except Exception as e:
        return -1, repr(e)[:200]


def kernel_status() -> str:
    owner, slug = KERNEL.split("/")
    rc, body = _get(f"https://www.kaggle.com/api/v1/kernels/status?user_name={owner}&kernel_slug={slug}")
    if rc == 200:
        try:
            return json.loads(body).get("status", "?")
        except Exception:
            pass
    return f"UNKNOWN(rc={rc} {body[:80]})"


def submission_history(limit: int = 8) -> list[dict]:
    rc, body = _get(f"https://www.kaggle.com/api/v1/competitions/submissions/list/{COMP}")
    rows = []
    if rc == 200:
        try:
            for s in json.loads(body)[:limit]:
                rows.append({
                    "date": s.get("date", "?"),
                    "description": (s.get("description") or "")[:60],
                    "status": s.get("status", "?"),
                    "publicScore": s.get("publicScore"),
                })
        except Exception:
            pass
    return rows


def monitor_tail(path: Path, n: int = 4) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [l for l in lines if l.strip()][-n:]


def main() -> None:
    now = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S CST")
    ks = kernel_status()
    hist = submission_history()
    mon = monitor_tail(DOCS / "_v25_monitor_log.md")

    print("=" * 72)
    print(f"ARC-AGI-3 冲分仪表盘  {now}")
    print("=" * 72)
    print(f"[kernel] {KERNEL}: {ks}")
    print()
    print("[最近提交]")
    for h in hist:
        print(f"  {h['date']}  {str(h['publicScore']):>7}  {h['status']:<12} {h['description']}")
    print()
    print("[监控日志尾部]")
    for m in mon:
        print(f"  {m[:150]}")
    print("=" * 72)

    if "--json" in sys.argv:
        json.dump({"time": now, "kernel": ks, "submissions": hist, "monitor": mon},
                  sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
