#!/usr/bin/env python3
"""Scorecard 404 失败分类与清理工具。

扫描已采集的 episode 产物，对其中引用的 scorecard_id 逐个探活：
  - alive              : scorecard 仍可 GET
  - not_found_404      : 服务端确认不存在（真销毁/无效 ID）
  - cookie_missing_404 : 首次 404 但换新 Arcade 实例后 GET 成功（会话 cookie 丢失）
  - error              : 其他异常（网络/超时/解析）

not_found_404 与 cookie_missing_404 的关联文件原子移动到 quarantine 目录，
并写 reaper_report.json。幂等：已移动的 episode 不重复处理。

用法:
    python tools/aop/scorecard_reaper.py data/episodes \
        --quarantine data/quarantine/stale_scorecard \
        --report data/quarantine/reaper_report.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ── 探活分类 ──────────────────────────────────────────────────────────────
ALIVE = "alive"
NOT_FOUND = "not_found_404"
COOKIE_MISSING = "cookie_missing_404"
ERROR = "error"


@dataclass
class ProbeResult:
    card_id: str
    status: str
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class ReaperReport:
    scanned_files: int = 0
    unique_card_ids: int = 0
    alive: list[str] = field(default_factory=list)
    not_found_404: list[str] = field(default_factory=list)
    cookie_missing_404: list[str] = field(default_factory=list)
    error: list[str] = field(default_factory=list)
    moved_files: list[str] = field(default_factory=list)
    skipped_already_quarantined: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_files": self.scanned_files,
            "unique_card_ids": self.unique_card_ids,
            "alive": self.alive,
            "not_found_404": self.not_found_404,
            "cookie_missing_404": self.cookie_missing_404,
            "error": self.error,
            "moved_files": self.moved_files,
            "skipped_already_quarantined": self.skipped_already_quarantined,
        }


# ── ID 提取 ───────────────────────────────────────────────────────────────
def _extract_ids_from_json(obj: Any) -> set[str]:
    """递归从一个 JSON 对象里找 scorecard_id / card_id 字段。"""
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("scorecard_id", "card_id") and isinstance(v, str) and v:
                found.add(v)
            else:
                found |= _extract_ids_from_json(v)
    elif isinstance(obj, list):
        for item in obj:
            found |= _extract_ids_from_json(item)
    return found


def collect_scorecard_ids(root: Path) -> dict[str, list[Path]]:
    """扫描 root 下所有 .json/.jsonl 文件，返回 {card_id: [file_paths]}。

    幂等：只读不写。跳过 quarantine 目录自身。
    """
    mapping: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in (".json", ".jsonl"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        ids: set[str] = set()
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    ids |= _extract_ids_from_json(json.loads(line))
                except json.JSONDecodeError:
                    continue
        else:
            try:
                ids = _extract_ids_from_json(json.loads(text))
            except json.JSONDecodeError:
                continue
        for cid in ids:
            mapping.setdefault(cid, []).append(path)
    return mapping


# ── 探活 ──────────────────────────────────────────────────────────────────
def _is_404(exc: BaseException) -> bool:
    """判断异常是否为 HTTP 404。兼容 requests.HTTPError 与通用 Exception。"""
    status = getattr(exc, "response", None)
    if status is not None:
        code = getattr(status, "status_code", None)
        if code == 404:
            return True
    msg = str(exc)
    return "404" in msg or "Not Found" in msg


def probe_scorecard(
    card_id: str,
    get_fn: Callable[[str], Any],
    retry_get_fn: Optional[Callable[[str], Any]] = None,
) -> ProbeResult:
    """探活单个 scorecard。

    Args:
        card_id: scorecard ID
        get_fn: 主 GET 调用（通常 arcade.get_scorecard）
        retry_get_fn: 可选的换实例重试 GET（用于区分 cookie_missing）
    """
    t0 = time.perf_counter()
    try:
        get_fn(card_id)
        return ProbeResult(card_id, ALIVE, latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as exc:
        if not _is_404(exc):
            return ProbeResult(card_id, ERROR, detail=repr(exc),
                               latency_ms=(time.perf_counter() - t0) * 1000)
        # 首次 404 → 尝试换实例重试
        if retry_get_fn is not None:
            try:
                retry_get_fn(card_id)
                return ProbeResult(card_id, COOKIE_MISSING, detail="recovered on fresh instance",
                                   latency_ms=(time.perf_counter() - t0) * 1000)
            except Exception as exc2:
                if _is_404(exc2):
                    return ProbeResult(card_id, NOT_FOUND, detail="404 on both attempts",
                                       latency_ms=(time.perf_counter() - t0) * 1000)
                return ProbeResult(card_id, ERROR, detail=f"retry: {repr(exc2)}",
                                   latency_ms=(time.perf_counter() - t0) * 1000)
        return ProbeResult(card_id, NOT_FOUND, detail="404, no retry available",
                           latency_ms=(time.perf_counter() - t0) * 1000)


# ── 清理 ──────────────────────────────────────────────────────────────────
def reap(
    root: Path,
    get_fn: Callable[[str], Any],
    retry_get_fn: Optional[Callable[[str], Any]] = None,
    quarantine: Optional[Path] = None,
    move: bool = True,
) -> ReaperReport:
    """扫描 + 探活 + 清理。返回 ReaperReport。

    幂等：已在 quarantine 目录下的文件不再处理。
    """
    report = ReaperReport()
    if quarantine is not None:
        quarantine.mkdir(parents=True, exist_ok=True)

    mapping = collect_scorecard_ids(root)
    # 排除已落入 quarantine 的文件（quarantine 常是 root 子目录，避免扫回）
    if quarantine is not None:
        for cid in list(mapping):
            mapping[cid] = [p for p in mapping[cid] if quarantine not in p.parents and p != quarantine]
            if not mapping[cid]:
                del mapping[cid]
    report.scanned_files = len({p for paths in mapping.values() for p in paths})
    report.unique_card_ids = len(mapping)

    for card_id in sorted(mapping):
        files = mapping[card_id]
        # 幂等：全部文件已在 quarantine → 跳过
        if quarantine is not None and all(quarantine in f.parents or f == quarantine for f in files):
            report.skipped_already_quarantined += 1
            continue

        result = probe_scorecard(card_id, get_fn, retry_get_fn)

        if result.status == ALIVE:
            report.alive.append(card_id)
            continue
        if result.status == NOT_FOUND:
            report.not_found_404.append(card_id)
        elif result.status == COOKIE_MISSING:
            report.cookie_missing_404.append(card_id)
        else:
            report.error.append(card_id)
            continue  # error 不清理，保留供诊断

        # 移动 404 关联文件到 quarantine
        if move and quarantine is not None:
            for f in files:
                if quarantine in f.parents:
                    continue
                dst = quarantine / f.name
                # 同名冲突时加 card_id 前缀
                if dst.exists():
                    dst = quarantine / f"{card_id[:8]}_{f.name}"
                try:
                    shutil.move(str(f), str(dst))
                    report.moved_files.append(str(f))
                except OSError:
                    pass

    return report


# ── CLI ───────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Scorecard 404 失败分类与清理")
    parser.add_argument("root", type=Path, help="episode 产物根目录")
    parser.add_argument("--quarantine", type=Path, default=Path("data/quarantine/stale_scorecard"))
    parser.add_argument("--report", type=Path, default=Path("data/quarantine/reaper_report.json"))
    parser.add_argument("--no-move", action="store_true", help="只分类不移动")
    args = parser.parse_args()

    # 延迟导入 arcade_agi，避免无 API key 时炸
    try:
        import arc_agi  # type: ignore[import-not-found]
        from arc_agi import OperationMode  # type: ignore[import-not-found]
    except ImportError:
        raise SystemExit("arc_agi 未安装，无法探活远程 scorecard")

    arc = arc_agi.Arcade(operation_mode=OperationMode.ONLINE)

    def get_fn(cid: str):
        arc.get_scorecard(cid)

    def retry_get_fn(cid: str):
        fresh = arc_agi.Arcade(operation_mode=OperationMode.ONLINE)
        fresh.get_scorecard(cid)

    rep = reap(args.root, get_fn, retry_get_fn, args.quarantine, move=not args.no_move)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"REAPED ids={rep.unique_card_ids} alive={len(rep.alive)} "
          f"not_found={len(rep.not_found_404)} cookie_missing={len(rep.cookie_missing_404)} "
          f"error={len(rep.error)} moved={len(rep.moved_files)} report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
