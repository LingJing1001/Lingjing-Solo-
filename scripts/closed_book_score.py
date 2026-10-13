#!/usr/bin/env python
"""Canonical closed-book score breakdown for the M1 remediation.

The headline number of this repo is NOT the canned 52.60. It is the
closed-book score produced with ``LINGJING_CLOSED_BOOK=2`` (strict), i.e.
frame-only CEAX + heatmap clicks with every baked answer removed.

Three run modes are chained so that each contribution can be isolated:

    canned  (default)  -> inline baked plans + EngineSolver + R3 + CEAX
    tables  (CB=1)     -> R3 shadow solvers + CEAX only
    strict  (CB=2)     -> CEAX + heatmap clicks only  == closed-book capability

From those three totals we derive the decomposition:

    hardcoded_tables_share = canned - tables
    r3_shadow_share        = tables - strict
    closed_book_share      = strict

Usage
-----
    # print the breakdown for the default (committed) bench artifacts
    python scripts/closed_book_score.py

    # regenerate the committed registry after a fresh 25-game run
    python scripts/closed_book_score.py --write state/score_registry.json

    # CI guard: fail (exit 1) if the registry drifts from the bench JSONs
    python scripts/closed_book_score.py --check

This script only reads JSON. It never touches the agent.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Committed bench artifacts (25 games x 400 steps each).
# ``strict`` points at the current closed-book run, which includes the
# frame-only AR25 + tu93 solvers.  Earlier stages are kept for the record:
#   state/bench_closed_book_strict_20261013.json       (0.6200, no frame solvers)
#   state/bench_closed_book_strict_ar25_20261013.json  (0.7312, AR25 = 1 level)
DEFAULT_SOURCES = {
    "canned": "state/bench_official_20261013_tu93.json",
    "tables": "state/bench_closed_book_tables_20261013.json",
    "strict": "state/bench_closed_book_strict_tu93_20261013.json",
}

# Official Kaggle public leaderboard reference, for the predictive-ratio note.
OFFICIAL_KAGGLE_REFERENCE = 0.04

HEADLINE_METRIC = "closed_book_strict"

SCHEMA = "lingjing.score_registry/v1"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def _load(path: str | Path) -> tuple[Path, dict]:
    p = _resolve(path)
    if not p.exists():
        raise SystemExit(f"[closed-book-score] missing bench artifact: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return p, data


def _mode_entry(path: Path, data: dict) -> dict:
    score = float(data["aggregate_score"])
    return {
        "score": round(score, 4),
        "score_exact": score,
        "levels": int(data.get("total_levels_completed", 0)),
        "wins": int(data.get("games_won", 0)),
        "zero_level_games": int(data.get("games_zero_levels", 0)),
        "max_steps": int(data.get("max_steps", 0)),
        "elapsed_sec": round(float(data.get("elapsed_sec", 0.0)), 1),
        "agent": data.get("agent"),
        "generated_at": data.get("generated_at"),
        "source": str(path.relative_to(REPO)).replace("\\", "/"),
    }


def build_registry(sources: dict[str, str]) -> dict:
    modes: dict[str, dict] = {}
    for key, rel in sources.items():
        p, d = _load(rel)
        modes[key] = _mode_entry(p, d)

    canned = modes["canned"]["score_exact"]
    tables = modes["tables"]["score_exact"]
    strict = modes["strict"]["score_exact"]

    hardcoded = canned - tables
    r3_shadow = tables - strict
    closed_book = strict

    def pct(part: float) -> float:
        return round(100.0 * part / canned, 2) if canned else 0.0

    def ratio(part: float) -> float | None:
        return round(part / OFFICIAL_KAGGLE_REFERENCE, 1) if OFFICIAL_KAGGLE_REFERENCE else None

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headline_metric": HEADLINE_METRIC,
        "definition": {
            "closed_book": "LINGJING_CLOSED_BOOK=2 (strict): frame-only CEAX + heatmap clicks; "
            "inline baked plans, EngineSolver and R3 shadow solvers disabled.",
            "canned": "default 5-layer decision chain; the 52.60 headline that must NOT be "
            "reported as capability.",
            "note": "Only the closed-book score is comparable to the official Kaggle public score.",
        },
        "modes": modes,
        "decomposition": {
            "hardcoded_tables": {
                "score": round(hardcoded, 4),
                "pct_of_canned": pct(hardcoded),
                "source": "canned - tables (layer 2 inline baked plans + layer 3 EngineSolver)",
            },
            "r3_shadow": {
                "score": round(r3_shadow, 4),
                "pct_of_canned": pct(r3_shadow),
                "source": "tables - strict (layer 4 R3 shadow solvers / IDDFS)",
            },
            "closed_book": {
                "score": round(closed_book, 4),
                "pct_of_canned": pct(closed_book),
                "source": "strict (layer 5 CEAX + heatmap clicks)",
            },
        },
        "official_kaggle": {
            "public_score": OFFICIAL_KAGGLE_REFERENCE,
            "ratio_canned": ratio(canned),
            "ratio_closed_book": ratio(closed_book),
        },
    }


def _fmt_table(reg: dict) -> str:
    m = reg["modes"]
    d = reg["decomposition"]
    lines = [
        "口径                    分数      关卡  WIN  零关  占罐头比",
        "----------------------------------------------------------------",
        f"canned  (主链)          {m['canned']['score']:>8.4f}  "
        f"{m['canned']['levels']:>4}  {m['canned']['wins']:>3}  {m['canned']['zero_level_games']:>4}   100.00%",
        f"tables  (CB=1)          {m['tables']['score']:>8.4f}  "
        f"{m['tables']['levels']:>4}  {m['tables']['wins']:>3}  {m['tables']['zero_level_games']:>4}   "
        f"{100.0 * m['tables']['score'] / m['canned']['score']:>5.2f}%",
        f"strict  (CB=2) **       {m['strict']['score']:>8.4f}  "
        f"{m['strict']['levels']:>4}  {m['strict']['wins']:>3}  {m['strict']['zero_level_games']:>4}   "
        f"{100.0 * m['strict']['score'] / m['canned']['score']:>5.2f}%",
        "",
        f"  ** = headline metric ({reg['headline_metric']})",
        "",
        "分解:",
        f"  硬编码动作表  {d['hardcoded_tables']['score']:>7.4f}  ({d['hardcoded_tables']['pct_of_canned']:>5.2f}%)",
        f"  R3 影子解法   {d['r3_shadow']['score']:>7.4f}  ({d['r3_shadow']['pct_of_canned']:>5.2f}%)",
        f"  真·闭卷能力   {d['closed_book']['score']:>7.4f}  ({d['closed_book']['pct_of_canned']:>5.2f}%)",
        "",
        f"官方 Kaggle 公开分 {reg['official_kaggle']['public_score']}  "
        f"-> 罐头 {reg['official_kaggle']['ratio_canned']}x / 闭卷 {reg['official_kaggle']['ratio_closed_book']}x",
    ]
    return "\n".join(lines)


def _check(reg: dict, path: Path) -> int:
    if not path.exists():
        print(f"[closed-book-score] FAIL: registry not found: {path}", file=sys.stderr)
        return 1
    old = json.loads(path.read_text(encoding="utf-8"))

    problems: list[str] = []
    if old.get("schema") != reg["schema"]:
        problems.append(f"schema: {old.get('schema')!r} != {reg['schema']!r}")
    if old.get("headline_metric") != reg["headline_metric"]:
        problems.append(
            f"headline_metric: {old.get('headline_metric')!r} != {reg['headline_metric']!r}"
        )
    for mode in ("canned", "tables", "strict"):
        a = old.get("modes", {}).get(mode, {})
        b = reg["modes"][mode]
        for field in ("score", "levels", "wins", "source"):
            if a.get(field) != b.get(field):
                problems.append(
                    f"modes.{mode}.{field}: registry={a.get(field)!r} != bench={b.get(field)!r}"
                )

    if problems:
        print("[closed-book-score] FAIL: score registry drifted from bench artifacts:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\n  Re-run the bench, then: python scripts/closed_book_score.py "
            "--write state/score_registry.json",
            file=sys.stderr,
        )
        return 1
    print("[closed-book-score] OK: registry matches bench artifacts.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canned", default=DEFAULT_SOURCES["canned"], help="canned bench JSON")
    ap.add_argument("--tables", default=DEFAULT_SOURCES["tables"], help="CB=1 bench JSON")
    ap.add_argument("--strict", default=DEFAULT_SOURCES["strict"], help="CB=2 bench JSON")
    ap.add_argument("--write", metavar="PATH", help="write the registry JSON to PATH")
    ap.add_argument("--check", action="store_true", help="compare registry against bench JSONs")
    ap.add_argument("--registry", default="state/score_registry.json", help="registry path for --check")
    ap.add_argument("--json", action="store_true", help="dump the registry as JSON only")
    args = ap.parse_args(argv)

    reg = build_registry(
        {"canned": args.canned, "tables": args.tables, "strict": args.strict}
    )

    if args.write:
        out = _resolve(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[closed-book-score] wrote {out}")

    if args.check:
        return _check(reg, _resolve(args.registry))

    if args.json:
        print(json.dumps(reg, ensure_ascii=False, indent=2))
    else:
        print(_fmt_table(reg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
