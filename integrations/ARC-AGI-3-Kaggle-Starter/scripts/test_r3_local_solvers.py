"""One-shot regression check: g50t / ka59 local solvers wired into R3 routing.

Run this AFTER the sandbox/exec environment recovers:

    python scripts/test_r3_local_solvers.py

It does three things, in order:

1. STATIC  - verify agent/my_agent.py actually contains the local-solver
             wiring (g50t/ka59 branch, shadow guard, budget call).
2. COMPILE - py_compile my_agent.py, g50t_solver.py, ka59_solver.py.
3. DYNAMIC - play g50t then ka59 via play_local.py with --parallel 1
             (multi-environment runs share global engine state and produce
             fake failures -- never parallel for these). A game PASSES when
             levels_completed >= 1 (the pre-wiring state was 0 levels).
             The subprocess stdout is scanned for the expected log markers:
               r3-attempt ... (local_solver t=90s)
               local_solver path_len=N

Exit code 0 = all passed; 1 = something failed (details printed).

Notes:
- Requires the arc-agi engine deps and vendor/ARC-AGI-3-Agents (make setup).
- LINGJING_SCRIPTS=1 is force-set so calibration/INLINE branches are active,
  matching the notebook configuration.
- Each game gets a generous wall-clock cap (default 45 min) because the
  local solvers are pure-Python search (g50t L1 ~11s, ka59 L1 ~6s for the
  fast phase, but the agent may keep playing beyond L1).
"""
from __future__ import annotations

import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent" / "my_agent.py"
SOLVERS = {
    "g50t": ROOT / "agent" / "g50t_solver.py",
    "ka59": ROOT / "agent" / "ka59_solver.py",
}
COMMON = ROOT / "agent" / "solver_common.py"
PLAY_LOCAL = ROOT / "scripts" / "play_local.py"
OUT_DIR = ROOT / "runs_r3_solver_test"

# Markers that must exist in my_agent.py for the wiring to be real.
STATIC_MARKERS = [
    'if gid in ("g50t", "ka59") and found is None:',
    '__import__("g50t_solver" if gid == "g50t" else "ka59_solver")',
    "_r3_time_limit(90.0)",
    "local_solver path_len=",
    "local_solver {gid} failed:",
    # shadow must be guarded so it does not re-search after a local solve
    "_has_sa and found is None and gid in _sa._SHADOW_MODULES",
]

MAX_STEPS = 400
MAX_MINUTES = "45"


def check_static() -> bool:
    print("=" * 60)
    print("[1/3] STATIC check: my_agent.py wiring markers")
    text = AGENT.read_text(encoding="utf-8")
    ok = True
    for marker in STATIC_MARKERS:
        if marker in text:
            print(f"  OK   {marker}")
        else:
            print(f"  MISS {marker}")
            ok = False
    if not ok:
        print("  -> wiring text missing; re-apply the local-solver patch.")
    return ok


def check_compile() -> bool:
    print("=" * 60)
    print("[2/3] COMPILE check: py_compile")
    ok = True
    for path in [AGENT, COMMON, *SOLVERS.values()]:
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"  OK   {path.name}")
        except py_compile.PyCompileError as exc:
            print(f"  FAIL {path.name}: {exc}")
            ok = False
    return ok


def check_dynamic() -> bool:
    print("=" * 60)
    print("[3/3] DYNAMIC check: play g50t and ka59 (--parallel 1)")
    OUT_DIR.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["LINGJING_SCRIPTS"] = "1"
    all_ok = True

    for gid in SOLVERS:
        print(f"\n--- game {gid} ---")
        # 陈旧结果防护: 上轮遗留的 <gid>.json 会在本轮崩溃时被误读成 PASS
        stale = OUT_DIR / f"{gid}.json"
        if stale.exists():
            stale.unlink()
        cmd = [
            sys.executable, str(PLAY_LOCAL),
            "--game", gid,
            "--max-steps", str(MAX_STEPS),
            "--max-minutes", MAX_MINUTES,
            "--parallel", "1",
            "--out-dir", str(OUT_DIR),
        ]
        print(f"  $ {' '.join(cmd)}")
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        # persist full log for post-mortem
        log_path = OUT_DIR / f"{gid}.log"
        log_path.write_text(out, encoding="utf-8")
        print(f"  log -> {log_path}")

        # expected solver markers in the agent log
        for marker, label in [
            (r"r3-attempt gid=%s .*local_solver" % gid, "R3 engaged local solver"),
            (r"local_solver path_len=\d+", "solver returned a path"),
        ]:
            m = re.search(marker, out)
            print(f"  {'OK  ' if m else 'warn'} {label}"
                  + (f"  ({m.group(0)})" if m else ""))

        result_path = OUT_DIR / f"{gid}.json"
        if not result_path.exists():
            print(f"  FAIL no result JSON written ({result_path})")
            all_ok = False
            continue
        row = json.loads(result_path.read_text(encoding="utf-8"))
        levels = row.get("levels_completed", 0)
        score = row.get("game_score") or 0
        state = row.get("state")
        err = row.get("error")
        print(f"  levels_completed={levels}  game_score={score}  "
              f"state={state}  error={err}")
        if err:
            print(f"  traceback tail:\n{(row.get('traceback') or '')[-1500:]}")
        if levels >= 1:
            print(f"  PASS {gid}: {levels} level(s) completed")
        else:
            print(f"  FAIL {gid}: still 0 levels")
            all_ok = False

    return all_ok


def main() -> None:
    print(f"repo      : {ROOT}")
    print(f"agent     : {AGENT}")
    print(f"out dir   : {OUT_DIR}")
    ok_static = check_static()
    ok_compile = check_compile() if ok_static else False
    ok_dynamic = check_dynamic() if (ok_static and ok_compile) else False

    print("=" * 60)
    print("RESULT:",
          "PASS" if (ok_static and ok_compile and ok_dynamic) else "FAIL")
    print("  static :", "PASS" if ok_static else "FAIL")
    print("  compile:", "PASS" if ok_compile else "FAIL")
    print("  dynamic:", "PASS" if ok_dynamic else "FAIL")
    sys.exit(0 if (ok_static and ok_compile and ok_dynamic) else 1)


if __name__ == "__main__":
    main()
