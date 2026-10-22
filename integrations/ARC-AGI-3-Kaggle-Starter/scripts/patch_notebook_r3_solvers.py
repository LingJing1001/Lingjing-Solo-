#!/usr/bin/env python
"""Embed the g50t/ka59 local solvers into the Kaggle notebook (offline patcher).

Why a script and not hand-editing: cell-2 stores every bundled file as a
single-quoted Python string literal with \\n escapes inside `_LJ_FILES`. Three
solver files (~800 lines total) hand-escaped via text edits is how notebooks
corrupt.
This script does the escaping programmatically (repr), rewrites the notebook
via json.load/dump, keeps a timestamped backup, and re-verifies the result.

What it patches (idempotent -- re-running is a no-op):

1. cell-2 (the _LJ_FILES builder):
   before the final `for _rel, _src in _LJ_FILES.items():` loop, append

       _LJ_FILES['solver_common.py'] = <repr of source>
       _LJ_FILES['g50t_solver.py']   = <repr of source>
       _LJ_FILES['ka59_solver.py']   = <repr of source>

   so the files materialize at /tmp/lingjing_solo/ on Kaggle (the two
   solvers `import solver_common`, so all three must be embedded).

2. cell-3 (my_agent.py source, R3 block):
   a) after `_tl = _r3_time_limit(20.0)` insert the local-solver branch
      (same logic as agent/my_agent.py's R3 wiring, plus /tmp/lingjing_solo
      as a module search dir, because on Kaggle my_agent.py lives in
      agents/templates/ while the solvers land in /tmp/lingjing_solo).
   b) guard the shadow branch with `and found is None` so a solved local
      path is never thrown away by a redundant shadow search.

3. Sanity assertions: cell-2 mentions both solver keys; cell-3 contains
   the local_solver markers and the guarded shadow condition.

Usage:
    python scripts/patch_notebook_r3_solvers.py
    python scripts/patch_notebook_r3_solvers.py --notebook /path/to.ipynb
    python scripts/patch_notebook_r3_solvers.py --dry-run   # show diffs, no write
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

STARTER = Path(__file__).resolve().parents[1]
NB_DEFAULT = Path(
    r"C:/Users/huawei/Documents/arc-prize-2026-smart-router-v2-dpl.ipynb"
)
SOLVERS = {
    # name -> (source path, sanity marker that must appear in the source)
    "solver_common.py": (
        STARTER / "agent" / "solver_common.py", "def replay_bfs("),
    "g50t_solver.py": (
        STARTER / "agent" / "g50t_solver.py", "def solve("),
    "ka59_solver.py": (
        STARTER / "agent" / "ka59_solver.py", "def solve("),
}

# --- anchors (must be unique inside the target cell, verified below) --------
A_FILES_LOOP = "for _rel, _src in _LJ_FILES.items():"
A_R3_TLIMIT = "_tl = _r3_time_limit(20.0)"
A_SHADOW = "if _has_sa and gid in _sa._SHADOW_MODULES:"

# Inserted into cell-2 just before the write loop. Placeholder tokens are
# replaced (never str.format -- the payload contains braces).
CELL2_INSERT_TMPL = (
    "# --- R3 local solvers embedded by scripts/patch_notebook_r3_solvers.py ---\n"
    "# Files materialize at /tmp/lingjing_solo/{name} on Kaggle; the R3 wiring in\n"
    "# my_agent.py adds that dir to sys.path when importing them.\n"
    "_LJ_FILES['{name}'] = {payload}\n"
)

# Inserted into cell-3 after the R3 budget line. NOTE: 24-space indent to sit
# inside the r3 `else: try:` block, exactly where the repo my_agent.py has it.
LOCAL_SOLVER_BRANCH = '''                        # 本地专用求解器（不依赖 C:/newtask-pi / shadow；机制来自直读游戏源码）：
                        #   g50t: 快档位置 BFS(引擎预言函数) + 回放聚焦 + 俯冲
                        #   ka59: 回放加权队列 + 俯冲（含 0022 方块点击候选）
                        # 失败一律兜底回 IDDFS，绝不影响主流程。
                        if gid in ("g50t", "ka59") and found is None:
                            try:
                                import os as _os5, sys as _sys5
                                for _cand in (
                                    _os5.path.dirname(_os5.path.abspath(__file__)),
                                    "/tmp/lingjing_solo",
                                ):
                                    if _cand and _cand not in _sys5.path and _os5.path.isdir(_cand):
                                        _sys5.path.insert(0, _cand)
                                _lmod = __import__("g50t_solver" if gid == "g50t" else "ka59_solver")
                                _lstl = _r3_time_limit(90.0)
                                print(f"[{BUILD_TAG}] r3-attempt gid={gid} L={levels} (local_solver t={_lstl}s)", flush=True)
                                _sp = _lmod.solve(env_ref._game, t_limit=_lstl)
                                if _sp:
                                    found = [
                                        (6, int(a["x"]), int(a["y"])) if isinstance(a, dict) else int(a)
                                        for a in _sp
                                    ]
                                    print(f"[{BUILD_TAG}] local_solver path_len={len(found)}", flush=True)
                            except Exception as _lse:
                                print(f"[{BUILD_TAG}] local_solver {gid} failed: {type(_lse).__name__}: {_lse}", flush=True)
'''

SHADOW_GUARDED = "if _has_sa and found is None and gid in _sa._SHADOW_MODULES:"


def _cell_text(cell) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def _set_cell_text(cell, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def _insert_after(text: str, anchor: str, block: str) -> str:
    """Insert `block` right after the (single) line containing `anchor`."""
    lines = text.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if anchor in ln]
    if len(hits) != 1:
        raise SystemExit(
            f"anchor not unique ({len(hits)} hits): {anchor!r}"
        )
    i = hits[0]
    eol = "\n" if lines[i].endswith("\n") else ""
    lines.insert(i + 1, block if block.endswith("\n") else block + eol)
    return "".join(lines)


def _replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"expected exactly 1 occurrence, got {n}: {old!r}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", type=Path, default=NB_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    nb_path: Path = args.notebook

    print(f"notebook : {nb_path}")
    print(f"starter  : {STARTER}")

    # ---- load inputs ------------------------------------------------------
    raw = nb_path.read_bytes()
    nb = json.loads(raw.decode("utf-8"))          # also validates the JSON
    cells = nb.get("cells") or []
    print(f"cells    : {len(cells)}")

    cell2 = next((c for c in cells if "_LJ_FILES" in _cell_text(c)), None)
    cell3 = next(
        (c for c in cells if "_tl = _r3_time_limit(20.0)" in _cell_text(c)),
        None,
    )
    if cell2 is None or cell3 is None:
        raise SystemExit("could not locate cell-2 (_LJ_FILES) or cell-3 (R3 block)")

    payloads = {}
    for name, (path, marker) in SOLVERS.items():
        src = path.read_text(encoding="utf-8")
        if marker not in src:
            raise SystemExit(f"{path} sanity check failed (missing {marker!r})")
        payloads[name] = repr(src)
        print(f"solver   : {name}  {len(src)} chars -> {len(payloads[name])} escaped")

    # ---- patch cell-2 -----------------------------------------------------
    t2 = _cell_text(cell2)
    if "g50t_solver.py" in t2:
        print("cell-2   : already patched (g50t_solver.py present) -- skip")
    else:
        block = "".join(
            CELL2_INSERT_TMPL.replace("{name}", name).replace("{payload}", payloads[name])
            for name in SOLVERS
        )
        if t2.count(A_FILES_LOOP) != 1:
            raise SystemExit(f"cell-2 anchor not unique: {A_FILES_LOOP!r}")
        t2 = t2.replace(A_FILES_LOOP, block + A_FILES_LOOP, 1)
        print(f"cell-2   : +{len(SOLVERS)} solver entries before write loop")
    _set_cell_text(cell2, t2)

    # ---- patch cell-3 -----------------------------------------------------
    t3 = _cell_text(cell3)
    if "local_solver path_len=" in t3:
        print("cell-3   : already patched (local_solver marker present) -- skip")
    else:
        if t3.count(A_R3_TLIMIT) != 1:
            raise SystemExit(f"cell-3 anchor not unique: {A_R3_TLIMIT!r}")
        t3 = _insert_after(t3, A_R3_TLIMIT, LOCAL_SOLVER_BRANCH)
        print("cell-3   : +local_solver branch after _r3_time_limit(20.0)")
    if SHADOW_GUARDED in t3:
        print("cell-3   : shadow guard already present -- skip")
    elif A_SHADOW in t3:
        t3 = _replace_once(t3, A_SHADOW, SHADOW_GUARDED)
        print("cell-3   : shadow branch guarded with 'found is None'")
    else:
        print("cell-3   : WARN shadow anchor not found (structure changed?)")
    _set_cell_text(cell3, t3)

    # ---- verify (in-memory, before writing) -------------------------------
    check2 = _cell_text(cell2)
    check3 = _cell_text(cell3)
    for name in SOLVERS:
        assert f"_LJ_FILES['{name}']" in check2, f"cell-2 missing {name}"
    for marker in (
        'if gid in ("g50t", "ka59") and found is None:',
        "__import__(\"g50t_solver\" if gid == \"g50t\" else \"ka59_solver\")",
        "_r3_time_limit(90.0)",
        "local_solver path_len=",
        "local_solver {gid} failed:",
        SHADOW_GUARDED,
    ):
        assert marker in check3, f"cell-3 missing {marker!r}"
    print("verify   : all markers present in patched cells")

    if args.dry_run:
        print("dry-run  : notebook NOT written")
        return

    # ---- backup + write ---------------------------------------------------
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = nb_path.with_name(nb_path.stem + f".pre-r3-solvers.{stamp}.ipynb.bak")
    shutil.copy2(nb_path, backup)
    print(f"backup   : {backup}")

    nb_path.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    # ---- post-write verification: the file must still be valid JSON -------
    recheck = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(recheck.get("cells") or []) == len(cells), "cell count changed?!"
    re_cell2 = next(c for c in recheck["cells"] if "_LJ_FILES" in _cell_text(c))
    re_cell3 = next(
        c for c in recheck["cells"]
        if "_tl = _r3_time_limit(20.0)" in _cell_text(c)
    )
    assert "g50t_solver.py" in _cell_text(re_cell2)
    assert "local_solver path_len=" in _cell_text(re_cell3)
    print("post-write: JSON re-parsed OK, markers intact")

    print("\nDONE. After this, on Kaggle rerun:")
    print("  1. cell-2 writes /tmp/lingjing_solo/{solver_common,g50t_solver,ka59_solver}.py")
    print("  2. cell-3 R3 block: local solver (90s) -> IDDFS fallback")
    print("  Expected log markers: 'r3-attempt ... (local_solver t=90s)'")
    print("                        'local_solver path_len=N'")


if __name__ == "__main__":
    main()
