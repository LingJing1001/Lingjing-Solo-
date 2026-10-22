"""Check that every call site's keyword arguments match the callee's signature.

Why this exists
---------------
The upstream merge initially took upstream's ``MyAgent._try_llm_batch`` while
keeping our whole-qwen call site, which passes ``history=`` / ``briefing=``.
Python only fails at *call* time, so the file imported fine, compiled fine and
the 5-step smoke test passed -- the TypeError only appeared once a real game
reached the LLM branch:

    TypeError: MyAgent._try_llm_batch() got an unexpected keyword argument 'history'

This checker finds that class of bug statically, in one pass, without a GPU.

Duplicate names resolve to the LAST definition, which is what Python runs.

Run:
    python scripts/check_call_signatures.py agent/my_agent.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def collect(tree: ast.AST) -> dict[str, tuple[set[str], bool]]:
    """name -> (accepted keyword names, accepts **kwargs)"""
    out: dict[str, tuple[set[str], bool]] = {}

    def record(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        a = node.args
        kw = {x.arg for x in a.kwonlyargs} | {
            x.arg for x in a.args if x.arg not in ("self", "cls")
        }
        out[node.name] = (kw, a.kwarg is not None)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            record(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    record(sub)
    return out


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "agent/my_agent.py")
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    sigs = collect(tree)

    problems: list[str] = []
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = None
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            name = fn.attr
        if name is None or name not in sigs:
            continue
        accepted, has_kwargs = sigs[name]
        kws = {k.arg for k in node.keywords if k.arg}
        if not kws or has_kwargs:
            continue
        checked += 1
        bad = kws - accepted
        if bad:
            problems.append(
                f"line {node.lineno}: {name}(...) got unexpected keyword(s) {sorted(bad)}"
                f"  [accepts: {sorted(accepted)}]"
            )

    print(f"[check_call_signatures] {path}: {len(sigs)} signatures, "
          f"{checked} keyword call sites checked")
    if problems:
        print(f"[check_call_signatures] {len(problems)} PROBLEM(S):")
        for p in problems:
            print("   ", p)
        return 1
    print("[check_call_signatures] OK - every keyword argument matches its callee")
    return 0


if __name__ == "__main__":
    sys.exit(main())
