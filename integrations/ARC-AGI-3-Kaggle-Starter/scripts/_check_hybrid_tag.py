"""Quick sanity check for hybrid my_agent INLINE lengths."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "agent" / "my_agent.py").read_text(encoding="utf-8")
tag = re.search(r'^BUILD_TAG = "([^"]+)"', src, re.M)
print("BUILD_TAG", tag.group(1) if tag else None)

tree = ast.parse(src)
for node in tree.body:
    name = None
    value = None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        name, value = node.target.id, node.value
    elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name, value = node.targets[0].id, node.value
    if name not in ("_LS20_LEVEL_ACTIONS", "_AR25_LEVEL_ACTIONS") or value is None:
        continue
    if name == "_LS20_LEVEL_ACTIONS":
        d = ast.literal_eval(value)
        total = sum(len(v) for v in d.values())
        print(name, {k: len(v) for k, v in sorted(d.items())}, "total", total)
    else:
        # AR25 uses BinOp concatenations — count ACTION lists via regex fallback
        m = re.search(r"_AR25_LEVEL_ACTIONS.*?=\s*\{(.*?)\n\}", src, re.S)
        print(name, "present", bool(m), "levels_lines", src.count("\n    ") if m else 0)
