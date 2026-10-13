import ast
import json
from pathlib import Path

nb = json.loads((Path(__file__).parent / "lingjing-duck-harness.ipynb").read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    try:
        ast.parse(src)
        print(f"ok cell {i}")
    except SyntaxError as e:
        print(f"FAIL cell {i}: {e}")
        raise
print("all_code_cells_ok")
