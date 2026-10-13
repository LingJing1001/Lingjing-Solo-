import json
from pathlib import Path

p = Path(__file__).resolve().parent / "lingjing-duck-harness.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
src = "".join(nb["cells"][8]["source"])
print(src)
print("---END---")
print("n_lines", src.count("\n"))
