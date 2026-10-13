import json
import re
from pathlib import Path

p = Path(__file__).resolve().parent / "lingjing-duck-harness.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
src = "".join(nb["cells"][8]["source"])
for n, line in enumerate(src.splitlines(), 1):
    if any(k in line for k in ("setup_commands", "_command_env", "for command", "PYTHON", "lingjing")):
        print(f"{n}: {line!r}")

pattern = re.compile(
    r"env\s*=\s*_command_env\(\)\s*"
    r"(?:print\([^\n]*\n)?"
    r"for command in json\.loads\(\(BUNDLE_DIR / [\"']setup_commands\.json[\"']\)"
    r"\.read_text\(\)\):",
    re.MULTILINE,
)
print("match", bool(pattern.search(src)))
m = re.search(
    r"for command in json\.loads\(\(BUNDLE_DIR / .setup_commands\.json.\).read_text\(\)\):",
    src,
)
print("loose", bool(m), repr(m.group(0)) if m else None)
