import json
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
# try json array
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    data = None

lines = []
if isinstance(data, list):
    for rec in data:
        if isinstance(rec, dict) and "data" in rec:
            lines.append(str(rec.get("stream_name", "out")) + "|" + rec["data"])
else:
    lines = raw.splitlines()

# print last 80 informative lines
interesting = []
for line in lines:
    low = line.lower()
    if any(
        k in line
        for k in (
            "Error",
            "ERROR",
            "Traceback",
            "Exception",
            "lingjing",
            "setup",
            "find-links",
            "wheels",
            "pip",
            "RuntimeError",
            "Assertion",
            "Papermill",
            "In [",
            "WHEELHOUSE",
            "PYTHON",
        )
    ):
        interesting.append(line[:400])

print("n_lines", len(lines), "interesting", len(interesting))
for x in interesting[-50:]:
    print(x.replace("\n", "\\n")[:380])
