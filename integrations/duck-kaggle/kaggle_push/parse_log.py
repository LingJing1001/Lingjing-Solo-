import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
logs = list(out.glob("*.log"))
print("logs", logs)
if not logs:
    raise SystemExit(1)
raw = logs[0].read_text(encoding="utf-8", errors="replace")
for pat in [
    "lingjing patched",
    "WHEELHOUSE_SLUG",
    "vllm031",
    "AssertionError",
    "No matching",
    "CalledProcessError",
    "CUDA GPU check",
    "Installing vLLM",
    "FileNotFound",
    "requirements.lock",
    "cp313",
]:
    print(("HIT" if pat in raw else "miss"), pat)

# Parse Kaggle JSONL-ish log entries
hits = []
for m in re.finditer(r'"data":"((?:\\.|[^"\\])*)"', raw):
    try:
        s = json.loads(f'"{m.group(1)}"')
    except Exception:
        continue
    if any(
        k in s
        for k in (
            "ERROR",
            "AssertionError",
            "CalledProcessError",
            "No matching",
            "lingjing patched",
            "WHEELHOUSE_SLUG",
            "CUDA GPU",
            "Installing vLLM",
            "vllm031",
        )
    ):
        hits.append(s.strip())
for line in hits[-40:]:
    print(">", line[:300])
