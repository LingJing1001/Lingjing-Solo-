"""Pack Lingjing Franzen patches into a tiny Kaggle dataset overlay."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DUCK_KAGGLE = HERE.parent
VENDOR = DUCK_KAGGLE.parent / "duck-harness"
SRC_ROOT = VENDOR / "ARC3-Inference" / "inference"
OUT = DUCK_KAGGLE / "overlay_dataset"
OVERLAY_INFERENCE = OUT / "inference"

PATCHED = [
    "agent/action_names.py",
    "agent/prompts.py",
    "agent/runtime_state.py",
    "agent/python_tool_sandbox.py",
    "agent/tool_agent.py",
    "framework/kaggle.py",
    "framework/solver.py",
]

INITS = [
    "__init__.py",
    "agent/__init__.py",
    "framework/__init__.py",
]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OVERLAY_INFERENCE.mkdir(parents=True)

    missing: list[str] = []
    for rel in PATCHED:
        src = SRC_ROOT / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dest = OVERLAY_INFERENCE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    for rel in INITS:
        src = SRC_ROOT / rel
        dest = OVERLAY_INFERENCE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dest)
        else:
            dest.write_text("", encoding="utf-8")

    if missing:
        raise SystemExit(f"missing patched files: {missing}")

    meta = {
        "title": "lingjing-duck-franzen-overlay",
        "id": "liaojiaxuan/lingjing-duck-franzen-overlay",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (OUT / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "LINGJING_OVERLAY.txt").write_text(
        "Franzen-style Duck patches for Lingjing\n" + "\n".join(PATCHED) + "\n",
        encoding="utf-8",
    )
    print("OK overlay", OUT)
    for p in sorted(OVERLAY_INFERENCE.rglob("*.py")):
        print(" ", p.relative_to(OUT).as_posix())


if __name__ == "__main__":
    main()
