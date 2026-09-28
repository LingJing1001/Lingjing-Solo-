#!/usr/bin/env python3
"""Build the Kaggle notebook from the canonical Lingjing package."""
from __future__ import annotations

import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER_BUILDER = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "scripts" / "build_notebook.py"
CANONICAL = ROOT / "lingjing_solo"

if not STARTER_BUILDER.is_file():
    raise SystemExit(f"Starter builder not found: {STARTER_BUILDER}")
if not CANONICAL.is_dir():
    raise SystemExit(f"Canonical package not found: {CANONICAL}")

os.environ["LINGJING_SRC"] = os.environ.get("LINGJING_SRC", str(CANONICAL))
runpy.run_path(str(STARTER_BUILDER), run_name="__main__")
