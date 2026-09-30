"""Delegate to the canonical builder in the sibling Kaggle Starter.

The Starter builder receives the canonical Lingjing source explicitly, so the
ARC checkout's legacy compatibility package is never selected accidentally.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

ARC = Path(__file__).resolve().parents[1]
INTEGRATIONS = ARC.parent
WORKSPACE = INTEGRATIONS.parent
STARTER = INTEGRATIONS / "ARC-AGI-3-Kaggle-Starter"
CANONICAL = WORKSPACE / "lingjing_solo"

os.environ["LINGJING_SRC"] = os.environ.get("LINGJING_SRC", str(CANONICAL))
runpy.run_path(str(STARTER / "scripts" / "build_notebook.py"), run_name="__main__")
