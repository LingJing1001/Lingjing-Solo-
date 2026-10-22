"""Push the CPU fallback submission kernel (no GPU, no model datasets).

Why this exists
---------------
The account's GPU quota (30 h/week) is exhausted and only refreshes at
2026-10-24T00:00:00Z. The competition's daily submission quota resets at
00:00 UTC. So the 10-23 window is a CPU-only window: a submission made there
must not need a GPU.

This kernel is:
  * SELFLEARN_LLM=0  -> pure algorithm route (engine solver + calibrated plans
    + CEAX). No model server, no vLLM wheelhouse, no model snapshot.
  * LINGJING_SCRIPTS=on -> the calibrated in-tree plan floor stays enabled
    (it is already validated by replay; it is not a new per-game hack).
  * enable_gpu=false, dataset_sources=[] -> consumes zero GPU quota.
  * internet disabled (competition rule).

It carries the full competition cell (`KAGGLE_IS_COMPETITION_RERUN` ->
wait for the gateway, copy the framework, register MyAgent, run
`python main.py --agent myagent`) and the dummy-submission cell, so it is a
valid competition submission -- unlike the A/B build, it has no offline
benchmark cell (the rerun does not need one, and every extra cell is runtime
the competition clock has to pay for).

Separate slug -- never the teammate's kernel:
  liaojiaxuan/smartrouter-merged-cpu-submit

Usage (starter venv python, has the kaggle package):
    .venv\\Scripts\\python.exe scripts\\push_cpu_submit_kernel.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

COMPETITION = "arc-prize-2026-arc-agi-3"
SLUG = "liaojiaxuan/smartrouter-merged-cpu-submit"
TITLE = "smartrouter-merged-cpu-submit"
NB_SRC = NOTEBOOKS / "submission_cpu.ipynb"
PUSH_DIR = NOTEBOOKS / "_push_cpu_submit"


def prepare() -> Path:
    if not NB_SRC.is_file():
        raise SystemExit(
            f"missing {NB_SRC}\n"
            "build it first:\n"
            "  $env:LINGJING_KAGGLE_ACCEL='cpu'; $env:LINGJING_SELFLEARN_LLM='0'; "
            "$env:LINGJING_BENCH='0'; $env:LINGJING_SKIP_METADATA_SYNC='1'; "
            "$env:LINGJING_NOTEBOOK_OUT='...\\notebooks\\submission_cpu.ipynb'; "
            "python scripts\\build_notebook.py"
        )
    if PUSH_DIR.exists():
        shutil.rmtree(PUSH_DIR)
    PUSH_DIR.mkdir(parents=True)
    shutil.copy2(NB_SRC, PUSH_DIR / "submission.ipynb")
    meta = {
        "id": SLUG,
        "title": TITLE,
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    (PUSH_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return PUSH_DIR


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    push_dir = prepare()
    print(f"=== PUSH CPU submission kernel -> {SLUG} ===", flush=True)
    print(f"  dir  : {push_dir}")
    print(f"  nb   : {NB_SRC.name} ({NB_SRC.stat().st_size} bytes)", flush=True)

    version = None
    for attempt in range(1, 4):
        try:
            resp = api.kernels_push(str(push_dir))
            d = resp.to_dict() if hasattr(resp, "to_dict") else getattr(resp, "__dict__", {})
            print("PUSH_RESP", d, flush=True)
            if isinstance(d, dict) and d.get("error"):
                raise RuntimeError(str(d["error"]))
            for key in ("versionNumber", "version_number", "currentVersionNumber"):
                if isinstance(d, dict) and d.get(key) is not None:
                    version = d[key]
                    break
            time.sleep(2)
            print("STATUS", api.kernels_status(SLUG), flush=True)
            (push_dir / "_last_version.txt").write_text(str(version), encoding="utf-8")
            print(f"OK {SLUG} version {version}", flush=True)
            print(f"URL https://www.kaggle.com/code/{SLUG}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"attempt {attempt} failed: {type(exc).__name__} {exc}", flush=True)
            time.sleep(5 + attempt)
    print("PUSH_FAILED", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
