"""Push the SELFLEARN_LLM A/B kernels (merged upstream + SmartRouter).

Two notebooks, one variable apart:

  submission_llm0.ipynb  SELFLEARN_LLM=0 -> pure algorithm (engine solver +
                         calibrated plans + CEAX). No vLLM, no model datasets:
                         the run is cheap and fast.
  submission_llm1.ipynb  SELFLEARN_LLM=1 -> the whole-episode Qwen planner
                         (lexicon structured view + in-kernel vLLM).

Both notebooks carry the offline self-benchmark cell, so a plain "Save & Run
All" plays the competition's 25 public environment files and writes
/kaggle/working/BENCH_SELFLEARN_{0,1}.json.

Separate slugs (never the teammate's smart-router kernel):
  liaojiaxuan/smartrouter-merged-ab-llm0
  liaojiaxuan/smartrouter-merged-ab-llm1

Run with the starter venv python (has the kaggle package):
    .venv\Scripts\python.exe scripts\push_ab_kernels.py
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

LLM_DATASETS = [
    "bilgisarmal/vllm031-cp313-wheelhouse",
    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
]
COMPETITION = "arc-prize-2026-arc-agi-3"

ARMS = [
    # (arm, notebook file, slug, title, datasets, needs_gpu)
    # llm0 is pure algorithm (engine solver + calibrated plans + CEAX): the
    # whole path is CPU, so it runs without burning GPU quota. That matters --
    # the account's 30 h/week GPU quota is currently exhausted.
    ("llm0", "submission_llm0.ipynb", "liaojiaxuan/smartrouter-merged-ab-llm0",
     "smartrouter-merged-ab-llm0", [], False),
    ("llm1", "submission_llm1.ipynb", "liaojiaxuan/smartrouter-merged-ab-llm1",
     "smartrouter-merged-ab-llm1", LLM_DATASETS, True),
]


def prepare(arm: str, nb_name: str, slug: str, title: str, datasets: list[str],
            needs_gpu: bool = True) -> Path:
    push_dir = NOTEBOOKS / f"_push_ab_{arm}"
    if push_dir.exists():
        shutil.rmtree(push_dir)
    push_dir.mkdir(parents=True)
    shutil.copy2(NOTEBOOKS / nb_name, push_dir / "submission.ipynb")
    meta = {
        "id": slug,
        "title": title,
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": needs_gpu,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": datasets,
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    if needs_gpu:
        meta["machine_shape"] = "NvidiaRtxPro6000"
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return push_dir


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    only = str(os.environ.get("LINGJING_AB_ARM", "")).strip()
    force_gpu = os.environ.get("LINGJING_AB_GPU", "").strip()

    api = KaggleApi()
    api.authenticate()

    rc = 0
    for arm, nb_name, slug, title, datasets, needs_gpu in ARMS:
        if only and only != arm:
            continue
        if force_gpu == "1":
            needs_gpu = True
        elif force_gpu == "0":
            needs_gpu = False
        push_dir = prepare(arm, nb_name, slug, title, datasets, needs_gpu)
        print(f"=== PUSH {arm} -> {slug} (gpu={needs_gpu}) ===", flush=True)
        ok = False
        for attempt in range(1, 4):
            try:
                resp = api.kernels_push(str(push_dir))
                ver = None
                if resp is not None:
                    d = resp.to_dict() if hasattr(resp, "to_dict") else getattr(resp, "__dict__", {})
                    print("PUSH_RESP", d, flush=True)
                    if isinstance(d, dict) and d.get("error"):
                        print("PUSH_ERROR", d["error"], flush=True)
                        raise RuntimeError(str(d["error"]))
                    for key in ("versionNumber", "version_number", "currentVersionNumber"):
                        if isinstance(d, dict) and d.get(key) is not None:
                            ver = d[key]
                            break
                time.sleep(2)
                st = api.kernels_status(slug)
                print("STATUS", st, flush=True)
                print("OK", slug, "version", ver, flush=True)
                print("URL", f"https://www.kaggle.com/code/{slug}", flush=True)
                (push_dir / "_last_version.txt").write_text(str(ver), encoding="utf-8")
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"attempt {attempt} failed:", type(exc).__name__, exc, flush=True)
                time.sleep(5 + attempt)
        if not ok:
            print(f"PUSH_FAILED {arm}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
