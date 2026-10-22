"""Push a competition submission kernel (CPU fallback arm or GPU LLM arm).

Why this exists
---------------
The account's weekly GPU quota (30 h/week) is exhausted and only refreshes at
2026-10-24T00:00:00Z, while the competition's daily submission quota resets at
00:00 UTC. So the 10-23 window is a CPU-only window: the submission made there
must not need a GPU. The 10-24 window has the GPU back and is where the LLM
arm goes.

Both arms carry the full competition cell
(`KAGGLE_IS_COMPETITION_RERUN` -> wait for the gateway, copy the framework,
register MyAgent, run `python main.py --agent myagent`) and the dummy
submission cell, so they are valid competition submissions. Neither carries
the offline benchmark cell: the competition rerun does not need it, and every
extra cell is runtime the 9-hour competition clock has to pay for.

Arms
----
  cpu : SELFLEARN_LLM=0, enable_gpu=false, no model datasets -> zero GPU quota
  llm : SELFLEARN_LLM=1, enable_gpu=true (RTX Pro 6000), vLLM wheelhouse +
        Qwen3.6-27B-FP8 snapshot attached

Separate slugs -- never the teammate's kernel
(`wangausx/arc-prize-2026-smart-router-v2-dpl`):
  cpu -> liaojiaxuan/smartrouter-merged-cpu-submit
  llm -> liaojiaxuan/smartrouter-merged-llm-submit

Usage (starter venv python, has the kaggle package):
    .venv\\Scripts\\python.exe scripts\\push_submit_kernel.py --arm cpu
    .venv\\Scripts\\python.exe scripts\\push_submit_kernel.py --arm llm
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

COMPETITION = "arc-prize-2026-arc-agi-3"
LLM_DATASETS = [
    "bilgisarmal/vllm031-cp313-wheelhouse",
    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
]

ARMS = {
    "cpu": {
        "nb": "submission_cpu.ipynb",
        "slug": "liaojiaxuan/smartrouter-merged-cpu-submit",
        "title": "smartrouter-merged-cpu-submit",
        "datasets": [],
        "gpu": False,
    },
    "llm": {
        "nb": "submission_llm.ipynb",
        "slug": "liaojiaxuan/smartrouter-merged-llm-submit",
        "title": "smartrouter-merged-llm-submit",
        "datasets": LLM_DATASETS,
        "gpu": True,
    },
}

BUILD_HINT = """build it first:
  $env:LINGJING_KAGGLE_ACCEL='{accel}'; $env:LINGJING_SELFLEARN_LLM='{selflearn}';
  $env:LINGJING_BENCH='0'; $env:LINGJING_SKIP_METADATA_SYNC='1';
  $env:LINGJING_NOTEBOOK_OUT='<starter>\\notebooks\\{nb}';
  .venv\\Scripts\\python.exe scripts\\build_notebook.py"""


def prepare(arm: str) -> Path:
    spec = ARMS[arm]
    nb_src = NOTEBOOKS / spec["nb"]
    if not nb_src.is_file():
        raise SystemExit(
            f"missing {nb_src}\n"
            + BUILD_HINT.format(
                accel="rtx6000" if spec["gpu"] else "cpu",
                selflearn="1" if arm == "llm" else "0",
                nb=spec["nb"],
            )
        )
    push_dir = NOTEBOOKS / f"_push_submit_{arm}"
    if push_dir.exists():
        shutil.rmtree(push_dir)
    push_dir.mkdir(parents=True)
    shutil.copy2(nb_src, push_dir / "submission.ipynb")
    meta = {
        "id": spec["slug"],
        "title": spec["title"],
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": spec["gpu"],
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": list(spec["datasets"]),
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    if spec["gpu"]:
        meta["machine_shape"] = "NvidiaRtxPro6000"
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return push_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default="cpu")
    args = ap.parse_args()
    spec = ARMS[args.arm]

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    push_dir = prepare(args.arm)
    print(f"=== PUSH {args.arm} -> {spec['slug']} (gpu={spec['gpu']}) ===", flush=True)
    print(f"  dir  : {push_dir}")
    print(f"  nb   : {spec['nb']} ({(NOTEBOOKS / spec['nb']).stat().st_size} bytes)", flush=True)

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
            print("STATUS", api.kernels_status(spec["slug"]), flush=True)
            (push_dir / "_last_version.txt").write_text(str(version), encoding="utf-8")
            print(f"OK {spec['slug']} version {version}", flush=True)
            print(f"URL https://www.kaggle.com/code/{spec['slug']}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"attempt {attempt} failed: {type(exc).__name__} {exc}", flush=True)
            time.sleep(5 + attempt)
    print("PUSH_FAILED", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
