#!/usr/bin/env python3
"""Gate a competition submit on the Phase B fingerprint.

Static checks (notebook + metadata + agent) can pass on this machine.
In-kernel Qwen startup cannot. Without a kernel log that contains
VLLM_READY True, VLLM_PROBE_OK and PHASE_B_OK, this script exits 2
and the submission must not be sent.

  python scripts/check_phase_b_fingerprint.py
  python scripts/check_phase_b_fingerprint.py --kernel-log path/to/kernel.log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent" / "my_agent.py"
META = ROOT / "notebooks" / "kernel-metadata.json"
NOTEBOOK = ROOT / "notebooks" / "submission.ipynb"


def _fail(msg: str) -> None:
    print(f"FINGERPRINT_FAIL {msg}")
    raise SystemExit(1)


def static_check() -> None:
    if not AGENT.is_file() or not META.is_file() or not NOTEBOOK.is_file():
        _fail("missing agent, kernel-metadata, or submission notebook")
    agent = AGENT.read_text(encoding="utf-8")
    if "llm-whole-qwen" not in agent or "smart-router" not in agent:
        _fail("BUILD_TAG missing smart-router or llm-whole-qwen")
    meta = json.loads(META.read_text(encoding="utf-8"))
    if meta.get("enable_internet") is not False:
        _fail(f"enable_internet={meta.get('enable_internet')!r}")
    if meta.get("machine_shape") != "NvidiaRtxPro6000":
        _fail(f"machine_shape={meta.get('machine_shape')!r}")
    sources = meta.get("dataset_sources") or []
    if not any("vllm" in str(s) for s in sources) or not any("qwen" in str(s) for s in sources):
        _fail(f"dataset_sources={sources}")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    kaggle = (nb.get("metadata") or {}).get("kaggle") or {}
    if kaggle.get("isInternetEnabled") is not False:
        _fail("notebook isInternetEnabled is not false")
    blob = "\n".join(
        "".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
        for c in nb.get("cells") or []
    )
    for needle in (
        "llm-whole-qwen",
        "VLLM_NOT_READY",
        "VLLM_PROBE_OK",
        "PHASE_B_FINGERPRINT",
        "PHASE_B_OK",
        "LINGJING_LLM_WHOLE=on",
        "LINGJING_LLM_MAX_CALLS=800",
    ):
        if needle not in blob:
            _fail(f"notebook missing {needle}")
    print("FINGERPRINT_STATIC_OK")
    print("BUILD_TAG_REQUIRES llm-whole-qwen")
    print("enable_internet false")
    print(f"machine_shape {meta.get('machine_shape')}")
    print(f"datasets {sources}")


def kernel_check(log_path: Path) -> None:
    if not log_path.is_file():
        print(f"FINGERPRINT_FAIL kernel log missing: {log_path}")
        raise SystemExit(2)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    needed = ("VLLM_READY True", "VLLM_PROBE_OK", "PHASE_B_OK", "llm-whole-qwen")
    missing = [n for n in needed if n not in text]
    if missing or "VLLM_SKIP" in text or "VLLM_NOT_READY" in text:
        print(f"FINGERPRINT_FAIL kernel log missing or failed: {missing or 'vllm not ready'}")
        raise SystemExit(2)
    print("FINGERPRINT_KERNEL_OK")
    print("SUBMIT_ALLOWED")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel-log", type=Path, default=None)
    args = p.parse_args()
    static_check()
    if args.kernel_log is None:
        print("SUBMIT_BLOCKED no in-kernel log yet (need VLLM_READY True, VLLM_PROBE_OK, PHASE_B_OK)")
        return 2
    kernel_check(args.kernel_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
