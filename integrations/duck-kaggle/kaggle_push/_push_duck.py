"""Push Duck kernel with retries; print version number on success."""
from __future__ import annotations

import os
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

PUSH_DIR = Path(__file__).resolve().parent
KERNEL = "liaojiaxuan/lingjing-duck-harness-v1"


def main() -> int:
    for i in range(1, 9):
        print(f"PUSH {i}", flush=True)
        try:
            api = KaggleApi()
            api.authenticate()
            resp = api.kernels_push(str(PUSH_DIR))
            ver = None
            if resp is not None:
                d = resp.to_dict() if hasattr(resp, "to_dict") else getattr(resp, "__dict__", {})
                print("PUSH_RESP", d, flush=True)
                for key in (
                    "versionNumber",
                    "version_number",
                    "currentVersionNumber",
                    "errorMessage",
                    "error_message",
                ):
                    if isinstance(d, dict) and key in d and d[key] is not None:
                        print(key, d[key], flush=True)
                        if "version" in key.lower() and ver is None:
                            ver = d[key]
            time.sleep(2)
            st = api.kernels_status(KERNEL)
            print("STATUS", st, flush=True)
            print("OK", KERNEL, "version", ver, flush=True)
            print(
                "URL",
                f"https://www.kaggle.com/code/{KERNEL}",
                flush=True,
            )
            if ver is not None:
                (PUSH_DIR / "_last_push_version.txt").write_text(str(ver), encoding="utf-8")
            return 0
        except Exception as e:
            print(type(e).__name__, e, flush=True)
            time.sleep(6 + i)
    print("PUSH_EXHAUSTED", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
