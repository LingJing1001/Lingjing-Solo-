"""Download Duck kernel output with long backoff."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

OUT = Path(__file__).resolve().parents[1] / "_err_out_conc8"
KERNEL = "liaojiaxuan/lingjing-duck-harness-v1"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(1, 21):
        print(f"TRY {i}", flush=True)
        try:
            api = KaggleApi()
            api.authenticate()
            api.kernels_output(KERNEL, str(OUT))
            print("LOGS_OK", flush=True)
            for p in sorted(OUT.rglob("*")):
                if p.is_file():
                    print(p.name, p.stat().st_size, flush=True)
            return 0
        except Exception as e:
            print(type(e).__name__, flush=True)
            time.sleep(min(30, 4 + i * 2))
    print("LOGS_FAIL", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
