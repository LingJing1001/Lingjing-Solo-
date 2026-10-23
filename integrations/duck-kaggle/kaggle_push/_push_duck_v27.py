"""Push the Duck v27 kernel (separate slug; the submitted v25 kernel stays intact).

Prerequisite: patch_cp313_v27.py has regenerated
kaggle_push/push_v27/lingjing-duck-harness-v27.ipynb + kernel-metadata.json,
and the overlay dataset has been re-versioned with the v27 solver.py.

Run with the starter venv python (has the kaggle package), same as _push_duck.py.
"""
from __future__ import annotations

import time
from pathlib import Path

PUSH_DIR = Path(__file__).resolve().parent / "push_v27"
KERNEL = "liaojiaxuan/lingjing-duck-harness-v27"


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

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
                for key in ("versionNumber", "version_number", "currentVersionNumber"):
                    if isinstance(d, dict) and d.get(key) is not None:
                        print(key, d[key], flush=True)
                        if ver is None:
                            ver = d[key]
            time.sleep(2)
            st = api.kernels_status(KERNEL)
            print("STATUS", st, flush=True)
            print("OK", KERNEL, "version", ver, flush=True)
            print("URL", f"https://www.kaggle.com/code/{KERNEL}", flush=True)
            if ver is not None:
                (PUSH_DIR / "_last_push_version_v27.txt").write_text(str(ver), encoding="utf-8")
            return 0
        except Exception as e:
            print(type(e).__name__, e, flush=True)
            time.sleep(6 + i)
    print("PUSH_EXHAUSTED", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
