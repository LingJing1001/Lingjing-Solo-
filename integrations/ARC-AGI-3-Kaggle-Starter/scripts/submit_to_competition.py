"""Submit a code-competition submission programmatically (no UI needed).

`competition_submit_code` is the API the "Submit to Competition" button ends
up calling: it hands the competition a <owner>/<slug> + version number, and
Kaggle re-runs that notebook *inside the competition rerun* (that is where
`KAGGLE_IS_COMPETITION_RERUN` becomes truthy, the gateway sidecar appears at
http://gateway:8001, and the real submission.parquet is produced by the
gateway, not by the notebook).

Guard rails baked in, because both quotas are hard walls:
  * the kernel must be COMPLETE (a RUNNING/ERROR kernel cannot be submitted);
  * the daily submission quota is checked before the call;
  * `--dry-run` prints the plan without spending the quota.

Usage (starter venv python):
    .venv\\Scripts\\python.exe scripts\\submit_to_competition.py \\
        --kernel liaojiaxuan/smartrouter-merged-cpu-submit --version 1 \\
        --message "CPU fallback: SELFLEARN_LLM=0 + scripts on"

    ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time

COMPETITION = "arc-prize-2026-arc-agi-3"
FILE_NAME = "submission.parquet"


def _status_str(st) -> str:
    """kernels_status() returns a dict whose 'status' may be a str or an enum."""
    s = st.get("status") if isinstance(st, dict) else getattr(st, "status", st)
    return str(getattr(s, "name", s) or s)


def limits(api):
    try:
        lim = api.competition_get_submission_limits(COMPETITION)
        return lim.to_dict() if hasattr(lim, "to_dict") else vars(lim)
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="owner/slug")
    ap.add_argument("--version", type=int, default=0, help="kernel version number")
    ap.add_argument("--message", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait-seconds", type=int, default=0,
                    help="poll the kernel until it leaves RUNNING (0 = don't wait)")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    if args.wait_seconds > 0:
        deadline = time.time() + args.wait_seconds
        while time.time() < deadline:
            s = _status_str(api.kernels_status(args.kernel))
            print(f"  kernel status = {s}", flush=True)
            if s not in ("RUNNING", "PENDING"):
                break
            time.sleep(20)

    st = api.kernels_status(args.kernel)
    status = _status_str(st)
    failure = st.get("failureMessage") if isinstance(st, dict) else getattr(st, "failureMessage", None)
    lim = limits(api)

    print("=" * 78)
    print("competition :", COMPETITION)
    print("kernel      :", args.kernel, "version", args.version)
    print("status      :", status, ("failure=" + str(failure)) if failure else "")
    print("limits      :", json.dumps(lim, ensure_ascii=False, default=str))
    print("=" * 78)

    if status != "COMPLETE":
        print(f"REFUSE: kernel is {status}, not COMPLETE -- a non-complete "
              "notebook cannot produce a competition submission.", flush=True)
        return 2

    allowed = lim.get("numAllowedNow")
    if allowed is not None and not allowed:
        print(f"REFUSE: submission quota says numAllowedNow={allowed} "
              f"(numToday={lim.get('numToday')}, numTotal={lim.get('numTotal')}) -- "
              "the daily quota is exhausted; retry after 00:00 UTC.", flush=True)
        return 3

    if args.dry_run:
        print("DRY-RUN: would call competition_submit_code("
              f"file_name={FILE_NAME!r}, kernel={args.kernel!r}, "
              f"kernel_version={args.version}, competition={COMPETITION!r})")
        return 0

    resp = api.competition_submit_code(
        file_name=FILE_NAME,
        message=args.message,
        competition=COMPETITION,
        kernel=args.kernel,
        kernel_version=args.version or None,
    )
    d = resp.to_dict() if hasattr(resp, "to_dict") else getattr(resp, "__dict__", {})
    print("SUBMIT_RESP", json.dumps(d, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
