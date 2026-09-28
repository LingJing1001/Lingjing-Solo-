#!/usr/bin/env python3
"""Sync the canonical Lingjing package and ARC submission agent.

The source of truth is ../lingjing_solo. Both integrations resolve it through
LINGJING_SRC; no duplicate package directory is maintained in either checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "lingjing_solo"
ARC = ROOT / "integrations" / "ARC-AGI-3"
STARTER = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify consumers match canonical source")
    args = parser.parse_args()

    required = (CANONICAL, ARC, STARTER)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing integration path(s): " + ", ".join(missing))

    agent_source = ARC / "agent" / "my_agent.py"
    agent_destination = STARTER / "agent" / "my_agent.py"
    if not agent_source.is_file():
        raise SystemExit(f"missing ARC submission agent: {agent_source}")

    if args.check:
        for duplicate in (ARC / "lingjing_solo", STARTER / "lingjing_solo"):
            if duplicate.exists():
                raise SystemExit(f"duplicate package must be absent: {duplicate}")
        print(f"sync check: canonical package is {CANONICAL}")
        print(f"submission agent: {sha256(agent_source)}")
        return 0

    if (ARC / "lingjing_solo").exists() or (STARTER / "lingjing_solo").exists():
        raise SystemExit("remove duplicate integration packages before syncing")
    shutil.copy2(agent_source, agent_destination)
    print(f"canonical: {CANONICAL}")
    print(f"starter agent: {agent_destination}")
    print(f"agent sha256: {sha256(agent_destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
