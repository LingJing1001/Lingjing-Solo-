"""哈希 / visited / beam 微基准：证明 C++ 探索核可用。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from lingjing_explore import (  # noqa: E402
    Visited,
    beam_top_k,
    hash_bytes,
    version,
)


def main() -> None:
    print("version:", version())
    payload = bytes((i * 17) % 256 for i in range(64 * 64 * 4))  # 多 plane 近似

    t0 = time.perf_counter()
    n = 20000
    h = 0
    for _ in range(n):
        h ^= hash_bytes(payload)
    dt = time.perf_counter() - t0
    print(f"hash_bytes x{n}: {dt:.4f}s  ({n/dt:,.0f} hash/s)  last={h}")

    vis = Visited()
    t0 = time.perf_counter()
    hits = 0
    for i in range(100000):
        if not vis.insert(i * 2654435761 & 0xFFFFFFFFFFFFFFFF):
            hits += 1
    dt = time.perf_counter() - t0
    print(f"visited insert 100k: {dt:.4f}s  size={len(vis)} dup_hits={hits}")

    scores = [float((i * 37) % 997) for i in range(5000)]
    t0 = time.perf_counter()
    top = beam_top_k(scores, 240)
    dt = time.perf_counter() - t0
    print(f"beam_top_k 5000->240: {dt*1000:.3f}ms  top0_score={scores[top[0]]}")
    print("OK — C++ explore core ready for unknown-env search")


if __name__ == "__main__":
    main()
