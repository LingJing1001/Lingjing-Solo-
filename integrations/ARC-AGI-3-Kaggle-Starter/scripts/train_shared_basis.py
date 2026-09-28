"""SSA-E3 离线重训：从转移重放库拟合跨游戏共享基（世界先验）。

完整闭环（白皮书 §7 E3）：
  1) 对局：设 LINCORE_REPLAY_PATH 后 SpectralMind 自动把效应 Δ 追加入重放库
  2) 本脚本：读重放 → 全局 PCA → 写共享基
  3) 下局：设 LINCORE_SHARED_BASIS 后 SpectralCeaxController 自动热启动

用法：
  LINCORE_REPLAY_PATH=... LINCORE_SHARED_BASIS=... \
  python scripts/train_shared_basis.py --k 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingjing_solo.lincore import SharedBasis, TransitionReplay  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replay", default=os.environ.get("LINCORE_REPLAY_PATH", ""))
    p.add_argument("--out", default=os.environ.get("LINCORE_SHARED_BASIS", ""))
    p.add_argument("--k", type=int, default=12)
    args = p.parse_args()
    if not args.replay:
        raise SystemExit("需要 --replay 或环境变量 LINCORE_REPLAY_PATH")
    if not args.out:
        raise SystemExit("需要 --out 或环境变量 LINCORE_SHARED_BASIS")

    rp = TransitionReplay(args.replay)
    M, sigs = rp.load()
    if M.shape[0] < 8:
        raise SystemExit(f"重放样本不足（{M.shape[0]} < 8），先多跑几局再训")
    sb = SharedBasis.fit(M, k=args.k, sigs=sigs)
    if not sb.ready:
        raise SystemExit("共享基拟合失败（样本退化）")
    ok = sb.save(args.out)
    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "replay": args.replay,
        "out": args.out,
        "n_transitions": sb.n_transitions,
        "n_games": sb.n_games,
        "k": int(sb.V.shape[0]) if sb.ready else 0,
        "saved": bool(ok),
        "meta": sb.meta,
    }
    sidecar = Path(args.out).with_suffix(".summary.json")
    sidecar.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
