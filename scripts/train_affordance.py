"""Affordance 管线入口：收割（可选）→ 数据统计 → LOGO 训练评测。

用法：
  python scripts/train_affordance.py                 # 数据集在就复用，否则先收割
  python scripts/train_affordance.py --mine          # 强制重新收割
  python scripts/train_affordance.py --no-walk       # 收割时跳过随机游走
  python scripts/train_affordance.py --min-clicks 10 # LOGO 留出游戏最少点击数

输出：
  state/affordance_dataset.jsonl   统一 StepRecord 数据集
  stdout                           数据统计 + LOGO 评测表
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingjing_solo.transfer.affordance import (  # noqa: E402
    AffordanceRanker, click_feature_dict, color_prior_train, logo_eval, record_label,
)
from lingjing_solo.transfer.miner import DEFAULT_OUT  # noqa: E402
from lingjing_solo.transfer.trajectory import (  # noqa: E402
    outcome_distribution, read_jsonl,
)


def _load_harvest():
    """收割器在边界侧 scripts/harvest_trajectories.py（引擎边界规则），
    scripts/ 不是包，按路径加载。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "harvest_trajectories", Path(__file__).resolve().parent / "harvest_trajectories.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def print_stats(records) -> None:
    dist = outcome_distribution(records)
    total = len(records)
    print(f"\n== 数据集 ==\n总记录 {total}（JSONL: {DEFAULT_OUT.name}）")
    print("结局分布: " + "  ".join(f"{k}={v}" for k, v in dist.items()))
    by_src: dict = {}
    for r in records:
        s = by_src.setdefault(r.source, [0, 0])
        s[0] += 1
        s[1] += 1 if click_feature_dict(r.context) is not None else 0
    print("来源      记录数   其中点击")
    for s, (n, c) in sorted(by_src.items()):
        print(f"  {s:12s} {n:6d} {c:8d}")
    games = sorted({r.gid for r in records})
    print(f"游戏数 {len(games)}: {' '.join(games)}")
    n_probe = sum(1 for r in records if r.is_probe)
    print(f"探针记录 {n_probe}（反事实引擎真值标签）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", action="store_true", help="强制重新收割")
    ap.add_argument("--no-walk", action="store_true", help="收割时跳过随机游走")
    ap.add_argument("--min-clicks", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--adapt-shots", type=int, default=8,
                    help="在线校准 shot 数（live agent 本局已观测的结局数）")
    args = ap.parse_args()

    if args.mine or not DEFAULT_OUT.is_file():
        t0 = time.time()
        print(f"== 收割 → {DEFAULT_OUT} ==")
        summary = _load_harvest().harvest_all(do_walk=not args.no_walk)
        print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
        print(f"收割用时 {time.time() - t0:.0f}s")
    records = read_jsonl(DEFAULT_OUT)
    if not records:
        print("数据集为空，退出")
        return 1
    print_stats(records)

    def run_logo(k: int) -> None:
        per = logo_eval(records, min_test_clicks=args.min_clicks,
                        epochs=args.epochs, adapt_shots=k)
        print(f"\n== LOGO 评测（每局留出，其余训练；epochs={args.epochs}，"
              f"k-shot={k}）==")
        hdr = (f"{'game':8s} {'nclk':>5s} {'shot':>4s} {'pair':>4s} "
               f"{'auc_m':>6s} {'auc_clr':>7s} {'top1_m':>6s} {'top1_clr':>8s}")
        print(hdr)
        rows = [(g, m) for g, m in per.items() if g != "pooled"]
        rows.sort(key=lambda kv: kv[1].get("auc_model", -1))
        for g, m in rows:
            if m.get("skipped"):
                print(f"{g:8s} {int(m['n_clicks']):5d}   (点击太少，跳过)")
                continue
            print(f"{g:8s} {int(m['n_clicks']):5d} {int(m.get('n_shots', 0)):4d} "
                  f"{int(m.get('n_pairs', 0)):4d} "
                  f"{m['auc_model']:6.3f} {m['auc_color_prior']:7.3f} "
                  f"{m['top1_model']:6.3f} {m['top1_color_prior']:8.3f}")
        p = per.get("pooled", {})
        if "auc_model" in p:
            print(f"{'pooled':8s} {'':5s} {'':4s} {int(p.get('n_pairs', 0)):4d} "
                  f"{p['auc_model']:6.3f} {'':7s} {p['top1_model']:6.3f}")
        return per

    run_logo(0)
    run_logo(args.adapt_shots)

    # 全量拟合一个部署模型，存权重摘要供 agent 侧复用
    model = AffordanceRanker(epochs=args.epochs).fit(records)
    out_w = ROOT / "state" / "affordance_ranker_weights.json"
    out_w.write_text(json.dumps({
        "keys": model.keys,
        "has_effect": model.models["has_effect"].w.tolist(),
        "productive": model.models["productive"].w.tolist(),
        "n_records": len(records),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n部署模型权重 → {out_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
