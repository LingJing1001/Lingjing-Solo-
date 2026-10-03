"""Autosearch: 自动搜索 ConvVariant 超参空间，找最优配置。

策略：
  1. 随机搜索 N 次试验，每次 2-fold 快速评估（~40s/次）
  2. 取 top-5，用 5-fold 严格验证（~90s/次）
  3. 报告最优，对比热图基线 P@8=0.293

搜索空间：
  variant:     base, wide, attention, dilated
  channels:    (12,24,32), (24,48,64), (16,32,48), (32,64,96)
  embed_dim:   32, 64, 128
  patch_radius: 5, 7, 11
  lr:          1e-3, 3e-3, 5e-3
  temp:        0.1, 0.2, 0.3
  steps:       500

用法：.venv/Scripts/python.exe autosearch.py [--budget 20]
"""
import sys, time, io, contextlib, argparse, random, json
from pathlib import Path
sys.path.insert(0, "F:/pro2")
import numpy as np
import torch
from lingjing_solo.neural.conv_variants import build_variant
from lingjing_solo.neural.click_encoder import (
    load_pool, train_click_encoder, propose_clicks_encoder, split_calibration,
)
from lingjing_solo.neural.eval_click_encoder import FILES, hits
from arc_adaptor import click_heatmap as CH

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:.1f}s] {msg}", flush=True)

# ───────────────────────── 搜索空间 ─────────────────────────
SEARCH_SPACE = {
    "variant":      ["base", "wide", "attention", "dilated"],
    "channels":     [(12, 24, 32), (24, 48, 64), (16, 32, 48), (32, 64, 96)],
    "embed_dim":    [32, 64, 128],
    "patch_radius": [5, 7, 11],
    "lr":           [1e-3, 3e-3, 5e-3],
    "temp":         [0.1, 0.2, 0.3],
    "steps":        [500],
}

def sample_config(rng):
    return {
        "variant":      rng.choice(SEARCH_SPACE["variant"]),
        "channels":     rng.choice(SEARCH_SPACE["channels"]),
        "embed_dim":    rng.choice(SEARCH_SPACE["embed_dim"]),
        "patch_radius": rng.choice(SEARCH_SPACE["patch_radius"]),
        "lr":           rng.choice(SEARCH_SPACE["lr"]),
        "temp":         rng.choice(SEARCH_SPACE["temp"]),
        "steps":        rng.choice(SEARCH_SPACE["steps"]),
    }

# ───────────────────────── 评估 ─────────────────────────
def make_folds(all_games, k=5):
    n = len(all_games)
    fs = (n + k - 1) // k
    return [all_games[i*fs:(i+1)*fs] for i in range(k)]

def eval_game(g, pool, enc, patch_radius, topk=8):
    frame = pool["frame_grid"][g]
    pts = pool["points"][g]
    eff = pool["eff_cells"].get(g, [])
    calib_pos, calib_neg, truth = split_calibration(eff, pts)
    props = propose_clicks_encoder(frame, enc, pts, calib_pos, calib_neg,
                                   patch_radius=patch_radius, topk=topk)
    cells = [(p["data"]["x"], p["data"]["y"]) for p in props]
    n_hit = sum(hits(c, eff) for c in cells)
    covered = sum(any(hits(e, cells) for e in eff) for e in eff)
    target = truth if truth else eff
    return n_hit / max(1, len(cells)), covered / max(1, len(target))

def kfold_eval(config, pool, all_games, folds_k=5, patch_radius_override=None):
    """跑 k-fold，返回 (mean_P@8, mean_R@8, std_P@8)。"""
    folds = make_folds(all_games, k=folds_k)
    pr = patch_radius_override or config["patch_radius"]
    all_p, all_r = [], []
    for fi in range(folds_k):
        heldout = folds[fi]
        train_games = [g for g in all_games if g not in heldout]
        train_pool = load_pool(FILES, games=train_games)
        enc = build_variant(config["variant"], seed=42,
                            num_colors=16, embed_dim=config["embed_dim"],
                            channels=config["channels"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train_click_encoder(enc, train_pool, steps=config["steps"],
                                lr=config["lr"], temp=config["temp"],
                                verbose=True, patience=0)
        for g in heldout:
            if g not in pool["frame_grid"]:
                continue
            p, r = eval_game(g, pool, enc, pr)
            all_p.append(p); all_r.append(r)
    return float(np.mean(all_p)), float(np.mean(all_r)), float(np.std(all_p))

def heatmap_baseline(pool, all_games):
    ps, rs = [], []
    for g in all_games:
        frame = pool["frame_grid"][g]
        eff = pool["eff_cells"].get(g, [])
        props = CH.propose_clicks(frame, topk=8)
        cells = [(p["data"]["x"], p["data"]["y"]) for p in props]
        n_hit = sum(hits(c, eff) for c in cells)
        covered = sum(any(hits(e, cells) for e in eff) for e in eff)
        ps.append(n_hit / max(1, len(cells)))
        rs.append(covered / max(1, len(eff)))
    return float(np.mean(ps)), float(np.mean(rs))

# ───────────────────────── 主流程 ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=20, help="随机搜索试验数")
    ap.add_argument("--topk", type=int, default=5, help="初筛后 5-fold 验证数")
    ap.add_argument("--out", default="state/autosearch_result.json")
    args = ap.parse_args()

    log("load_pool ...")
    pool = load_pool(FILES)
    all_games = sorted(pool["frame_grid"].keys())
    log(f"patches={len(pool['labels'])} eff={int((pool['labels']==1).sum())} games={len(all_games)}")

    log("heatmap baseline ...")
    hb_p, hb_r = heatmap_baseline(pool, all_games)
    print(f"  热图基线: P@8={hb_p:.3f}  R@8={hb_r:.3f}", flush=True)

    # Phase 1: 随机搜索 + 2-fold 快速初筛
    log(f"Phase 1: 随机搜索 {args.budget} 次，2-fold 初筛 ...")
    rng = random.Random(42)
    results = []
    for i in range(args.budget):
        cfg = sample_config(rng)
        tag = f"{cfg['variant'][:4]}_ch{cfg['channels'][0]}_ed{cfg['embed_dim']}_pr{cfg['patch_radius']}_lr{cfg['lr']:.0e}_t{cfg['temp']}"
        log(f"  [{i+1}/{args.budget}] {tag}")
        try:
            mp, mr, sp = kfold_eval(cfg, pool, all_games, folds_k=2)
        except Exception as e:
            log(f"    FAIL: {type(e).__name__}: {e}")
            mp, mr, sp = -1.0, -1.0, 1.0
        results.append((cfg, mp, mr, sp, tag))
        print(f"    P@8={mp:.3f}  R@8={mr:.3f}", flush=True)

    # Phase 2: top-K 5-fold 验证
    results.sort(key=lambda x: -x[1])
    top = results[:args.topk]
    log(f"Phase 2: top-{args.topk} 5-fold 验证 ...")
    print(f"\n{'rank':>4s} {'config':>50s} {'2fold-P@8':>10s} {'5fold-P@8':>10s} {'5fold-R@8':>10s}", flush=True)
    print("-" * 90)
    verified = []
    for rank, (cfg, p2, r2, s2, tag) in enumerate(top, 1):
        log(f"  verify rank={rank} {tag}")
        mp, mr, sp = kfold_eval(cfg, pool, all_games, folds_k=5)
        verified.append((cfg, mp, mr, sp, tag))
        print(f"{rank:4d} {tag:>50s} {p2:10.3f} {mp:10.3f} {mr:10.3f}", flush=True)

    verified.sort(key=lambda x: -x[1])
    best = verified[0]
    print(f"\n=== 最优配置 ===")
    print(f"  {best[4]}")
    print(f"  P@8={best[1]:.3f}  R@8={best[2]:.3f}  std={best[3]:.3f}")
    print(f"  vs 热图基线 P@8={hb_p:.3f}  Δ={best[1]-hb_p:+.3f}")
    if best[1] > hb_p:
        print("  ✅ 超越热图基线！")
    else:
        print("  ❌ 未超越热图基线")

    # ! 保存结果
    out = Path("F:/pro2") / args.out
    payload = {
        "heatmap_baseline": {"P@8": hb_p, "R@8": hb_r},
        "best": {
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best[0].items()},
            "P@8": best[1], "R@8": best[2], "std": best[3],
        },
        "all_verified": [
            {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
             "P@8": p, "R@8": r, "std": s, "tag": t}
            for c, p, r, s, t in verified
        ],
        "budget": args.budget,
        "elapsed_s": round(time.time() - T0, 1),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存 → {out}")
    log("DONE")

if __name__ == "__main__":
    main()
