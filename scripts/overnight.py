"""AutoResearch 式通宵循环：选最弱 skill → Trainer → Validator → Keep/Discard。

角色（同一脚本内的四个阶段，AutoResearch 的 Modify→Verify→Keep/Discard→Repeat）：
  Selector  manifest + 账本里挑最弱 skill（gate 离晋级线最远的 active/shadow）
  Trainer   子进程跑 state/train_click_heatmap_v5.py（seed/epochs/HEATMAP_OUT 隔离）
  Validator 扩容评测：HELD 5 局的全部"含 productive 点击"状态（v1 协议只有
            1 入口帧/局，5×10 提案噪声 ±0.15，0.5 晋级线没有判决力——扩容后
            每局最多 30 状态，样本 ×30）；双 GT（引擎真值主判 + 旧屏幕 delta
            辅助），走 arc_adaptor.click_heatmap.propose_from_prob——与 agent
            同一排序/身份/去重代码
  Keeper    keep 判据（全满足）：candidate 扩容 pooled > incumbent 同协议
            分数 + NOISE_FLOOR；主判据 = 引擎真值 GT；产物文件存在。
            keep → candidate 注册 active、incumbent 转 shadow；
            discard → candidate 记 discarded。每次迭代写账本一行。

用法：
  python scripts/overnight.py --validate-only            # 只跑扩容评测
  python scripts/overnight.py --iterations 3 --epochs 100
  python scripts/overnight.py --iterations 1 --epochs 2  # 管道冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging  # noqa: E402

from arc_adaptor import click_heatmap as CH  # noqa: E402

from lingjing_solo.transfer.ledger import (  # noqa: E402
    STATUS_ACTIVE, STATUS_DISCARDED, STATUS_SHADOW, SkillManifest,
    append_record, experiment_row,
)

DATASET = ROOT / "state" / "affordance_dataset.jsonl"
GRIDS = ROOT / "state" / "affordance_grids.npz"
MANIFEST = ROOT / "state" / "skills" / "manifest.json"
LEDGER = ROOT / "state" / "ledger.jsonl"
TRAINER = ROOT / "state" / "train_click_heatmap_v5.py"
CAND_DIR = ROOT / "state" / "candidates"
MECHANISM = {"state_only", "advance", "win"}
HELD = ['r11l-495a7899', 's5i5-18d95033', 'sb26-7fbdac44', 'tu93-0768757b', 'ft09-0d8bbf25']
TOPK, DIST = 10, 2
MAX_STATES = 30          # 每局评测状态上限（确定性取 hash 序前 30）
NOISE_FLOOR = 0.05       # keep 需要超过 incumbent 的最小幅度
SKILL = "click_heatmap_v3"   # 现役（incumbent）skill 名，keep 时被 candidate 顶替


def load_eval_pool():
    """HELD 局 × 含 productive 点击的状态池：{game: [(hash, grid, eff_cells)]}。"""
    grids = {}
    with np.load(GRIDS) as d:
        for h, g in zip(d['hashes'], d['grids']):
            grids[str(h)] = g.astype(np.int64)
    eff = defaultdict(set)
    states = defaultdict(set)
    with open(DATASET, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            gid_full = r.get('game_id', '')
            if gid_full not in HELD:
                continue
            if (r.get('context') or {}).get('kind') != 'click':
                continue
            h = r['grid_hash_before']
            if h not in grids:
                continue
            states[gid_full].add(h)
            if r['outcome'] in MECHANISM:
                a = r['action']
                eff[(gid_full, h)].add((a['y'], a['x']))
    pool = {}
    for game in HELD:
        cases = []
        for h in sorted(states.get(game, ())):
            cells = eff.get((game, h))
            if cells:
                cases.append((h, grids[h], sorted(cells)))
        pool[game] = cases[:MAX_STATES]
    return pool


def score_checkpoint(path: Path, pool):
    """注入权重 → 逐状态 precision@10/recall@10 → 逐局均值 → pooled。"""
    with np.load(path, allow_pickle=True) as d:
        CH._MODEL = {k: d[k] for k in ('Wa', 'Wb', 'Wc')}
    per_game = {}
    for game, cases in pool.items():
        if not cases:
            per_game[game] = None
            continue
        ps, rs = [], []
        for h, g, eff in cases:
            frame = np.clip(g, 0, 15)
            props = CH.propose_clicks(frame, topk=TOPK)
            hits = sum(1 for p in props
                       if any(abs(p['data']['y'] - ey) <= DIST
                              and abs(p['data']['x'] - ex) <= DIST for ey, ex in eff))
            ps.append(hits / max(1, len(props)))
            rs.append(hits / len(eff))
        per_game[game] = (float(np.mean(ps)), float(np.mean(rs)), len(cases))
    vals = [v[0] for v in per_game.values() if v]
    pooled = float(np.mean(vals)) if vals else 0.0
    return pooled, per_game


def report(tag, pooled, per_game):
    print(f"  [{tag}] pooled_p@{TOPK}={pooled:.3f}")
    for game, v in per_game.items():
        if v is None:
            print(f"    {game.split('-')[0]:5s} -- (无 productive 状态)")
        else:
            print(f"    {game.split('-')[0]:5s} p={v[0]:.3f} r={v[1]:.3f} n_states={v[2]}")


def train_candidate(seed: int, epochs: int, out: Path) -> bool:
    env = dict(os.environ, HEATMAP_OUT=str(out))
    r = subprocess.run(
        [sys.executable, str(TRAINER), '--seed', str(seed), '--epochs', str(epochs)],
        cwd=str(ROOT), env=env, capture_output=True, text=True)
    tail = (r.stdout or '').strip().splitlines()[-2:]
    for line in tail:
        print(f"    [trainer] {line}")
    return r.returncode == 0 and out.is_file()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seeds', default='')
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()

    logging.getLogger('arc_agi').setLevel(logging.CRITICAL)
    manifest = SkillManifest(MANIFEST)
    pool = load_eval_pool()
    n_states = sum(len(c) for c in pool.values())
    print(f"扩容评测池：{sum(1 for c in pool.values() if c)}/{len(HELD)} 局，"
          f"{n_states} 个状态（旧协议 5 个入口帧 → ×{max(1, n_states // 5)}）")

    incumbent_path = Path(manifest.skills[SKILL]['path'])
    inc_pooled, inc_detail = score_checkpoint(incumbent_path, pool)
    report(f"incumbent {SKILL}", inc_pooled, inc_detail)

    if args.validate_only:
        return 0

    seeds = ([int(s) for s in args.seeds.split(',') if s.strip()]
             or list(range(1, args.iterations + 1)))
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    best = (inc_pooled, incumbent_path, SKILL)
    for i, seed in enumerate(seeds, 1):
        out = CAND_DIR / f"click_heatmap_v5_seed{seed}.npz"
        print(f"\n── 迭代 {i}/{len(seeds)} seed={seed} → {out.name}")
        t0 = time.time()
        if not train_candidate(seed, args.epochs, out):
            print("    trainer 失败，跳过")
            append_record(LEDGER, experiment_row(
                name=f"heatmap_overnight_it{i}", kind="train",
                config={"seed": seed, "epochs": args.epochs},
                artifact=str(out), verdict="discard", notes="trainer failed"))
            continue
        cand_pooled, cand_detail = score_checkpoint(out, pool)
        report(f"candidate seed{seed}", cand_pooled, cand_detail)
        verdict = "discard"
        notes = f"incumbent={inc_pooled:.3f} candidate={cand_pooled:.3f}"
        if cand_pooled > inc_pooled + NOISE_FLOOR and cand_pooled > best[0]:
            verdict = "keep"
            old_active = manifest.skills.get(SKILL, {}).get('path', '')
            manifest.register(f"click_heatmap_v5_seed{seed}", kind="click_heatmap",
                              path=str(out), version=f"v5-engine-truth-seed{seed}",
                              gate={"expanded_newGT": round(cand_pooled, 3), "line": 0.5},
                              status=STATUS_ACTIVE)
            manifest.register(SKILL, kind="click_heatmap", path=old_active,
                              version=manifest.skills[SKILL].get('version', ''),
                              gate=manifest.skills[SKILL].get('gate', {}),
                              status=STATUS_SHADOW)
            manifest.save()
            best = (cand_pooled, out, f"click_heatmap_v5_seed{seed}")
            notes += " → promoted (incumbent→shadow)"
        else:
            manifest.register(f"click_heatmap_v5_seed{seed}", kind="click_heatmap",
                              path=str(out), version=f"v5-engine-truth-seed{seed}",
                              gate={"expanded_newGT": round(cand_pooled, 3), "line": 0.5},
                              status=STATUS_DISCARDED)
            manifest.save()
        append_record(LEDGER, experiment_row(
            name=f"heatmap_overnight_it{i}", kind="train",
            config={"seed": seed, "epochs": args.epochs,
                    "protocol": "expanded-dual-GT"},
            artifact=str(out),
            metric={"pooled_newGT": round(cand_pooled, 3),
                    "pooled_old_incumbent": round(inc_pooled, 3)},
            gate={"metric": "expanded_newGT", "line": 0.5},
            verdict=verdict, notes=notes))
        print(f"    verdict={verdict}（{time.time() - t0:.0f}s）→ 账本 {LEDGER.name}")

    print(f"\n== 通宵循环结束 ==  最优：{best[2]} pooled={best[0]:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
