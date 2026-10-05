"""Train click-heatmap v5: engine-truth labels from the affordance dataset.

对 v3/v4 病根清单（docs/热图提议器优化方案.md）的逐条回应：
  1. 标签毒化 → productive（state_only/advance/win，引擎真值，来自
     state/affordance_dataset.jsonl）。v1-v4 的 n_changed>=3 是屏幕 delta 标注，
     把 state_only（纯状态变化、屏幕不动）正样本当负样本——毒化源本身。
  2. 数据薄 → 2221 条点击记录（探针 537 / 游走 1040 / 计划 593 / ...），
     正样本约数百条 vs v1 的 162。
  3. 配对 → margin ranking 在同基态（同 grid_hash）内配对。部署形态就是
     "同一屏幕上选一个点"；v4 文档承认 per-frame pairing 没做，这里做了。
  4. 可靠性 → loss 按来源加权（probe 0.5 / walk 0.6 / route-plan-solution 1.0）。
  5. 选择 → checkpoint 按 HELD 局入口态端到端 precision@K（走
     arc_adaptor.click_heatmap.propose_from_prob，与 agent 同一排序/身份/去重
     代码）挑选。v4b 教训：池化 AUC 会选中入口态变差的权重。
  6. 种子固定（v3 无种子，17ch-vs-16ch 的"消融收益"分不清是不是噪声）。

held-out 与 v3/v4 相同：r11l / s5i5 / sb26 / tu93 / ft09（从未参与训练），
数字可与 0.400（v4 最好成绩）与 0.5 晋级线直接对比。
输出：state/click_heatmap_v5.npz（Wa/Wb/Wc/version/in_channels），
validate_click_proposer.py 与 propose_clicks 直接可用。

用法：python state/train_click_heatmap_v5.py [--epochs 400] [--seed 0]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'F:/pro2')
from arc_adaptor import click_heatmap as CH   # noqa: E402

DATASET = Path('F:/pro2/state/affordance_dataset.jsonl')
GRIDS = Path('F:/pro2/state/affordance_grids.npz')
OUT = Path(os.environ.get('HEATMAP_OUT', 'F:/pro2/state/click_heatmap_v5.npz'))
HELD = {'r11l-495a7899', 's5i5-18d95033', 'sb26-7fbdac44', 'tu93-0768757b', 'ft09-0d8bbf25'}
HELD_GIDS = {g.split('-')[0] for g in HELD}
MECHANISM = {'state_only', 'advance', 'win'}
H1 = 24
TOPK = 10
DIST = 2
MARGIN = 1.0
BCE_W = 0.3
PAIRS_PER_GROUP = 24


def load_samples():
    grids = {}
    with np.load(GRIDS) as d:
        for h, g in zip(d['hashes'], d['grids']):
            grids[str(h)] = g.astype(np.int64)
    samples = []
    with open(DATASET, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            ctx = r.get('context') or {}
            if ctx.get('kind') != 'click' or r['grid_hash_before'] not in grids:
                continue
            act = r.get('action') or {}
            x, y = act.get('x'), act.get('y')
            if x is None or y is None:
                continue
            samples.append({
                'gid': r['gid'], 'game_id': r['game_id'], 'level': r['level_idx'],
                'hash': r['grid_hash_before'], 'x': int(x), 'y': int(y),
                'color': int(ctx.get('click_color', grids[r['grid_hash_before']][int(y), int(x)])),
                'label': 1.0 if r['outcome'] in MECHANISM else 0.0,
                'w': float(r['reliability']), 'source': r['source'],
                'step': r['step_idx'],
            })
    return samples, grids


class HeatNet(nn.Module):
    def __init__(self, in_ch=17):
        super().__init__()
        self.a = nn.Conv2d(in_ch, H1, 1, bias=False)
        self.b = nn.Conv2d(H1, H1, 3, padding=1, bias=False)
        self.c = nn.Conv2d(H1, 16, 1, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.b.weight, a=5 ** 0.5)
        nn.init.zeros_(self.c.weight)

    def forward(self, x):
        return self.c(torch.relu(self.b(torch.relu(self.a(x)))))


def build_inputs(grids, hashes):
    """只为主用网格建输入栈（float32），确定性顺序；返回 (栈, hash→行号)。"""
    hashes = sorted(hashes)
    inputs = np.stack([CH.build_input(grids[h]) for h in hashes]).astype(np.float32)
    return inputs, {h: i for i, h in enumerate(hashes)}


def pair_index(samples):
    """同基态内 (正样本行, 负样本行) 对，每组采样上限 PAIRS_PER_GROUP。"""
    by_group = defaultdict(lambda: ([], []))
    for i, s in enumerate(samples):
        (by_group[s['hash']][0 if s['label'] > 0 else 1]).append(i)
    pairs = []
    for h, (pos, neg) in by_group.items():
        if not pos or not neg:
            continue
        allp = [(p, n) for p in pos for n in neg]
        if len(allp) > PAIRS_PER_GROUP:
            allp = allp[::len(allp) // PAIRS_PER_GROUP][:PAIRS_PER_GROUP]
        pairs.extend(allp)
    return pairs


def entry_cases(samples, grids):
    """HELD 局的入口态（level0 首步网格）+ 引擎真值有效点击（productive）。"""
    eff = defaultdict(set)
    frames = {}
    for s in samples:
        if s['game_id'] in HELD and s['level'] == 0:
            frames.setdefault(s['game_id'], s['hash'])
            if s['label'] > 0:
                eff[s['game_id']].add((s['y'], s['x']))
    cases = []
    for game, h in sorted(frames.items()):
        cases.append((game, h, grids[h], sorted(eff.get(game, ()))))
    return cases


def entry_precision(net, cases, inputs, idx):
    """端到端 precision@K：走 propose_from_prob（与 agent 同一决策代码）。"""
    net.eval()
    per_game = {}
    with torch.no_grad():
        for game, h, g, eff in cases:
            if not eff:
                per_game[game] = (None, 0)   # 零有效点击局：剔除但上报
                continue
            x = torch.from_numpy(inputs[idx[h]][None])
            prob = 1.0 / (1.0 + torch.exp(net(x))[0].numpy())
            props = CH.propose_from_prob(prob, np.clip(g, 0, 15), topk=TOPK,
                                         threshold=0.5, enforce=True)
            hits = sum(1 for p in props
                       if any(abs(p['data']['y'] - ey) <= DIST
                              and abs(p['data']['x'] - ex) <= DIST for ey, ex in eff))
            per_game[game] = (hits / max(1, len(props)), len(props))
    net.train()
    vals = [v for v, _ in per_game.values() if v is not None]
    return (sum(vals) / len(vals)) if vals else 0.0, per_game


def main():
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 0
    epochs = int(sys.argv[sys.argv.index('--epochs') + 1]) if '--epochs' in sys.argv else 250
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, __import__('os').cpu_count() or 8))

    samples, grids = load_samples()
    n_pos = sum(1 for s in samples if s['label'] > 0)
    used = {s['hash'] for s in samples}
    PIC_CAP = 10 ** 9
    pos_hashes = sorted({s['hash'] for s in samples if s['label'] > 0})
    neg_hashes = sorted({s['hash'] for s in samples if s['label'] == 0})
    rng0 = np.random.default_rng(seed)
    rng0.shuffle(neg_hashes)
    used = set(pos_hashes + neg_hashes[:max(0, PIC_CAP - len(pos_hashes))])
    samples = [s for s in samples if s['hash'] in used]
    print(f'样本 {len(samples)}（productive 正样本 {n_pos}），主用网格 {len(used)}'
          f'（正样本网格优先，cap={PIC_CAP}），HELD={sorted(HELD_GIDS)}')
    inputs, idx = build_inputs(grids, used)
    cases = entry_cases(samples, grids)
    pairs = pair_index(samples)
    print(f'同基态配对 {len(pairs)} 对')

    pi = np.array([idx[s['hash']] for s in samples])
    px = np.array([s['x'] for s in samples])
    py = np.array([s['y'] for s in samples])
    ly = torch.from_numpy(np.array([s['label'] for s in samples], np.float32))
    w = torch.from_numpy(np.array([s['w'] for s in samples], np.float32))
    pos_i = torch.from_numpy(np.array([p[0] for p in pairs]))
    neg_i = torch.from_numpy(np.array([p[1] for p in pairs]))
    row_hash = [s['hash'] for s in samples]
    pair_hash = [samples[p[0]]['hash'] for p in pairs]
    col = torch.from_numpy(np.array([s['color'] for s in samples]))                                        # 被点格自身颜色通道（监督目标）

    net = HeatNet(17)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    CHUNK = 64
    torch.set_num_threads(min(8, __import__('os').cpu_count() or 8))
    all_hashes = sorted(idx)                            # 按 hash 分块（row 是其行号）
    SUBSAMPLE = 1.0                                     # 每 epoch 随机抽 70% 网格
    best = (-1.0, None)

    for ep in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        pick = set(rng0.choice(all_hashes, max(1, int(len(all_hashes) * SUBSAMPLE)),
                              replace=False))
        loss_sum = 0.0
        ep_hashes = [h for h in all_hashes if h in pick]
        chunks = [ep_hashes[i:i + CHUNK] for i in range(0, len(ep_hashes), CHUNK)]
        for ch in chunks:
            chs = set(ch)
            X = torch.from_numpy(inputs[[idx[h] for h in ch]])
            logits = net(X)                                # (C,16,64,64)
            m = [i for i, h in enumerate(row_hash) if h in chs]
            if len(m) == 0:
                continue
            row_local = {idx[h]: j for j, h in enumerate(ch)}
            mi = torch.from_numpy(np.array([row_local[pi[i]] for i in m]))
            sel = logits[mi, col[m], torch.from_numpy(py[m]), torch.from_numpy(px[m])]
            loss = (bce(sel, ly[m]) * w[m]).sum() / len(samples)
            mp = [i for i, h in enumerate(pair_hash) if h in chs]
            if mp:
                pos_n = pos_i.numpy()[np.array(mp)]
                neg_n = neg_i.numpy()[np.array(mp)]
                pos_l = torch.from_numpy(np.array([row_local[pi[p]] for p in pos_n]))
                neg_l = torch.from_numpy(np.array([row_local[pi[n]] for n in neg_n]))
                d = (logits[pos_l, col[torch.from_numpy(pos_n)],
                            torch.from_numpy(py[pos_n]), torch.from_numpy(px[pos_n])]
                     - logits[neg_l, col[torch.from_numpy(neg_n)],
                              torch.from_numpy(py[neg_n]), torch.from_numpy(px[neg_n])])
                loss = loss + torch.relu(MARGIN - d).sum() / max(1, len(pairs))
            loss.backward()
            loss_sum += float(loss)
        opt.step()

        if ep % 5 == 0 or ep == epochs:
            prec, per_game = entry_precision(net, cases, inputs, idx)
            star = ' *' if prec > best[0] else ''
            detail = '  '.join(
                f"{g.split('-')[0]}={'--' if v is None else f'{v:.2f}'}"
                for g, (v, _) in per_game.items())
            print(f'  ep{ep:4d} loss={loss_sum:.4f} entry_p@{TOPK}={prec:.3f} {detail}{star}',
                  flush=True)
            if prec > best[0]:
                best = (prec, {k: v.detach().numpy().astype(np.float32)
                               for k, v in net.state_dict().items()})
    prec, _ = entry_precision(net, cases, inputs, idx)
    print(f'final entry_p@{TOPK}={prec:.3f}  best={best[0]:.3f}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    keep = best[1] if best[1] is not None else {
        k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez(OUT,
             Wa=keep['a.weight'].reshape(keep['a.weight'].shape[0], -1),
             Wb=keep['b.weight'],
             Wc=keep['c.weight'].reshape(keep['c.weight'].shape[0], -1),
             version=np.array([f'v5-engine-truth-seed{seed}']),
             in_channels=np.array([17]),
             entry_precision=np.array([best[0]]))
    print(f'💾 {OUT}（best entry_p@{TOPK}={best[0]:.3f}）')


if __name__ == '__main__':
    main()
