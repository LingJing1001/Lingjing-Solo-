"""Train click-heatmap v3: structural feature plane + cell-dedup evaluation (P3).

Input planes per frame (built by arc_adaptor/click_heatmap.build_input):
  16 colour one-hot planes + 1 structural plane = log(same-colour connected
  component size) per cell, normalised — the transferable "small button vs big
  wall" cue that colour identity alone cannot provide (v2 finding: colour does
  not transfer across games).
Loss: margin ranking (pos beats same-frame negs) + 0.3 BCE anchor; weak probes
(1-2 cell changes) excluded; route clicks are positives by construction.
Held-out games: r11l / s5i5 / sb26 / tu93 / ft09 (never trained on).
"""
import hashlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'F:/pro2')
from arc_adaptor import click_heatmap as CH

LABELS = 'F:/pro2/state/click_labels.npz'
ROUTES = 'F:/pro2/state/route_clicks.npz'
OUT = 'F:/pro2/state/click_heatmap_v3.npz'
OUT_NOPLANE = 'F:/pro2/state/click_heatmap_v3_noplane.npz'
# P3 ablation (docs/热图提议器优化方案.md): same data + same loss, without the
# connected-component plane, to decide whether the structural feature earns its keep.
USE_PLANE = '--no-plane' not in sys.argv
H1 = 24
HELD_OUT = {'r11l-495a7899', 's5i5-18d95033', 'sb26-7fbdac44', 'tu93-0768757b', 'ft09-0d8bbf25'}
PAD = 16
CROP = 2 * PAD


def build_planes(frame):
    """(17,64,64) with the structural plane; (16,64,64) colour-only under --no-plane."""
    return CH.build_input(frame) if USE_PLANE else CH.onehot(frame)


def load_all():
    d = np.load(LABELS, allow_pickle=False)
    games = [g for g in d['games']]
    frames = list(d['frames'].astype(np.int64))
    px = list(np.clip(d['px'].astype(np.int64), 0, 63))
    py = list(np.clip(d['py'].astype(np.int64), 0, 63))
    label = [1 if n >= 3 else 0 for n in d['n_changed']]
    keep = [n == 0 or n >= 3 for n in d['n_changed']]
    groups = list(games[g] for g in d['game_idx'])

    r = np.load(ROUTES, allow_pickle=False)
    for i in range(len(r['frames'])):
        frames.append(r['frames'][i].astype(np.int64))
        px.append(int(r['px'][i])); py.append(int(r['py'][i]))
        label.append(1)
        groups.append('route:' + str(r['games'][i]))
    keep = np.asarray(keep + [True] * len(r['frames']), bool)
    return frames, np.asarray(px), np.asarray(py), np.asarray(label, np.float32), groups, keep


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


def main():
    frames, px, py, label, groups, keep = load_all()

    # deduplicate frames: one input stack per unique picture (25 entries + route states)
    uniq, uidx, first_at = {}, [], []
    for i, f in enumerate(frames):
        h = hashlib.md5(np.asarray(f, np.int8).tobytes()).hexdigest()
        if h not in uniq:
            uniq[h] = len(uniq)
            first_at.append(i)             # slot k of `inputs` is frames[first_at[k]]
        uidx.append(uniq[h])
    # NOT frames[:len(uniq)]: first occurrences sit at 0, 81, 221, ... so a head
    # slice would feed almost every probe the wrong picture (measured 83/84 wrong).
    assert sorted(first_at) == sorted(set(first_at)) and max(first_at) < len(frames)
    inputs = np.stack([build_planes(frames[i]) for i in first_at])   # (U,C,64,64)
    padded = np.pad(inputs, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)))
    uidx = np.asarray(uidx)
    print(f'unique frames={len(uniq)} samples={len(frames)}', flush=True)

    held = np.array([g in HELD_OUT for g in groups])
    tr = np.where(~held & keep)[0]
    va = np.where(held & keep)[0]
    colors = np.asarray([frames[i][py[i], px[i]] for i in range(len(frames))])

    net = HeatNet(in_ch=inputs.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-4)

    def crop(idx):
        out = np.empty((len(idx), inputs.shape[1], CROP, CROP), np.float32)
        for j, i in enumerate(idx):
            out[j] = padded[uidx[i], :, py[i]:py[i] + CROP, px[i]:px[i] + CROP]
        return torch.from_numpy(out)

    def sel_at(logits, idx):
        ch = torch.from_numpy(colors[idx])
        ar = torch.arange(len(idx))
        return logits[ar, ch, torch.full((len(idx),), PAD), torch.full((len(idx),), PAD)]

    def scores(idx):
        with torch.no_grad():
            return sel_at(net(crop(idx)), idx).numpy()

    by_group = {}
    for i in tr:
        by_group.setdefault(int(sum(hash(g) % 99991 for g in groups[i:i+1])), []).append(i)
    pair_groups = []
    for g, members in by_group.items():
        pos = [i for i in members if label[i] > 0]
        neg = [i for i in members if label[i] == 0]
        if pos and neg:
            pair_groups.append((pos, neg))
    print(f'training samples={len(tr)} pair_groups={len(pair_groups)} '
          f'(pos={int(label[tr].sum())})', flush=True)

    y_all = torch.from_numpy(label)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(1, 301):
        rng = np.random.RandomState(epoch)
        pairs = []
        for pos, neg in pair_groups:
            for p in pos:
                for n in rng.choice(neg, size=min(2, len(neg)), replace=False):
                    pairs.append((p, int(n)))
        perm = np.random.RandomState(epoch).permutation(len(pairs))
        tot = 0.0
        for i in range(0, len(pairs), 256):
            chunk = [pairs[j] for j in perm[i:i + 256]]
            p_idx = np.asarray([p for p, _ in chunk])
            n_idx = np.asarray([n for _, n in chunk])
            both = np.concatenate([p_idx, n_idx])
            opt.zero_grad()
            logits = net(crop(both))
            k = len(p_idx)
            ar = torch.arange(k)
            s_p = logits[ar, torch.from_numpy(colors[p_idx]), PAD, PAD]
            s_n = logits[ar + k, torch.from_numpy(colors[n_idx]), PAD, PAD]
            rank = torch.clamp(1.0 - (s_p - s_n), min=0).mean()
            b = bce(s_p, y_all[p_idx]) + bce(s_n, y_all[n_idx])
            loss = rank + 0.3 * b
            loss.backward()
            opt.step()
            tot += float(loss)
        if epoch % 50 == 0:
            sv = scores(va)
            order = np.argsort(sv)
            ranks = np.empty(len(sv)); ranks[order] = np.arange(1, len(sv) + 1)
            npos = int(label[va].sum()); nneg = len(va) - npos
            auc = (ranks[label[va] > 0].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)
            print(f'epoch {epoch}: loss={tot:.3f} val_auc={auc:.3f}', flush=True)

    sd = net.state_dict()
    out = OUT if USE_PLANE else OUT_NOPLANE
    np.savez_compressed(
        out,
        Wa=sd['a.weight'].squeeze(-1).squeeze(-1).numpy().astype(np.float32),
        Wb=sd['b.weight'].numpy().astype(np.float32),
        Wc=sd['c.weight'].squeeze(-1).squeeze(-1).numpy().astype(np.float32),
        held_out=np.asarray(sorted(HELD_OUT)),
        version=np.asarray(['v3-structural' if USE_PLANE else 'v3-ablation-16ch']),
        in_channels=np.asarray([inputs.shape[1]]),
    )
    print('saved', out, 'in_channels=%d' % inputs.shape[1])


if __name__ == '__main__':
    main()
