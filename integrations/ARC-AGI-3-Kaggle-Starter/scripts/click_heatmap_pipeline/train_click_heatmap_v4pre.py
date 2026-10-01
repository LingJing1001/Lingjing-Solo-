"""v4pre experiment: contrastive trunk pretraining (START_HERE recipe) + frozen head.

Question: does contrastive pretraining of the trunk on UNLABELED game frames
(the arc3/START_HERE idea, moved from static grids to game frames) improve the
click-heatmap head vs v3's joint training from scratch?

Arms:
  pretrain  trunk (convA 1x1 17->24 + convB 3x3 24->24) with InfoNCE: two
            jittered 32x32 crops of the same frame are positives, other frames
            in the batch are negatives; projection head 24->32, temperature 0.1.
  finetune  FREEZE trunk, train only convC (24->16 per-colour click logits) with
            BCE on labelled probe cells (same data/rules as v3).
Baseline:  v3 (joint from scratch) — same held-out five games.
"""
import hashlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, 'F:/pro2')
from arc_adaptor import click_heatmap as CH

LABELS = 'F:/pro2/state/click_labels.npz'
ROUTES = 'F:/pro2/state/route_clicks.npz'
OUT = 'F:/pro2/state/click_heatmap_v4pre.npz'
H1 = 24
HELD_OUT = {'r11l-495a7899', 's5i5-18d95033', 'sb26-7fbdac44', 'tu93-0768757b', 'ft09-0d8bbf25'}
PAD = 16
CROP = 2 * PAD


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


class Trunk(nn.Module):
    """same layout as v3's convA+convB (numpy-exportable)."""
    def __init__(self, in_ch=17):
        super().__init__()
        self.a = nn.Conv2d(in_ch, H1, 1, bias=False)
        self.b = nn.Conv2d(H1, H1, 3, padding=1, bias=False)

    def forward(self, x):
        return F.relu(self.b(F.relu(self.a(x))))


class Net(nn.Module):
    """frozen trunk + trainable per-colour click head."""
    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk
        self.c = nn.Conv2d(H1, 16, 1, bias=False)
        nn.init.zeros_(self.c.weight)

    def forward(self, x):
        return self.c(self.trunk(x))


def unique_frames(frames):
    seen, ordered, uidx = {}, [], []
    for f in frames:
        h = hashlib.md5(np.asarray(f, np.int8).tobytes()).hexdigest()
        if h not in seen:
            seen[h] = True
            ordered.append(f)
        uidx.append(len(ordered) - 1)
    return np.asarray(ordered), np.asarray(uidx)


def rand_crops(xp, idx, rng):
    """random 32x32 crops from padded (N,17,96,96), fully inside the 64x64 area."""
    out = np.empty((len(idx), xp.shape[1], CROP, CROP), np.float32)
    for j, i in enumerate(idx):
        ox = int(rng.randint(0, 64 - CROP + 1)) + PAD
        oy = int(rng.randint(0, 64 - CROP + 1)) + PAD
        out[j] = xp[i, :, oy:oy + CROP, ox:ox + CROP]
    return torch.from_numpy(out)


def pretrain(inputs):
    trunk = Trunk(inputs.shape[1])
    proj = nn.Linear(H1, 32, bias=False)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(proj.parameters()), lr=1e-3)
    rng = np.random.RandomState(0)
    xp = np.pad(inputs, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)))
    for step in range(1, 301):
        idx = rng.choice(len(xp), size=min(32, len(xp)), replace=False)
        f1 = trunk(rand_crops(xp, idx, rng)).mean(dim=(2, 3))
        f2 = trunk(rand_crops(xp, idx, rng)).mean(dim=(2, 3))
        z1 = F.normalize(proj(f1), dim=1)
        z2 = F.normalize(proj(f2), dim=1)
        logits = z1 @ z2.t() / 0.1
        target = torch.arange(len(idx))
        loss = F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f'  pretrain step {step}: infonce={loss.item():.3f}', flush=True)
    return trunk


def main():
    frames, px, py, label, groups, keep = load_all()
    frames = np.asarray(frames)
    uniq_frames, uidx = unique_frames(frames)
    inputs = np.stack([CH.build_input(f) for f in uniq_frames])
    xp = np.pad(inputs, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)))
    held = np.array([g in HELD_OUT for g in groups])
    tr = np.where(~held & keep)[0]
    va = np.where(held & keep)[0]
    colors = np.asarray([frames[i][py[i], px[i]] for i in range(len(frames))])
    y_all = torch.from_numpy(label)

    print('arm1: contrastive trunk pretraining on unlabeled frames', flush=True)
    trunk = pretrain(inputs)

    net = Net(trunk)
    for p in net.trunk.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(net.c.parameters(), lr=3e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    def head_logits(idx):
        with torch.no_grad():
            feat = net.trunk(crop(idx, uidx, inputs, xp))
        ch = torch.from_numpy(colors[idx])
        ar = torch.arange(len(idx))
        return net.c(feat)[ar, ch, PAD, PAD]

    def crop(idx, uidx, inputs, xp):
        out = np.empty((len(idx), inputs.shape[1], CROP, CROP), np.float32)
        for j, i in enumerate(idx):
            out[j] = xp[uidx[i], :, py[i]:py[i] + CROP, px[i]:px[i] + CROP]
        return torch.from_numpy(out)

    for epoch in range(1, 301):
        perm = np.random.RandomState(epoch).permutation(len(tr))
        for i in range(0, len(tr), 128):
            idx = tr[perm[i:i + 128]]
            opt.zero_grad()
            sel = head_logits(idx)
            loss = bce(sel, y_all[idx])
            loss.backward()
            opt.step()
        if epoch % 100 == 0:
            with torch.no_grad():
                sv = head_logits(va).numpy()
            order = np.argsort(sv)
            ranks = np.empty(len(sv)); ranks[order] = np.arange(1, len(sv) + 1)
            npos = int(label[va].sum()); nneg = len(va) - npos
            auc = (ranks[label[va] > 0].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)
            print(f'epoch {epoch}: val_auc={auc:.3f}', flush=True)

    sd = net.state_dict()
    np.savez_compressed(
        OUT,
        Wa=sd['trunk.a.weight'].squeeze(-1).squeeze(-1).numpy().astype(np.float32),
        Wb=sd['trunk.b.weight'].numpy().astype(np.float32),
        Wc=sd['c.weight'].squeeze(-1).squeeze(-1).numpy().astype(np.float32),
        held_out=np.asarray(sorted(HELD_OUT)),
        version=np.asarray(['v4pre-contrastive-trunk']),
    )
    print('saved', OUT)


if __name__ == '__main__':
    main()
