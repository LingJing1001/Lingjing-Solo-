"""点击提议编码器 — ConvVariant patch 编码 + SupCon 配对训练 + few-shot 校准。

任务（ARC-AGI-3 口径）：给定 64×64 游戏帧，提议最可能"有效"的点击格
（有效 = 点击引起帧变化，来自探针真值 click_labels*.npz）。

与旧热图提议器（arc_adaptor.click_heatmap）的区别：
    - 本模块用可训练的 ConvVariant 编码器对点击格邻域 patch 做嵌入，
      SupCon/InfoNCE 配对目标：有效 patch 类内聚合、类间分离
      （修复旧对比目标"同一内容所有变换相似度≈1"的判别力缺失）。
    - 部署协议模拟真实探针预算：held-out 局用 ≤3 个有效 + 3 个无效格
      做 few-shot 原型校准，再对其余格打分。

Public API:
    load_pool(files, patch_radius=7)      — 合并 npz → (patches, labels, 元信息)
    train_click_encoder(enc, pool, ...)   — SupCon 训练，参数写回 enc
    propose_clicks_encoder(frame, ...)    — 校准 + 打分 + NMS → top-k 格
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .conv_variants import ConvVariant
from .conv_frontier_torch import build_param_tensors, one_hot_batch, forward as tv_forward

_EPS = 1e-8


# ───────────────────────── 数据：patch 池 ─────────────────────────

def load_pool(files: Sequence[str], patch_radius: int = 7,
              games: Optional[Sequence[str]] = None):
    """合并多个 click_labels npz 为 patch 池。

    Returns dict:
        patches  (N,1,2r+1,2r+1) int8   点击格邻域（边界零填充）
        labels   (N,) int8              1=有效点击 0=无效
        game     (N,) <U16              游戏名
        frames   (N,) int64             所在帧在该局内的行号（用于取入口帧）
        frame_grid: {game: 入口帧 64×64}
        points:     {game: [(px, py), ...]} 该局出现过的全部点阵格
        eff_cells:  {game: [(px, py), ...]} 全部有效格（探针真值）
    """
    R = patch_radius
    patches, labels, gnames, frows = [], [], [], []
    frame_grid: Dict[str, np.ndarray] = {}
    points: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    eff_cells: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    seen_frames: Dict[str, set] = defaultdict(set)

    for fp in files:
        d = np.load(fp, allow_pickle=False)
        gnames_file = d["games"].tolist()
        # 一次性读出大数组，避免 NpzFile 每次 d[key] 重复解压
        frames_arr = d["frames"]
        px_arr = d["px"]
        py_arr = d["py"]
        eff_arr = d["eff"]
        gidx_arr = d["game_idx"]
        for i in range(len(px_arr)):
            g = gnames_file[int(gidx_arr[i])]
            if games is not None and g not in games:
                continue
            px, py = int(px_arr[i]), int(py_arr[i])
            fr = frames_arr[i]
            key = (int(gidx_arr[i]),)  # 帧数组即该行快照
            patches.append(_extract_patch(fr, px, py, R))
            labels.append(int(eff_arr[i]))
            gnames.append(g)
            frows.append(len(seen_frames[g]))
            if g not in frame_grid:
                frame_grid[g] = fr.copy()
            seen_frames[g].add(len(seen_frames[g]))
            if (px, py) not in points[g]:
                points[g].append((px, py))
            if int(eff_arr[i]) == 1 and (px, py) not in eff_cells[g]:
                eff_cells[g].append((px, py))

    # 首帧取该局第一行实际帧（上面 frame_grid 已是首见帧）
    return {
        "patches": np.stack(patches).astype(np.int8),
        "labels": np.asarray(labels, dtype=np.int8),
        "game": np.asarray(gnames),
        "frame_row": np.asarray(frows, dtype=np.int64),
        "frame_grid": frame_grid,
        "points": dict(points),
        "eff_cells": dict(eff_cells),
    }


def _extract_patch(frame: np.ndarray, px: int, py: int, radius: int) -> np.ndarray:
    H, W = frame.shape
    out = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int8)
    y0, y1 = py - radius, py + radius + 1
    x0, x1 = px - radius, px + radius + 1
    fy0, fy1 = max(0, y0), min(H, y1)
    fx0, fx1 = max(0, x0), min(W, x1)
    if fy0 < fy1 and fx0 < fx1:
        out[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = frame[fy0:fy1, fx0:fx1]
    return out


# ───────────────────────── 训练：SupCon ─────────────────────────

def _supcon_loss(E: torch.Tensor, y: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """SupCon（Khosla et al. 2020）。y∈{0,1}，两类各自充当对方的负类。"""
    z = F.normalize(E, dim=1)
    sim = z @ z.T / temp
    n = E.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=E.device)
    pos_mask = (y[:, None] == y[None, :]) & ~eye
    if not pos_mask.any():
        return E.new_zeros(())
    # log-softmax over 非 self 全体
    # 注意：不能把 sim 对角线填 -inf 后再乘 pos_mask，因为 -inf * 0 = nan。
    # logsumexp 用 masked 版排除 self，logp 用原 sim（全 finite）。
    logp = sim - torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=1, keepdim=True)
    loss = -(logp * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
    return loss[pos_mask.any(1)].mean()


def _augment_patch(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """随机翻转 + 90° 旋转增强（保持 15×15 邻域语义）。"""
    if rng.random() < 0.5:
        p = p[:, ::-1]
    if rng.random() < 0.5:
        p = p[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        p = np.rot90(p, k)
    return np.ascontiguousarray(p)


def train_click_encoder(enc: ConvVariant, pool: dict, steps: int = 500,
                        per_class: int = 48, lr: float = 1e-3,
                        temp: float = 0.2, seed: int = 7,
                        log_every: int = 100, verbose: bool = True,
                        weight_decay: float = 1e-4,
                        warmup_frac: float = 0.1,
                        grad_clip: float = 1.0,
                        patience: int = 150,
                        optimizer: str = "adamw",
                        augment: bool = False) -> List[float]:
    """SupCon 训练 patch 编码器；每步平衡采样 正/负 各 per_class 个。

    修复训练发散（v1 用 SGD lr=0.01 无调度，loss 在 4.0~4.7 震荡）：
      - AdamW + weight_decay 替代 SGD（对比学习对自适应优化器更友好）
      - warmup（前 warmup_frac 线性升温）+ cosine annealing
      - 梯度裁剪 max_norm=grad_clip
      - early stopping：训练 loss 连续 patience 步不降则停
      - temp 0.1→0.2：原值梯度过于尖锐
      - augment=True：每步对采样 patch 生成翻转/旋转增强视图拼入 batch，
        SupCon 正对自然包含 (patch, augment(patch))，学习不变性
    """
    rng = np.random.default_rng(seed)
    P = build_param_tensors(enc)
    order = list(P.keys())
    params = [P[k] for k in order]
    if optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=float(lr), momentum=0.9)
    else:
        opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    warmup_steps = max(1, int(steps * warmup_frac))
    base_lr = float(lr)
    X_all = torch.from_numpy(
        np.stack([np.eye(int(enc.num_colors), dtype=np.float32)[
            np.clip(p, 0, enc.num_colors - 1)].transpose(2, 0, 1)
            for p in pool["patches"]]))
    y_all = torch.from_numpy(pool["labels"].astype(np.int64))
    pos_idx = np.where(pool["labels"] == 1)[0]
    neg_idx = np.where(pool["labels"] == 0)[0]
    history: List[float] = []
    best_loss = float("inf")
    no_improve = 0
    for st in range(1, steps + 1):
        # warmup + cosine annealing
        if st <= warmup_steps:
            cur_lr = base_lr * st / warmup_steps
        else:
            progress = (st - warmup_steps) / max(1, steps - warmup_steps)
            cur_lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        sel = np.concatenate([
            rng.choice(pos_idx, min(per_class, len(pos_idx)), replace=False),
            rng.choice(neg_idx, min(per_class, len(neg_idx), ), replace=False),
        ])
        rng.shuffle(sel)
        if augment:
            # 对采样 patch 生成增强视图，拼入 batch；label 重复
            aug_patches = np.stack([
                _augment_patch(pool["patches"][i], rng) for i in sel
            ])
            X_aug = torch.from_numpy(np.stack([
                np.eye(int(enc.num_colors), dtype=np.float32)[
                    np.clip(p, 0, enc.num_colors - 1)].transpose(2, 0, 1)
                for p in aug_patches
            ]))
            X_batch = torch.cat([X_all[sel], X_aug], dim=0)
            y_batch = torch.cat([y_all[sel], y_all[sel]], dim=0)
            E = tv_forward(enc, X_batch, P)
            loss = _supcon_loss(E, y_batch, temp=temp)
        else:
            E = tv_forward(enc, X_all[sel], P)
            loss = _supcon_loss(E, y_all[sel], temp=temp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        cur = float(loss)
        history.append(cur)
        if cur < best_loss - 1e-4:
            best_loss = cur
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (st % log_every == 0 or st == 1):
            print(f"    supcon step {st:4d}/{steps}  loss={cur:.4f}  lr={cur_lr:.2e}  best={best_loss:.4f}")
        if patience > 0 and no_improve >= patience:
            if verbose:
                print(f"    early stop @ step {st}  best_loss={best_loss:.4f}  (no improve {no_improve} steps)")
            break
    for k in order:
        setattr(enc, k, P[k].detach().numpy().copy())
    return history


# ───────────────────────── 部署：校准提议 ─────────────────────────

def _embed_patches(enc: ConvVariant, patches: np.ndarray, P=None) -> np.ndarray:
    """(N,h,w) int8 → (N,dim) 归一化嵌入（torch 批量，无梯度）。"""
    embs = []
    B = 256
    for i in range(0, len(patches), B):
        X = one_hot_batch(enc, list(patches[i:i + B]))
        if P is not None:
            E = tv_forward(enc, X, P)
        else:
            with torch.no_grad():
                E = tv_forward(enc, X, build_param_tensors(enc))
        embs.append(F.normalize(E, dim=1).detach().numpy())
    return np.concatenate(embs, axis=0)


def propose_clicks_encoder(frame: np.ndarray, enc: ConvVariant,
                           points: Sequence[Tuple[int, int]],
                           calib_pos: Sequence[Tuple[int, int]],
                           calib_neg: Sequence[Tuple[int, int]],
                           patch_radius: int = 7, topk: int = 8,
                           nms_radius: int = 4) -> List[dict]:
    """few-shot 校准的点击提议。

    score = mean cos(emb(cell), pos_protos) − mean cos(emb(cell), neg_protos)
    NMS 去近邻后返回 top-k：[{"data": {"x", "y"}, "score", "color"}]
    """
    R = patch_radius
    cells = list(points)
    if not cells:
        return []
    patches = np.stack([_extract_patch(frame, x, y, R) for x, y in cells])
    E = _embed_patches(enc, patches)
    if calib_pos:
        Ep = _embed_patches(enc, np.stack(
            [_extract_patch(frame, x, y, R) for x, y in calib_pos]))
        pos_proto = Ep.mean(axis=0, keepdims=True)
    else:
        pos_proto = np.zeros((1, E.shape[1]), dtype=np.float32)
    if calib_neg:
        En = _embed_patches(enc, np.stack(
            [_extract_patch(frame, x, y, R) for x, y in calib_neg]))
        neg_proto = En.mean(axis=0, keepdims=True)
    else:
        neg_proto = np.zeros((1, E.shape[1]), dtype=np.float32)
    scores = E @ pos_proto[0] - E @ neg_proto[0]

    order = np.argsort(-scores)
    kept: List[dict] = []
    for i in order:
        x, y = cells[i]
        if any(max(abs(x - k["data"]["x"]), abs(y - k["data"]["y"])) <= nms_radius
               for k in kept):
            continue
        kept.append({"data": {"x": int(x), "y": int(y)},
                     "score": float(scores[i]),
                     "color": int(frame[y, x])})
        if len(kept) >= topk:
            break
    return kept


def split_calibration(eff_cells: Sequence[Tuple[int, int]],
                      points: Sequence[Tuple[int, int]],
                      n_pos: int = 3, n_neg: int = 3,
                      min_neg_dist: int = 6):
    """确定性划分校准格/真值格：校准消耗 ≤n_pos 个有效格 + n_neg 个远处无效格。"""
    eff = sorted(eff_cells)
    if len(eff) <= n_pos:
        calib_pos, truth = list(eff), []
    else:
        picks = {0, len(eff) // 2, len(eff) - 1}
        picks = sorted(picks)[:n_pos]
        calib_pos = [eff[i] for i in picks]
        truth = [c for i, c in enumerate(eff) if i not in picks]
    far_neg = [c for c in sorted(points)
               if c not in eff
               and all(max(abs(c[0] - e[0]), abs(c[1] - e[1])) > min_neg_dist
                       for e in eff)]
    step = max(1, len(far_neg) // max(1, n_neg))
    calib_neg = far_neg[::step][:n_neg]
    return calib_pos, calib_neg, truth


# ───────────────────────── 直接监督：BCE 分类 ─────────────────────────

def train_click_classifier(enc: ConvVariant, pool: dict, steps: int = 1000,
                           lr: float = 1e-3, seed: int = 7,
                           log_every: int = 100, verbose: bool = True,
                           weight_decay: float = 1e-4,
                           warmup_frac: float = 0.1,
                           grad_clip: float = 1.0,
                           patience: int = 200,
                           batch_size: int = 512) -> dict:
    """直接监督训练：BCE 二分类 eff ∈ {0,1}，无需 few-shot 校准。

    与 train_click_encoder（SupCon + 校准）的区别：
      - 损失：BCE(logit, eff) 直接学点击有效性，而非学嵌入几何
      - 推理：直接用 logit 排序，无需校准原型（消除校准误差）
      - pos_weight 自动处理正负不平衡（3335 正 vs 56119 负）

    Returns: {"w_cls": (embed_dim,) ndarray, "b_cls": () ndarray}
    """
    rng = np.random.default_rng(seed)
    P = build_param_tensors(enc)
    order = list(P.keys())
    w_cls = torch.zeros(int(enc.embed_dim), dtype=torch.float32, requires_grad=True)
    b_cls = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    params = [P[k] for k in order] + [w_cls, b_cls]
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    warmup_steps = max(1, int(steps * warmup_frac))
    base_lr = float(lr)

    X_all = torch.from_numpy(
        np.stack([np.eye(int(enc.num_colors), dtype=np.float32)[
            np.clip(p, 0, enc.num_colors - 1)].transpose(2, 0, 1)
            for p in pool["patches"]]))
    y_all = torch.from_numpy(pool["labels"].astype(np.float32))
    n_pos = float((pool["labels"] == 1).sum())
    n_neg = float((pool["labels"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)

    n = len(pool["labels"])
    history: List[float] = []
    best_loss = float("inf")
    no_improve = 0
    for st in range(1, steps + 1):
        if st <= warmup_steps:
            cur_lr = base_lr * st / warmup_steps
        else:
            progress = (st - warmup_steps) / max(1, steps - warmup_steps)
            cur_lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        sel = rng.choice(n, min(batch_size, n), replace=False)
        E = tv_forward(enc, X_all[sel], P)
        logit = E @ w_cls + b_cls
        loss = F.binary_cross_entropy_with_logits(logit, y_all[sel], pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        cur = float(loss)
        history.append(cur)
        if cur < best_loss - 1e-4:
            best_loss = cur
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (st % log_every == 0 or st == 1):
            print(f"    bce step {st:4d}/{steps}  loss={cur:.4f}  lr={cur_lr:.2e}  best={best_loss:.4f}")
        if patience > 0 and no_improve >= patience:
            if verbose:
                print(f"    early stop @ step {st}  best_loss={best_loss:.4f}")
            break
    for k in order:
        setattr(enc, k, P[k].detach().numpy().copy())
    return {
        "w_cls": w_cls.detach().numpy().copy(),
        "b_cls": b_cls.detach().item(),
    }


def propose_clicks_classifier(frame: np.ndarray, enc: ConvVariant,
                              points: Sequence[Tuple[int, int]],
                              cls_head: dict,
                              patch_radius: int = 7, topk: int = 8,
                              nms_radius: int = 4) -> List[dict]:
    """直接用分类头 logit 排序，无需 few-shot 校准。

    logit = emb(cell) @ w_cls + b_cls
    NMS 去近邻后返回 top-k。
    """
    R = patch_radius
    cells = list(points)
    if not cells:
        return []
    patches = np.stack([_extract_patch(frame, x, y, R) for x, y in cells])
    E = _embed_patches(enc, patches)
    w = cls_head["w_cls"]
    b = cls_head["b_cls"]
    scores = E @ w + b

    order = np.argsort(-scores)
    kept: List[dict] = []
    for i in order:
        x, y = cells[i]
        if any(max(abs(x - k["data"]["x"]), abs(y - k["data"]["y"])) <= nms_radius
               for k in kept):
            continue
        kept.append({"data": {"x": int(x), "y": int(y)},
                     "score": float(scores[i]),
                     "color": int(frame[y, x])})
        if len(kept) >= topk:
            break
    return kept


# ───────────────────────── 直接监督：BCE 分类 ─────────────────────────

def train_click_classifier(enc: ConvVariant, pool: dict, steps: int = 1000,
                           lr: float = 1e-3, seed: int = 7,
                           log_every: int = 100, verbose: bool = True,
                           weight_decay: float = 1e-4,
                           warmup_frac: float = 0.1,
                           grad_clip: float = 1.0,
                           patience: int = 200,
                           batch_size: int = 512) -> dict:
    """直接监督训练：BCE 二分类 eff ∈ {0,1}，无需 few-shot 校准。

    与 train_click_encoder（SupCon + 校准）的区别：
      - 损失：BCE(logit, eff) 直接学点击有效性，而非学嵌入几何
      - 推理：直接用 logit 排序，无需校准原型（消除校准误差）
      - pos_weight 自动处理正负不平衡（3335 正 vs 56119 负）

    Returns: {"w_cls": (embed_dim,) ndarray, "b_cls": () ndarray}
    """
    rng = np.random.default_rng(seed)
    P = build_param_tensors(enc)
    order = list(P.keys())
    w_cls = torch.zeros(int(enc.embed_dim), dtype=torch.float32, requires_grad=True)
    b_cls = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    params = [P[k] for k in order] + [w_cls, b_cls]
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    warmup_steps = max(1, int(steps * warmup_frac))
    base_lr = float(lr)

    X_all = torch.from_numpy(
        np.stack([np.eye(int(enc.num_colors), dtype=np.float32)[
            np.clip(p, 0, enc.num_colors - 1)].transpose(2, 0, 1)
            for p in pool["patches"]]))
    y_all = torch.from_numpy(pool["labels"].astype(np.float32))
    n_pos = float((pool["labels"] == 1).sum())
    n_neg = float((pool["labels"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)

    n = len(pool["labels"])
    history: List[float] = []
    best_loss = float("inf")
    no_improve = 0
    for st in range(1, steps + 1):
        if st <= warmup_steps:
            cur_lr = base_lr * st / warmup_steps
        else:
            progress = (st - warmup_steps) / max(1, steps - warmup_steps)
            cur_lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        sel = rng.choice(n, min(batch_size, n), replace=False)
        E = tv_forward(enc, X_all[sel], P)
        logit = E @ w_cls + b_cls
        loss = F.binary_cross_entropy_with_logits(logit, y_all[sel], pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        cur = float(loss)
        history.append(cur)
        if cur < best_loss - 1e-4:
            best_loss = cur
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (st % log_every == 0 or st == 1):
            print(f"    bce step {st:4d}/{steps}  loss={cur:.4f}  lr={cur_lr:.2e}  best={best_loss:.4f}")
        if patience > 0 and no_improve >= patience:
            if verbose:
                print(f"    early stop @ step {st}  best_loss={best_loss:.4f}")
            break
    for k in order:
        setattr(enc, k, P[k].detach().numpy().copy())
    return {
        "w_cls": w_cls.detach().numpy().copy(),
        "b_cls": b_cls.detach().item(),
    }


def propose_clicks_classifier(frame: np.ndarray, enc: ConvVariant,
                              points: Sequence[Tuple[int, int]],
                              cls_head: dict,
                              patch_radius: int = 7, topk: int = 8,
                              nms_radius: int = 4) -> List[dict]:
    """直接用分类头 logit 排序，无需 few-shot 校准。

    logit = emb(cell) @ w_cls + b_cls
    NMS 去近邻后返回 top-k。
    """
    R = patch_radius
    cells = list(points)
    if not cells:
        return []
    patches = np.stack([_extract_patch(frame, x, y, R) for x, y in cells])
    E = _embed_patches(enc, patches)
    w = cls_head["w_cls"]
    b = cls_head["b_cls"]
    scores = E @ w + b

    order = np.argsort(-scores)
    kept: List[dict] = []
    for i in order:
        x, y = cells[i]
        if any(max(abs(x - k["data"]["x"]), abs(y - k["data"]["y"])) <= nms_radius
               for k in kept):
            continue
        kept.append({"data": {"x": int(x), "y": int(y)},
                     "score": float(scores[i]),
                     "color": int(frame[y, x])})
        if len(kept) >= topk:
            break
    return kept
