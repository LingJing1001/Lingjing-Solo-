"""卷积++v2 的 torch autograd 训练通道。

动机:
    conv_frontier_v2._sgd_step 原本用有限差分数值梯度,对每个参数做 2 次全量前向。
    base 变体 13412 个参数 × 单次 loss 前向 ~25ms → 一个 epoch ≈ 11 分钟,
    7 个变体跑 4 epoch ≈ 7.2 小时。本模块把同样的 forward / loss 公式重写成
    torch 可微实现,用 autograd 拿真梯度,单个 epoch 降到亚秒级。

语义守恒(刻意不改的东西):
    - 每个变体的 forward 结构与 conv_variants 里 numpy 版逐层对应,包括
      relu 位置、stride-2 下采样、skip 投影的加法、SE 的 zero-pad 分支。
    - 三个 loss 的公式与 conv_frontier_v2 完全一致:
        * InfoNCE 的"正样本对"仍是 grids 列表里相邻的 (2i, 2i+1) —— 注意这
          并不是同一张图的两个增强视图,而是两张不同的合成图;这是原实现的
          口径,这里保留,只换求解梯度的方式。
        * 重建 loss 仍是 -mean(var(e)) + 0.1*relu(1-mean|e|)^2。
        * 微扰 loss 仍是 1 - <z(g), z(augment(g))>,样本数 ns=8。
    - 优化器仍是全批量 SGD(原数值版一个 epoch 走一步),步长 lr 不变。

已知与 numpy 版的差异(都朝好的方向):
    - 数值版在有限差分过程中每调用一次 loss 就会重新抽一次 rng(增强噪声直接
      进梯度)。这里一步之内的增强只抽一次,梯度是确定性的。
    - eval 用固定 seed 的 rng 抽微扰样本,变体之间可比;numpy 版沿用了训练 rng。

Public API:
    build_param_tensors(enc)   — numpy 编码器 → 可微 torch 参数(dict, 保持属性名)
    one_hot_batch(enc, grids)  — grid 列表 → (N, num_colors, H, W) tensor
    forward(enc, X, P)         — batched encode,结构与 conv_variants 同名变体一致
    combined_loss_t(...)       — 与 _combined_loss 同式的 torch 版
    sgd_step(enc, grids, rng, lr) — 单步全批量 SGD,参数写回 enc,返回步前 loss
    embed_arrays(enc, grids, P)   — no-grad 整批 embedding,用于与 numpy 对拍

多 epoch 循环仍由 conv_frontier_v2.run_frontier_benchmark 掌握,v2._sgd_step 委托到本模块。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .conv_variants import ConvVariant

_EPS = 1e-8


def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


def _conv_s2(x, w, b):
    """对应 _conv2d_stride2: 3×3, pad=1, stride=2。"""
    return F.conv2d(x, w, b, stride=2, padding=1)


def _conv_dil_s2(x, w, b, dilation=2):
    """对应 _conv2d_dilated: 3×3, dilation, pad=dilation, stride=2。

    numpy 版对越界索引做 clip;在我们的网格尺寸下最大索引 2*(h2-1)+2d ≤ h+3,
    正好落在 pad 后范围内,所以与 torch 的 zero-pad 等价。
    """
    return F.conv2d(x, w, b, stride=2, padding=dilation, dilation=dilation)


def _depthwise_s2(x, w, b):
    """对应 _depthwise_conv2d: weight (C,1,3,3), 逐通道, stride=2。"""
    c = x.shape[1]
    return F.conv2d(x, w.reshape(c, 1, 3, 3), b, stride=2, padding=1, groups=c)


def _pointwise(x, w, b):
    """对应 _pointwise_conv2d: weight (C_out,C_in,1,1)。"""
    co, ci = w.shape[0], w.shape[1]
    return F.conv2d(x, w.reshape(co, ci, 1, 1), b)


def _se(x, w1, b1, w2, b2, reduction: int):
    """对应 _channel_attention, batch 版。x: (N,c,H,W)。"""
    c = x.shape[1]
    squeezed = x.mean(dim=(2, 3))
    mid = max(1, c // reduction)
    if c <= mid:
        h = _relu(squeezed[:, :mid])
    else:
        h = _relu(F.linear(squeezed, w1, b1))
    exc = F.linear(h, w2, b2)
    if exc.shape[1] < c:
        exc = torch.cat([exc, exc.new_zeros(exc.shape[0], c - exc.shape[1])], dim=1)
    scale = torch.sigmoid(torch.clamp(exc, -20.0, 20.0))
    return x * scale[:, :, None, None]


def _fc_head(pooled, w_fc, b_fc):
    z = _relu(pooled @ w_fc + b_fc)
    return z / (z.norm(dim=-1, keepdim=True) + _EPS)


def build_param_tensors(enc: ConvVariant) -> Dict[str, torch.Tensor]:
    """把编码器上的 float ndarray 属性搬成需要梯度的 torch 参数。"""
    P: Dict[str, torch.Tensor] = {}
    for attr in dir(enc):
        v = getattr(enc, attr)
        if isinstance(v, np.ndarray) and v.dtype in (np.float32, np.float64):
            P[attr] = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).requires_grad_(True)
    return P


def param_names(enc: ConvVariant) -> List[str]:
    return [
        a for a in dir(enc)
        if isinstance(getattr(enc, a), np.ndarray) and getattr(enc, a).dtype in (np.float32, np.float64)
    ]


def one_hot_batch(enc: ConvVariant, grids: Sequence[np.ndarray]) -> torch.Tensor:
    """复刻 ConvVariant._one_hot(含 >32 缩放与 clip),返回 (N, num_colors, H, W)。

    要求同批 grid 尺寸一致(benchmark 生成的都是 16×16);不一致时由调用方分组。
    """
    c = int(enc.num_colors)
    out = []
    for g in grids:
        a = np.asarray(g, dtype=np.int16)
        if a.ndim == 2 and a.shape[0] >= 58:
            a = a[:58]
        h, w = a.shape
        if h > 32 or w > 32:
            ys = np.linspace(0, h, 32, endpoint=False).astype(np.int32)
            xs = np.linspace(0, w, 32, endpoint=False).astype(np.int32)
            a = a[ys][:, xs]
        eye = np.eye(c, dtype=np.float32)
        out.append(eye[np.clip(a, 0, c - 1)].transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out).astype(np.float32, copy=False))


def forward(enc: ConvVariant, X: torch.Tensor, P: Dict[str, torch.Tensor]) -> torch.Tensor:
    """batched encode,结构与 conv_variants 中同名变体一一对应。"""
    name = enc.name
    red = int(getattr(enc, "reduction", 4))

    if name in ("base", "wide"):
        h = _relu(_conv_s2(X, P["w1"], P["b1"]))
        h = _relu(_conv_s2(h, P["w2"], P["b2"]))
        h = _relu(_conv_s2(h, P["w3"], P["b3"]))
        return _fc_head(h.mean(dim=(2, 3)), P["w_fc"], P["b_fc"])

    if name == "dilated":
        h = _relu(_conv_dil_s2(X, P["w1"], P["b1"]))
        h = _relu(_conv_dil_s2(h, P["w2"], P["b2"]))
        h = _relu(_conv_dil_s2(h, P["w3"], P["b3"]))
        return _fc_head(h.mean(dim=(2, 3)), P["w_fc"], P["b_fc"])

    if name == "deep":
        h = _relu(_conv_s2(X, P["w1"], P["b1"]))
        h = _relu(_conv_s2(h, P["w2"], P["b2"]))
        h = _relu(_conv_s2(h, P["w3"], P["b3"]))
        h = _relu(_conv_s2(h, P["w4"], P["b4"]))
        return _fc_head(h.mean(dim=(2, 3)), P["w_fc"], P["b_fc"])

    if name == "separable":
        h = _relu(_pointwise(_depthwise_s2(X, P["dw1"], P["bd1"]), P["pw1"], P["bp1"]))
        h = _relu(_pointwise(_depthwise_s2(h, P["dw2"], P["bd2"]), P["pw2"], P["bp2"]))
        h = _relu(_pointwise(_depthwise_s2(h, P["dw3"], P["bd3"]), P["pw3"], P["bp3"]))
        return _fc_head(h.mean(dim=(2, 3)), P["w_fc"], P["b_fc"])

    if name == "residual":
        c = _relu(_conv_s2(X, P["w1"], P["b1"]))
        h = c + (X.mean(dim=(2, 3)) @ P["proj1"])[:, :, None, None]
        c = _relu(_conv_s2(h, P["w2"], P["b2"]))
        h = c + (h.mean(dim=(2, 3)) @ P["proj2"])[:, :, None, None]
        c = _relu(_conv_s2(h, P["w3"], P["b3"]))
        h = c + (h.mean(dim=(2, 3)) @ P["proj3"])[:, :, None, None]
        return _fc_head(h.mean(dim=(2, 3)), P["w_fc"], P["b_fc"])

    if name == "attention":
        h = _relu(_conv_s2(X, P["w1"], P["b1"]))
        h = _se(h, P["se1_w1"], P["se1_b1"], P["se1_w2"], P["se1_b2"], red)
        h = _relu(_conv_s2(h, P["w2"], P["b2"]))
        h = _se(h, P["se2_w1"], P["se2_b1"], P["se2_w2"], P["se2_b2"], red)
        h = _relu(_conv_s2(h, P["w3"], P["b3"]))
        return _fc_head(h.mean(dim=(2, 3)), P["w_fc"], P["b_fc"])

    raise NotImplementedError(f"torch 通道尚未移植变体 '{name}';_={ch}")


def embed_arrays(enc: ConvVariant, grids: Sequence[np.ndarray], P: Dict[str, torch.Tensor]) -> np.ndarray:
    """不做梯度的整批 embedding,返回 numpy (N, embed_dim)。"""
    with torch.no_grad():
        return forward(enc, one_hot_batch(enc, grids), P).numpy()


def contrastive_loss_t(E: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """InfoNCE,与 _contrastive_loss 同式(相邻两两成对)。"""
    n = E.shape[0]
    if n < 4:
        return E.new_zeros(())
    norm = E / (E.norm(dim=1, keepdim=True) + _EPS)
    sim = norm @ norm.T / temp
    total = E.new_zeros(())
    cnt = 0
    for i in range(0, n - 1, 2):
        pos = sim[i, i + 1]
        keep = torch.ones(n, dtype=torch.bool, device=E.device)
        keep[i] = False
        keep[i + 1] = False
        neg = sim[i][keep]
        total = total + (-pos + torch.logsumexp(neg, dim=0))
        cnt += 1
    return total / max(1, cnt)


def reconstruction_loss_t(E: torch.Tensor) -> torch.Tensor:
    """与 _reconstruction_loss 同式:-mean(var) + 0.1*relu(1-mean|z|)^2。"""
    if E.shape[0] == 0:
        return E.new_zeros(())
    tv = torch.stack([torch.var(e, unbiased=False) for e in E]).mean()
    mn = E.norm(dim=1).mean()
    return -tv + 0.1 * torch.clamp(1.0 - mn, min=0.0) ** 2


def perturb_loss_t(E: torch.Tensor, E_aug: torch.Tensor, idxs: np.ndarray) -> torch.Tensor:
    """与 _perturb_loss 同式:1 - <z(g), z(aug(g))>。"""
    if E_aug.shape[0] == 0:
        return E.new_zeros(())
    tot = E.new_zeros(())
    for j, i in enumerate(idxs):
        tot = tot + 1.0 - (E[int(i)] * E_aug[j]).sum()
    return tot / max(1, E_aug.shape[0])


def combined_loss_t(
    enc: ConvVariant,
    P: Dict[str, torch.Tensor],
    X: torch.Tensor,
    Xa: torch.Tensor,
    idxs: np.ndarray,
    wc: float = 0.5,
    wr: float = 0.3,
    wp: float = 0.2,
    temp: float = 0.07,
) -> torch.Tensor:
    """三个 loss 的加权和(需要梯度的那一版)。"""
    E = forward(enc, X, P)
    Ea = forward(enc, Xa, P) if Xa is not None and Xa.numel() else None
    lp = perturb_loss_t(E, Ea, idxs) if Ea is not None else E.new_zeros(())
    return wc * contrastive_loss_t(E, temp) + wr * reconstruction_loss_t(E) + wp * lp


def _augmented_view(enc: ConvVariant, grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from .conv_frontier_v2 import _augment_grid  # 延迟导入,避免循环依赖

    return _augment_grid(grid, rng)


def _make_views(enc: ConvVariant, grids: Sequence[np.ndarray], rng: np.random.Generator, ns: int):
    idxs = rng.choice(len(grids), min(ns, len(grids)), replace=False)
    idxs = np.asarray(idxs)
    if idxs.size == 0:
        return None, idxs
    views = [_augmented_view(enc, grids[int(i)], rng) for i in idxs]
    return one_hot_batch(enc, views), idxs


def sgd_step(
    enc: ConvVariant,
    grids: Sequence[np.ndarray],
    rng: np.random.Generator,
    lr: float = 0.01,
    ns: int = 8,
    loss_kwargs: Dict[str, float] | None = None,
) -> float:
    """全批量 autograd 走一步 SGD;参数就地写回 enc,返回步前 loss。"""
    loss_kwargs = loss_kwargs or {}
    P = build_param_tensors(enc)
    order = list(P.keys())
    opt = torch.optim.SGD([P[k] for k in order], lr=float(lr))
    X = one_hot_batch(enc, list(grids))
    Xa, idxs = _make_views(enc, grids, rng, ns)

    with torch.no_grad():
        before = float(combined_loss_t(enc, P, X, Xa, idxs, **loss_kwargs))
    loss = combined_loss_t(enc, P, X, Xa, idxs, **loss_kwargs)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    for k in order:
        setattr(enc, k, P[k].detach().numpy().copy())
    return before
