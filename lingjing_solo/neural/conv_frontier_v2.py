"""卷积++ 前沿可视化 v2 — 真实训练 + 对比学习 + ARC 风格数据。"""
from __future__ import annotations
import json, os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from .conv_variants import ConvVariant, VARIANT_NAMES, build_variant

@dataclass
class VariantRecord:
    name: str
    param_count: int
    loss_history: List[float] = field(default_factory=list)
    encode_times: List[float] = field(default_factory=list)
    train_time_s: float = 0.0

    @property
    def best_loss(self) -> float:
        return min(self.loss_history) if self.loss_history else float("inf")

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("inf")

    @property
    def tail_slope(self) -> float:
        if len(self.loss_history) < 4:
            return 0.0
        tail = self.loss_history[-4:]
        x = np.arange(len(tail), dtype=np.float64)
        y = np.array(tail, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        denom = float(dx @ dx)
        if denom < 1e-12:
            return 0.0
        return float((dx @ (y - ym)) / denom)

    @property
    def still_improving(self) -> bool:
        return self.tail_slope < -1e-4

    @property
    def slope_reliable(self) -> bool:
        """tail_slope 需要至少 4 个评估点才有意义。"""
        return len(self.loss_history) >= 4

    @property
    def verdict(self) -> str:
        if not self.slope_reliable:
            return f"样本不足({len(self.loss_history)}/4)"
        if self.tail_slope > 1e-4:
            return "↑回升"
        return "↓还在降" if self.still_improving else "—已收敛"

    @property
    def state_color(self) -> str:
        """绿=仍在降,红=已收敛,橙=loss 回升,灰=评估点太少判不了趋势。"""
        if not self.slope_reliable:
            return "#7f8c8d"
        if self.tail_slope > 1e-4:
            return "#d35400"
        return "#27ae60" if self.still_improving else "#c0392b"

    @property
    def avg_encode_ms(self) -> float:
        if not self.encode_times:
            return 0.0
        return float(np.mean(self.encode_times) * 1000)

@dataclass
class FrontierResult:
    records: Dict[str, VariantRecord] = field(default_factory=dict)
    epoch_count: int = 0
    grid_count: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    def sorted_by_loss(self) -> List[Tuple[str, VariantRecord]]:
        return sorted(self.records.items(), key=lambda kv: kv[1].best_loss)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_count": self.epoch_count,
            "grid_count": self.grid_count,
            "config": self.config,
            "variants": {
                name: {
                    "param_count": rec.param_count,
                    "best_loss": rec.best_loss,
                    "final_loss": rec.final_loss,
                    "tail_slope": rec.tail_slope,
                    "still_improving": rec.still_improving,
                    "verdict": rec.verdict,
                    "avg_encode_ms": rec.avg_encode_ms,
                    "train_time_s": rec.train_time_s,
                    "loss_history": rec.loss_history,
                }
                for name, rec in self.records.items()
            },
        }

    def save_json(self, path: str = "conv_frontier_result.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str = "conv_frontier_result.json") -> "FrontierResult":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        result = cls(
            epoch_count=data.get("epoch_count", 0),
            grid_count=data.get("grid_count", 0),
            config=data.get("config", {}),
        )
        for name, v in data.get("variants", {}).items():
            rec = VariantRecord(name=name, param_count=v.get("param_count", 0))
            rec.loss_history = v.get("loss_history", [])
            rec.train_time_s = v.get("train_time_s", 0.0)
            result.records[name] = rec
        return result

def _generate_arc_style_grids(n=40, size=16, n_colors=10, seed=42):
    """生成 ARC 风格的合成 grid。"""
    rng = np.random.default_rng(seed)
    grids = []
    for i in range(n):
        pt = i % 7
        g = np.zeros((size, size), dtype=np.int16)
        if pt == 0:
            g[:, :] = rng.integers(1, n_colors, (size, size)).astype(np.int16)
        elif pt == 1:
            half = size // 2
            left = rng.integers(1, n_colors, (size, half)).astype(np.int16)
            g[:, :half] = left
            g[:, size - half:] = left[:, ::-1]
        elif pt == 2:
            half = size // 2
            top = rng.integers(1, n_colors, (half, size)).astype(np.int16)
            g[:half, :] = top
            g[size - half:, :] = top[::-1, :]
        elif pt == 3:
            cx, cy = size // 2, size // 2
            r = rng.integers(2, size // 3)
            c = rng.integers(1, n_colors)
            ys, xs = np.ogrid[:size, :size]
            g[((xs - cx) ** 2 + (ys - cy) ** 2) <= r ** 2] = c
        elif pt == 4:
            co, ci = rng.integers(1, n_colors), rng.integers(1, n_colors)
            while ci == co:
                ci = rng.integers(1, n_colors)
            t = rng.integers(1, 3)
            g[:, :] = co
            g[t:-t, t:-t] = 0
            t2 = t + rng.integers(1, 3)
            if t2 < size // 2:
                g[t2:-t2, t2:-t2] = ci
        elif pt == 5:
            c = rng.integers(1, n_colors)
            sw = rng.integers(1, 4)
            for col in range(0, size, sw * 2):
                g[:, col:min(col + sw, size)] = c if rng.random() > 0.5 else 0
        else:
            for _ in range(rng.integers(2, 5)):
                cx, cy = rng.integers(2, size - 2), rng.integers(2, size - 2)
                w, h = rng.integers(2, 5), rng.integers(2, 5)
                c = rng.integers(1, n_colors)
                g[cy:min(cy + h, size), cx:min(cx + w, size)] = c
        grids.append(g)
    return grids

def _augment_grid(grid, rng, strength=0.15):
    """数据增强。"""
    g = grid.copy()
    if rng.random() < strength:
        g = np.clip(g + rng.integers(0, 2, g.shape).astype(np.int16), 0, 15).astype(np.int16)
    if rng.random() < strength * 0.5:
        s = rng.integers(-1, 2)
        if s != 0:
            g = np.roll(g, shift=s, axis=rng.integers(0, 2))
    return g

def _compute_embeddings(encoder, grids):
    """计算所有 grid 的 embedding。"""
    t0 = time.perf_counter()
    embs = [encoder.encode(g) for g in grids]
    return embs, time.perf_counter() - t0

def _contrastive_loss(embs, temp=0.07):
    """InfoNCE 对比损失。"""
    n = len(embs)
    if n < 4:
        return 0.0
    arr = np.stack(embs)
    norm = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T / temp
    total, cnt = 0.0, 0
    for i in range(0, n - 1, 2):
        if i + 1 >= n:
            break
        pos = sim[i, i + 1]
        mask = np.ones(n, dtype=bool)
        mask[i] = mask[i + 1] = False
        neg = sim[i, mask]
        lse = np.logaddexp(0, neg.max()) + np.log(np.exp(neg - neg.max()).sum() + 1e-12)
        total += float(-pos + lse)
        cnt += 1
    return total / max(1, cnt)

def _reconstruction_loss(embs):
    """重建损失。"""
    if not embs:
        return 0.0
    tv = sum(float(np.var(e)) for e in embs)
    mn = float(np.mean([np.linalg.norm(e) for e in embs]))
    return -tv / len(embs) + 0.1 * max(0, 1 - mn) ** 2

def _perturb_loss(enc, grids, rng, ns=8):
    """微扰一致性损失。"""
    idxs = rng.choice(len(grids), min(ns, len(grids)), replace=False)
    tot, cnt = 0.0, 0
    for i in idxs:
        g = grids[i]
        z1 = enc.encode(g)
        z2 = enc.encode(_augment_grid(g, rng))
        n = min(len(z1), len(z2))
        tot += 1 - float(z1[:n] @ z2[:n])
        cnt += 1
    return tot / max(1, cnt)

def _combined_loss(enc, grids, rng, wc=0.5, wr=0.3, wp=0.2):
    """组合 loss。"""
    embs, _ = _compute_embeddings(enc, grids)
    return wc * _contrastive_loss(embs) + wr * _reconstruction_loss(embs) + wp * _perturb_loss(enc, grids, rng)

def _sgd_step_numeric(enc, grids, rng, lr=0.01, lf=None):
    """数值梯度下降一步(有限差分,慢;保留用于与 autograd 对拍)。"""
    if lf is None:
        lf = lambda e: _combined_loss(e, grids, rng)
    bl = lf(enc)
    eps = 1e-4
    for an in dir(enc):
        a = getattr(enc, an)
        if not isinstance(a, np.ndarray) or a.dtype not in (np.float32, np.float64):
            continue
        grad = np.zeros_like(a)
        it = np.nditer(a, ["multi_index"], ["readwrite"])
        while not it.finished:
            ix = it.multi_index
            ov = a[ix]
            a[ix] = ov + eps
            lp = lf(enc)
            a[ix] = ov - eps
            lm = lf(enc)
            a[ix] = ov
            grad[ix] = (lp - lm) / (2 * eps)
            it.iternext()
        setattr(enc, an, a - lr * grad)
    return bl

_TORCH_GRAD_FALLBACK_LOGGED = False

def _use_torch_grad() -> bool:
    """默认用 autograd;设 ARC_CONV_FRONTIER_GRAD=numeric 可切回有限差分。"""
    return os.environ.get("ARC_CONV_FRONTIER_GRAD", "torch").strip().lower() != "numeric"

def _sgd_step(enc, grids, rng, lr=0.01, lf=None):
    """梯度下降一步:默认 torch autograd,可选数值有限差分。"""
    global _TORCH_GRAD_FALLBACK_LOGGED
    if _use_torch_grad():
        try:
            from .conv_frontier_torch import sgd_step as _torch_sgd_step
        except Exception as exc:  # torch 未安装等
            if not _TORCH_GRAD_FALLBACK_LOGGED:
                print(f"[frontier] torch 通道不可用({exc}),回退数值梯度")
                _TORCH_GRAD_FALLBACK_LOGGED = True
        else:
            return _torch_sgd_step(enc, grids, rng, lr=lr)
    return _sgd_step_numeric(enc, grids, rng, lr=lr, lf=lf)

def run_frontier_benchmark(variant_names=VARIANT_NAMES, grids=None, n_grids=40, train_epochs=10, eval_epochs=5, seed=7, verbose=True, lr=0.005):
    """跑所有变体的 benchmark。"""
    if grids is None:
        grids = _generate_arc_style_grids(n=n_grids, seed=seed)
    res = FrontierResult(
        epoch_count=max(1, train_epochs // eval_epochs),
        grid_count=len(grids),
        config={"train_epochs": train_epochs, "eval_every": eval_epochs, "lr": lr, "seed": seed, "n_grids": len(grids)},
    )
    for vn in variant_names:
        if verbose:
            print(f"[frontier] 变体 {vn:12s} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        enc = build_variant(vn, seed=seed)
        vidx = VARIANT_NAMES.index(vn) if vn in VARIANT_NAMES else 0
        rng = np.random.default_rng(seed * 100 + vidx)
        rec = VariantRecord(name=vn, param_count=enc.param_count())
        for ep in range(train_epochs):
            _sgd_step(enc, grids, rng, lr=lr)
            if (ep + 1) % eval_epochs == 0 or ep == train_epochs - 1:
                eval_rng = np.random.default_rng(seed + 991)
                rec.loss_history.append(float(_combined_loss(enc, grids, eval_rng)))
                te = time.perf_counter()
                for g in grids[:5]:
                    enc.encode(g)
                rec.encode_times.append(time.perf_counter() - te)
        rec.train_time_s = time.perf_counter() - t0
        res.records[vn] = rec
        if verbose:
            print(
                f"best={rec.best_loss:.6f}  slope={rec.tail_slope:+.6f}  {rec.verdict}  "
                f"params={rec.param_count:>6d}  {rec.train_time_s:.2f}s"
            )
    return res

def plot_frontier(res, sp=None):
    """前沿曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ranked = res.sorted_by_loss()
    names = [n for n, _ in ranked]
    losses = [r.best_loss for _, r in ranked]
    recs = [r for _, r in ranked]
    slopes = [r.tail_slope for _, r in ranked]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5), dpi=140)
    cols = [r.state_color for r in recs]
    bars = ax.bar(range(len(names)), losses, color=cols, width=0.6, zorder=3)
    for i, (b, s, r) in enumerate(zip(bars, slopes, recs)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(losses) * 0.01,
            f"slope={s:+.5f}\n{r.verdict}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=r.state_color,
            fontweight="bold",
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Best Loss", fontsize=11)
    ax.set_title("卷积++前沿曲线:v2真实训练\n绿=改善 红=收敛 橙=回升 灰=评估点不足", fontsize=13, pad=12)
    ax.grid(axis="y", color="#ddd", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if sp:
        fig.savefig(sp, bbox_inches="tight", facecolor="white")
        print(f"[frontier] 前沿曲线→{sp}")
    plt.close(fig)

def plot_training_curves(res, sp=None):
    """训练曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    pal = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#c0392b", "#16a085", "#f39c12", "#2c3e50"]
    for idx, (n, r) in enumerate(res.sorted_by_loss()):
        c = pal[idx % len(pal)]
        done = r.slope_reliable and not r.still_improving
        ls = "--" if done else "-"
        lw = 1.2 if done else 2.0
        ax.plot(r.loss_history, label=f"{n}(best={r.best_loss:.5f})", color=c, ls=ls, lw=lw, marker="o", ms=3)
    ax.set_xlabel(f"评估点(每{res.config.get('eval_every', '?')}轮)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("v2训练曲线", fontsize=13, pad=10)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(color="#eee", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    if sp:
        fig.savefig(sp, bbox_inches="tight", facecolor="white")
        print(f"[frontier] 训练曲线→{sp}")
    plt.close(fig)

def plot_frontier_dashboard(res, sp=None):
    """一页三图 dashboard。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=140)
    ranked = res.sorted_by_loss()
    pal = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#c0392b", "#16a085", "#f39c12", "#2c3e50"]

    # 前沿柱状图
    ax = axes[0]
    names = [n for n, _ in ranked]
    losses = [r.best_loss for _, r in ranked]
    recs = [r for _, r in ranked]
    slopes = [r.tail_slope for _, r in ranked]
    cols = [r.state_color for r in recs]
    bars = ax.bar(range(len(names)), losses, color=cols, width=0.6, zorder=3)
    short = {"↓还在降": "↓", "—已收敛": "—", "↑回升": "↑"}
    for i, (b, s, r) in enumerate(zip(bars, slopes, recs)):
        tag = short.get(r.verdict, "?")
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(losses) * 0.01,
            f"{s:+.5f}{tag}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=r.state_color,
            fontweight="bold",
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Best Loss")
    ax.set_title("前沿\n绿=降 红=收敛 橙=回升 灰=点不足", fontsize=11)
    ax.grid(axis="y", color="#ddd", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)

    # 训练曲线
    ax = axes[1]
    for idx, (n, r) in enumerate(ranked):
        c = pal[idx % len(pal)]
        done = r.slope_reliable and not r.still_improving
        ax.plot(r.loss_history, label=n, color=c, ls="--" if done else "-", lw=1.5, marker="o", ms=2.5)
    ax.set_xlabel("Eval")
    ax.set_ylabel("Loss")
    ax.set_title("训练曲线", fontsize=11)
    ax.legend(fontsize=7.5, loc="best", ncol=2)
    ax.grid(color="#eee", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # 参数量 vs loss 散点
    ax = axes[2]
    for idx, (n, r) in enumerate(ranked):
        c = r.state_color
        m = "o" if not r.slope_reliable else ("v" if r.still_improving else "s")
        ax.scatter(r.param_count, r.best_loss, color=c, marker=m, s=120, zorder=3, ec="white", lw=0.8)
        ax.annotate(n, (r.param_count, r.best_loss), textcoords="offset points", xytext=(6, 4), fontsize=8, color=c)
    ax.set_xlabel("参数量")
    ax.set_ylabel("Best Loss")
    ax.set_title("参数vsLoss", fontsize=11)
    ax.grid(color="#eee", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("v2 Dashboard", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    if sp:
        fig.savefig(sp, bbox_inches="tight", facecolor="white")
        print(f"[frontier] Dashboard→{sp}")
    plt.close(fig)

def quick_run_and_plot(variant_names=VARIANT_NAMES, n_grids=30, train_epochs=8, eval_epochs=2, save_dir=".", lr=0.005):
    """一键跑 benchmark + 画图。"""
    sdp = Path(save_dir)
    sdp.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("  卷积++v2 Benchmark (真实训练+对比学习)")
    print("=" * 65)
    print(f"  配置: grids={n_grids}, epochs={train_epochs}, eval={eval_epochs}, lr={lr}")

    res = run_frontier_benchmark(
        variant_names, n_grids=n_grids, train_epochs=train_epochs, eval_epochs=eval_epochs, lr=lr, verbose=True
    )

    res.save_json(str(sdp / "conv_frontier_v2_result.json"))
    plot_frontier(res, str(sdp / "conv_frontier_v2_curve.png"))
    plot_training_curves(res, str(sdp / "conv_frontier_v2_training.png"))
    plot_frontier_dashboard(res, str(sdp / "conv_frontier_v2_dashboard.png"))

    print("\n" + "=" * 65)
    print("  前沿排序:")
    print("=" * 65)
    for i, (n, r) in enumerate(res.sorted_by_loss(), 1):
        if not r.slope_reliable:
            mark = "⚪"
        elif r.tail_slope > 1e-4:
            mark = "🟠"
        else:
            mark = "🟢" if r.still_improving else "🔴"
        a = mark + r.verdict
        print(
            f"  {i}. {n:12s} best={r.best_loss:.6f} slope={r.tail_slope:+.6f} "
            f"params={r.param_count:>6d} time={r.train_time_s:.2f}s {a}"
        )

    print(f"\n  输出: {sdp.resolve()}")
    return res