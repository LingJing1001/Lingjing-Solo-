"""校准层。这是整件事里唯一"能自己写、也真的该自己写"的部分。

Jev 的方法叫 RLCD（Reinforcement Learning for **Calibrated Decisions**）——
它不训模型输出文字，只训模型输出**决策 + 配得上这个决策的概率**。真正有用的
不是那个模型，是"概率要等于频率"这条约束。

我们只校准一件事：**第一名的概率**。因为 agent 做决策时只读第一名。

    p̂ = sigmoid( a · logit(p_top) + b )

两个参数，`a` 管斜率（对应温度 T = 1/a），`b` 管偏置。`b` 是必须的：
后端如果系统性偏保守（总说 0.6、实际 0.9），只有斜率补偿不了，要有个平移。

为什么不用原来的多项温度缩放：一个键的 9 票会产生 9 次高度相关的更新，
T 被单一键拖到边界（实测 ar25 上 ECE 从 0.17 涨到 0.41）。改成"第一名对错"
这个二分类之后，每次更新就是一个独立样本，而且直接优化我们唯一在意的量。

而且这个变换是**单调**的：不改变 argmax，所以它永远不会替 agent 改主意 ——
只改变"我有多确定"，从而只影响"要不要再问一次"。

**样本不足就恒等**（`min_n`）。这条不是保守，是算术：用 8 个点拟合出来的
校准曲线，在没见过的局面上的误差比不校准更大。AOP 的 AUC 0.513 就是
"样本不够还硬训"的下场。
"""
from __future__ import annotations

import math
from typing import Iterable, Optional


def beta_mean(successes: float, trials: float, prior: float = 1.0) -> float:
    """Beta(prior, prior) 先验下的后验均值。

    为什么要它而不是 `successes / trials`：1 票就是 100% 显然是假的，0 票就是
    0% 也是假的。加一个均匀先验，1 票一致得到 0.67、2 票 0.75、3 票 0.8 ——
    证据越多越接近 1，但永远不到 1。这正是"结论可以被推翻"的数学形式。
    """
    s = max(0.0, float(successes))
    n = max(0.0, float(trials))
    if n < s:
        n = s
    return (s + prior) / (n + 2.0 * prior)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class Temperature:
    """第一名概率的校准器：`p̂ = sigmoid(a·logit(p) + b)`。

    命名沿用"温度"（T = 1/a）：T > 1 表示原来太自信、要压平；T < 1 表示太保守。
    """

    def __init__(self, min_n: int = 12, lr: float = 0.08,
                 slope_lo: float = 0.2, slope_hi: float = 5.0) -> None:
        self.min_n = max(1, int(min_n))
        self.lr = float(lr)
        self.slope_lo = float(slope_lo)
        self.slope_hi = float(slope_hi)
        self.a = 1.0
        self.b = 0.0
        self.n = 0
        self.nll_sum = 0.0

    # ── 读数 ────────────────────────────────────────────────
    @property
    def temperature(self) -> float:
        """T = 1/a。样本不够时恒为 1。"""
        if not self.active:
            return 1.0
        return 1.0 / self.a

    @property
    def active(self) -> bool:
        return self.n >= self.min_n

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / self.n if self.n else float("nan")

    # ── 变换 ────────────────────────────────────────────────
    def calibrate_top(self, p: float) -> float:
        if not self.active:
            return p
        return _sigmoid(self.a * _logit(p) + self.b)

    def apply(self, dist: dict) -> dict:
        """保 argmax，只改"第一名有多可信"。

        除第一名之外的选项按原有比例分摊剩下的概率，所以**排序永远不变** ——
        这个变换在任何情况下都不会让 agent 换一个选项。它只改变"要不要再问一次"。
        """
        if not dist:
            return {}
        total = sum(v for v in dist.values() if v > 0)
        if total <= 0:
            return {}
        p = {k: v / total for k, v in dist.items() if v > 0}
        if len(p) < 2 or not self.active:
            return p
        top = max(p, key=lambda k: p[k])
        p_top = p[top]
        want = self.calibrate_top(p_top)
        # 夹住：低于"均分 + 余量"时第一名会被翻掉，那就不是校准而是改主意了
        n = len(p)
        want = min(max(want, 1.0 / n + 0.05), 1.0 - 1e-6)
        out = {top: want}
        others = {k: v for k, v in p.items() if k != top}
        s_other = sum(others.values())
        if s_other > 0:
            for k, v in others.items():
                out[k] = (1.0 - want) * v / s_other
        s = sum(out.values()) or 1.0
        return {k: v / s for k, v in out.items()}

    def apply_one(self, p: float) -> float:
        """标量概率（Noul 用）。"""
        return self.calibrate_top(float(p))

    # ── 学 ──────────────────────────────────────────────────
    def learn(self, dist: dict, correct: str) -> Optional[float]:
        """`correct` 是本次实际成立的选项。返回本次的 NLL（便于盯曲线）。

        只用"第一名对不对"这一个信号，因为决策也只用到这一步。
        """
        if not dist or correct not in dist:
            return None
        total = sum(v for v in dist.values() if v > 0)
        if total <= 0:
            return None
        p = {k: v / total for k, v in dist.items()}
        top = max(p, key=lambda k: p[k])
        p_top = min(max(p[top], 1e-4), 1 - 1e-4)
        ok = 1.0 if top == correct else 0.0

        z = self.a * _logit(p_top) + self.b
        pc = _sigmoid(z) if ok else 1.0 - _sigmoid(z)
        nll = -math.log(min(max(pc, 1e-9), 1.0))

        self.n += 1
        self.nll_sum += nll
        if self.n < self.min_n:
            return nll          # 门槛之前只计数，一个参数都不动

        g = _sigmoid(z) - ok     # BCE 对 z 的梯度
        self.a -= self.lr * g * _logit(p_top)
        self.b -= self.lr * g
        self.a = min(max(self.a, self.slope_lo), self.slope_hi)
        self.b = min(max(self.b, -6.0), 6.0)
        return nll

    # ── 可证伪 ──────────────────────────────────────────────
    def ece(self, pairs: Iterable[tuple[float, int]], bins: int = 8) -> float:
        """Expected Calibration Error：说 0.7 的事，是不是真的约 70% 发生。"""
        buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
        for p, ok in pairs:
            i = min(bins - 1, max(0, int(p * bins)))
            buckets[i].append((p, ok))
        n = sum(len(b) for b in buckets)
        if n == 0:
            return float("nan")
        err = 0.0
        for b in buckets:
            if not b:
                continue
            conf = sum(p for p, _ in b) / len(b)
            acc = sum(ok for _, ok in b) / len(b)
            err += len(b) / n * abs(conf - acc)
        return err

    # ── 落地 ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "n": self.n, "nll_sum": self.nll_sum,
                "min_n": self.min_n, "lr": self.lr}

    @classmethod
    def from_dict(cls, d: dict) -> "Temperature":
        t = cls(min_n=int(d.get("min_n", 12)), lr=float(d.get("lr", 0.08)))
        t.a = float(d.get("a", 1.0))
        t.b = float(d.get("b", 0.0))
        t.n = int(d.get("n", 0))
        t.nll_sum = float(d.get("nll_sum", 0.0))
        return t

    def text(self) -> str:
        if not self.active:
            return f"T=1.00（样本 {self.n}/{self.min_n}，未校准）"
        return (f"T={self.temperature:.2f}(a={self.a:.2f},b={self.b:+.2f}"
                f"，样本 {self.n})")
