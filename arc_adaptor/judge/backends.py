"""后端：把「原始计数」变成「先验分布」。

后端是**唯一**可以换的地方，也是唯一允许外来的地方：

    CountBackend     纯计数 → Beta 后验。默认，零依赖，不需要训练。
    CallableBackend  任意函数 (Question) -> dict。接模型、接外部 API、接文件表。
    ChainBackend     依次尝试，谁先给出非均匀分布就听谁的。

换后端不影响校准层。这条分界很重要：以后 128K 模型到位时，`CallableBackend`
里换成"模型现场读画面给概率"即可，温度校准、ECE、样本门槛全部原样保留 ——
那时候它才是真正跑在我们自己转移分布上的 Jev。
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

from .api import CHOICE, NOUL, SCORE, Choice, Noul, Question, Score
from .calibration import beta_mean


class Backend(Protocol):
    name: str

    def distribution(self, q: Question) -> dict[str, float]:
        ...


class CountBackend:
    """把计数折成 Beta 后验再归一。

    没有证据的选项得到 0.5（均匀先验），有证据的选项按票数向 0 或 1 推。
    3 票支持 0 票反对 → 0.8；3 支持 3 反对 → 0.5。这是"证据的量"在起作用，
    而不是"哪个看起来更合理"。
    """

    name = "count"

    def distribution(self, q: Question) -> dict[str, float]:
        if isinstance(q, Choice):
            raw = {o: beta_mean(q.support.get(o, 0.0),
                                 q.support.get(o, 0.0) + q.oppose.get(o, 0.0))
                   for o in q.options}
        elif isinstance(q, Score):
            raw = {o: beta_mean(q.support.get(o, 0.0),
                                q.support.get(o, 0.0) + q.oppose.get(o, 0.0))
                   for o in q.items}
        elif isinstance(q, Noul):
            p = beta_mean(q.support, q.support + q.oppose)
            raw = {"true": p, "false": 1.0 - p}
        else:  # pragma: no cover - 类型已收窄
            return {}
        return _norm(raw)


class CallableBackend:
    """把任意函数包成后端。函数返回未归一的分数字典即可。

    这个插槽是给"未来的模型"准备的，不是占位符：现在就能用它把一份离线学好的
    先验表接进来，也能用它把冠军视图对拍出来的结论接进来。
    """

    def __init__(self, fn: Callable[[Question], Optional[dict[str, float]]],
                 name: str = "callable") -> None:
        self.fn = fn
        self.name = name

    def distribution(self, q: Question) -> dict[str, float]:
        out = self.fn(q)
        if not out:
            return {}
        return _norm({k: float(v) for k, v in out.items() if float(v) > 0})


class ChainBackend:
    """按顺序问。前一个返回空（或全 0）就落到下一个。

    典型用法：跨关携带的结论表在前，纯计数在后。这样"上一关学过的东西"
    优先，"这一关现算的证据"兜底；两者都没有时退回均匀先验。
    """

    name = "chain"

    def __init__(self, *backends: Backend) -> None:
        self.backends = [b for b in backends if b is not None]

    def distribution(self, q: Question) -> dict[str, float]:
        for b in self.backends:
            d = b.distribution(q)
            if d and any(v > 0 for v in d.values()):
                return d
        return {}


def _norm(d: dict[str, float]) -> dict[str, float]:
    s = sum(v for v in d.values() if v > 0)
    if s <= 0:
        return {}
    return {k: (v / s if v > 0 else 0.0) for k, v in d.items()}


def uniform(q: Question) -> dict[str, float]:
    """没有任何后端能回答时的兜底：均匀分布。"""
    if isinstance(q, Choice):
        keys = q.options
    elif isinstance(q, Score):
        keys = q.items
    elif isinstance(q, Noul):
        keys = ("true", "false")
    else:
        keys = ()
    return {k: 1.0 / len(keys) for k in keys} if keys else {}


__all__ = ["Backend", "CallableBackend", "ChainBackend", "CountBackend",
           "CHOICE", "NOUL", "SCORE", "uniform"]
