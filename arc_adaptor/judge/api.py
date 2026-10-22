"""问题与答案的**契约**。这里没有后端，也没有任何概率公式。

一个 `Choice` 只描述"在这些选项里挑一个"，附带的 `support/oppose` 是**原始
计数**（我见到几次支持它、几次反对它），不是概率。把它们变成概率是后端的事，
把概率变准是校准层的事。

为什么坚持只传计数：一旦某个位置的"结论"以概率常量形式写进源码，它就再也
不会因为新证据而改变。计数是证据，常量是结论 —— 这个项目只要前者。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

# 三种原语，对应 Jev 的三类问题
CHOICE = "choice"    # 从列表里选一个
SCORE = "score"      # 按标准打分
NOUL = "noul"        # 这句话真还是假


@dataclass(frozen=True)
class Choice:
    """从 `options` 里选一个。

    `support` / `oppose` 缺某个选项时按 0 处理 —— "没证据"和"有反证"是两件事，
    前者应该退到均匀先验，后者应该被压下去。

    `options` 不限定字符串：位移 `(dr, dc)` 这种结构化对象也能直接当选项，
    省掉"转成文本再解析回来"这一层容易出错的翻译。
    """
    name: str
    options: tuple
    support: Mapping = field(default_factory=dict)
    oppose: Mapping = field(default_factory=dict)
    note: str = ""
    kind: str = CHOICE


@dataclass(frozen=True)
class Noul:
    """判断一句话成立的概率。`support/oppose` 同样是计数。"""
    name: str
    statement: str
    support: float = 0.0
    oppose: float = 0.0
    note: str = ""
    kind: str = NOUL


@dataclass(frozen=True)
class Score:
    """给每个条目打 0~1 的分。用于"几条候选目标哪个更像"这种排序。"""
    name: str
    items: tuple
    support: Mapping = field(default_factory=dict)
    oppose: Mapping = field(default_factory=dict)
    note: str = ""
    kind: str = SCORE


Question = Choice | Noul | Score


@dataclass
class Answer:
    """结构化答案。没有自然语言，只有能被代码直接分支的东西。"""
    name: str
    kind: str
    choice: Optional[object] = None
    probabilities: dict = field(default_factory=dict)
    confidence: float = 0.0
    # 校准前后的分布都留着：出问题时能判断是"后端先验错了"还是"校准错了"
    raw: dict = field(default_factory=dict)
    calibrated: bool = False
    note: str = ""

    # ── 使用方 ──────────────────────────────────────────────
    def top(self, k: int = 2) -> list[tuple]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:k]

    def margin(self) -> float:
        """第一名与第二名的差。比 confidence 更能说明"这件事真的有结论吗"。"""
        t = self.top(2)
        if len(t) < 2:
            return t[0][1] if t else 0.0
        return t[0][1] - t[1][1]

    def decided(self, at: float = 0.5) -> bool:
        return self.choice is not None and self.confidence >= at

    def text(self) -> str:
        parts = " ".join(f"{k}={v:.2f}" for k, v in self.top(4))
        tag = "已校准" if self.calibrated else "未校准"
        return f"{self.name}: {self.choice} ({parts}) [{tag}]"

    def __str__(self) -> str:  # pragma: no cover - 便于调试
        return self.text()
