"""Jev 风格的**判断原语**。

Jev 的产品形态是三件事：类型化的提问、结构化的答案（带概率分布）、
以及**校准过**的置信度。这三件事都能自己写；写不出来的是它背后的底座
模型与合成训练数据 —— 那部分不做，也不假装做了。

这个包只负责：

    question ──▶ 后端给先验分布 ──▶ 温度校准 ──▶ Answer(choice, probs, confidence)

三条纪律：

1. **后端可插拔**。默认后端只吃计数，不吃关卡知识；想接模型/外部 API 就
   换一个 `CallableBackend`，校准层原样保留。
2. **样本不足就恒等**。`min_n` 没到之前不做任何校准 —— 1486 条样本、
   36 个正例就敢训一个预测器，结果就是 AUC 0.513 的 AOP。宁可"没有校准"，
   也不要"假的校准"。
3. **概率必须能被证伪**。每个 Answer 都留着未校准的先验，`Judge.ece()`
   随时能算出预测与实际的偏差；对不上就是错的。
"""
from .api import Answer, Choice, Noul, Score, Question
from .backends import Backend, CallableBackend, CountBackend
from .calibration import Temperature, beta_mean
from .judge import DECIDE_AT, Judge

__all__ = [
    "Answer",
    "Backend",
    "CallableBackend",
    "Choice",
    "CountBackend",
    "DECIDE_AT",
    "Judge",
    "Noul",
    "Question",
    "Score",
    "Temperature",
    "beta_mean",
]
