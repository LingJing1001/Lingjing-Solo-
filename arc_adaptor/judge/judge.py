"""判官：三原语的统一入口。

    judge.ask(Choice(...))      → Answer
    judge.ask_many([q1, q2])    → [Answer, ...]
    judge.learn(answer, 真相)    → 更新温度校准
    judge.ece()                 → 校准得准不准（可证伪）

设计上只做两件事：**不让后端看到关卡知识**，**不让结论写进源码**。
后端只吃计数，计数由调用方从运行时的证据里现算；`Answer` 只活在一次决策里，
跨关携带的是**温度**（一个关于"我到底多可靠"的标量），不是任何具体判断。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from .api import CHOICE, NOUL, SCORE, Answer, Choice, Noul, Question, Score
from .backends import Backend, CountBackend, uniform
from .calibration import Temperature

# 行动的默认门槛。注意：这是**代价**问题，不是认识论问题 —— 说 "0.6" 不等于
# 说 "0.6 以下就是假的"。任何调用方都可以传自己的门槛，也可以 set_threshold。
DECIDE_AT = 0.6


class Judge:
    """一个有记忆的判断器：温度参数跨关、跨盘保留，其余全部现算。"""

    def __init__(self, backend: Optional[Backend] = None,
                 min_n: int = 12, path: Optional[str] = None) -> None:
        self.backend: Backend = backend or CountBackend()
        self.cal: dict[str, Temperature] = {
            CHOICE: Temperature(min_n=min_n),
            NOUL: Temperature(min_n=min_n),
            SCORE: Temperature(min_n=min_n),
        }
        self.thresholds: dict[str, float] = {}
        self.log: list[dict] = []
        self.asked = 0
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self.load(self.path)
        # 从文件里带过来的样本量。日志只活在本进程，参数却是跨进程累积的，
        # 不分开算就会出现"反馈 15 条"配 T=1.00(a=…,样本 45) 这种自相矛盾的行。
        self._prior_n = sum(t.n for t in self.cal.values())

    # ── 问 ──────────────────────────────────────────────────
    def ask(self, q: Question) -> Answer:
        self.asked += 1
        kind = getattr(q, "kind", CHOICE)
        raw = self.backend.distribution(q)
        if not raw:
            raw = uniform(q)
        cal = self.cal.setdefault(kind, Temperature())
        probs = cal.apply(raw) if len(raw) > 1 else dict(raw)
        # 单选项时 apply 无意义；保底归一
        s = sum(probs.values())
        if s > 0 and abs(s - 1.0) > 1e-9:
            probs = {k: v / s for k, v in probs.items()}

        if isinstance(q, Noul):
            choice = "true" if probs.get("true", 0.0) >= 0.5 else "false"
        else:
            choice = max(probs, key=lambda k: probs[k]) if probs else None
        conf = probs.get(choice, 0.0) if choice else 0.0
        return Answer(name=q.name, kind=kind, choice=choice,
                      probabilities=probs, confidence=conf,
                      raw=dict(raw), calibrated=cal.active,
                      note=getattr(q, "note", ""))

    def ask_many(self, qs: Iterable[Question]) -> list[Answer]:
        """一次问一批。接口与 Jev 一致（它们"加问题几乎不加时间"；
        本地后端本来就是顺序的，但调用方看到的形状相同）。"""
        return [self.ask(q) for q in qs]

    def yes(self, q: Noul, at: Optional[float] = None) -> bool:
        """Noul 的常见用法：这句话成不成立。"""
        a = self.ask(q)
        return a.choice == "true" and a.confidence >= (
            self.thresholds.get(q.name, DECIDE_AT) if at is None else at)

    def pick(self, q: Choice, at: Optional[float] = None) -> Optional[str]:
        """Choice 的常见用法：有明确赢家就返回它，否则 None（＝别急着定）。"""
        a = self.ask(q)
        gate = self.thresholds.get(q.name, DECIDE_AT) if at is None else at
        if a.confidence < gate:
            return None
        return a.choice

    # ── 反馈：现实打脸 ──────────────────────────────────────
    def learn(self, ans: Answer, correct: str) -> Optional[float]:
        """告诉它"这次实际成立的是哪个"。这是唯一让校准变准的入口。

        只在**结果确定**时调用：过关事件、对拍得到的真值。用猜出来的结果
        去校准，等于用错答案训练，比不校准更糟。
        """
        cal = self.cal.get(ans.kind)
        if cal is None:
            return None
        nll = cal.learn(ans.raw, correct)
        if nll is not None:
            self.log.append({"kind": ans.kind, "name": ans.name,
                             "raw": ans.raw, "correct": correct,
                             "confidence": ans.confidence, "nll": nll})
            if len(self.log) > 2000:
                self.log = self.log[-2000:]
        return nll

    def set_threshold(self, name: str, value: float) -> None:
        self.thresholds[name] = float(value)

    # ── 可证伪 ──────────────────────────────────────────────
    def ece(self, kind: Optional[str] = None) -> float:
        """用**当前**参数重算所有历史样本的校准置信度，再算可靠性误差。

        为什么不是直接用答题时记下的 confidence：那时参数还没学到，拿旧置信度
        算出来的误差反映的是"历史"，不是"现在准不准"。
        """
        key = kind or CHOICE
        cal = self.cal.get(key)
        if cal is None:
            return float("nan")
        pairs: list[tuple[float, int]] = []
        for r in self.log:
            if kind is not None and r["kind"] != kind:
                continue
            raw = r["raw"] or {}
            total = sum(v for v in raw.values() if v > 0)
            if total <= 0:
                continue
            p_raw = max(raw.values()) / total
            ok = 1 if max(raw, key=lambda k: raw[k]) == r["correct"] else 0
            pairs.append((cal.calibrate_top(p_raw), ok))
        return cal.ece(pairs) if pairs else float("nan")

    def text(self) -> str:
        """一行摘要，挂在每次状态快照里。"""
        n = self._prior_n + len(self.log)
        if not n:
            return f"后端 {self.backend.name}，{self.asked} 次提问，还没有反馈"
        act = [k for k in (CHOICE, NOUL, SCORE) if self.cal[k].active]
        if not act:
            gate = self.cal[CHOICE].min_n
            return (f"后端 {self.backend.name}，反馈 {n}/{gate} 条"
                    f"（不到门槛不做校准）")
        ts = " ".join(f"{k}:T={self.cal[k].temperature:.2f}"
                      + (f",b={self.cal[k].b:+.2f}"
                         if abs(self.cal[k].b) >= 0.05 else "")
                      for k in act)
        e = self.ece()
        tail = f"，ECE {e:.3f}" if e == e else ""
        return f"后端 {self.backend.name}，反馈 {n} 条，{ts}{tail}"

    def report(self) -> str:
        lines = [f"判官：后端 {self.backend.name}，已提问 {self.asked} 次，"
                 f"有反馈 {self._prior_n + len(self.log)} 条"
                 + (f"（其中本次 {len(self.log)} 条）" if self._prior_n else "")]
        for kind in (CHOICE, NOUL, SCORE):
            t = self.cal[kind]
            n = sum(1 for r in self.log if r["kind"] == kind)
            lines.append(f"  {kind:6s} {t.text()}  反馈 {n}")
        e = self.ece()
        if e == e:      # not NaN
            lines.append(f"  校准误差 ECE = {e:.3f}"
                         f"（0 完美，>0.1 说明置信度和实际频率对不上）")
        return "\n".join(lines)

    # ── 落地 ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"backend": self.backend.name,
                "cal": {k: v.to_dict() for k, v in self.cal.items()},
                "thresholds": dict(self.thresholds)}

    def save(self, path: Optional[str] = None) -> str:
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("没有给路径")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return str(p)

    def load(self, path: str) -> None:
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— 校准文件坏了不能拖垮对局
            return
        for k, v in (d.get("cal") or {}).items():
            if k in self.cal:
                self.cal[k] = Temperature.from_dict(v)
        self.thresholds.update(d.get("thresholds") or {})
