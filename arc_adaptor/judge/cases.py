"""把手头的**证据**组装成问题。这里只翻译计数，不下结论。

每一条 case 都是"我见到了什么"，不是"所以是什么"：

    key_semantics  这个键对这个体，位移语义定下来了吗
    displacement   如果定了，是往哪移
    probe_target   这几个还在打鼓的键，下一步该验哪个
    worth_recheck  某个键在某个位置推不动过，值不值得换个位置再试

共同点：`support/oppose` 全是**票数**。3 票一致和 9 票一致在答案上是
0.8 和 0.917 —— 差的不只是数字，是"还能不能再被推翻"的程度。
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from .api import Choice, Noul


def key_semantics(act: str, votes: Counter, noop: int = 0,
                  unclear: int = 0, who: str = "") -> Noul:
    """「这个键对这个体有一个确定的位移语义」成立吗？

    支持 = 最主流位移的票数；反对 = 所有非主流位移的票数。

    注意 `noop` 与 `unclear` 不在这里当反对证据：被墙挡住一次、画面闪一下，
    都不构成"这个键没有语义"。它们属于另外两个问题（`is_effective`）。
    """
    if not votes:
        return Noul(name=f"semantics:{act}:{who}",
                    statement=f"{act} 对 {who} 有确定的位移语义",
                    support=0.0, oppose=float(noop + unclear),
                    note=f"还没见它动过（被挡 {noop} 次）")
    top = votes.most_common(1)[0][1]
    total = sum(votes.values())
    return Noul(name=f"semantics:{act}:{who}",
                statement=f"{act} 对 {who} 有确定的位移语义",
                support=float(top), oppose=float(total - top),
                note=f"票型 {dict(votes)}")


def displacement(act: str, votes: Counter, who: str = "") -> Choice:
    """定了语义之后，具体是哪个位移？选项就是票里出现过的那些。"""
    opts = tuple(sorted(votes, key=lambda d: (-votes[d], d)))
    return Choice(name=f"shift:{act}:{who}", options=opts,
                  support={d: float(votes[d]) for d in opts},
                  note=f"{act} 在 {who} 上的位移票型")


def probe_target(cands: Iterable[tuple[str, int, int, bool]]) -> Choice:
    """几个键还打鼓，下一步验哪个？

    每个候选给 `(act, 冲突票数, 总票数, 当前局面是否已验过)`。
    支持 = 冲突票数（还没定就值得问），反对 = 已验过（同一局面重复问没新信息）。
    """
    items = list(cands)
    opts = tuple(a for a, *_ in items)
    support = {a: float(c) for a, c, _n, _p in items}
    oppose = {a: (2.0 if p else 0.0) for a, _c, _n, p in items}
    note = " ".join(f"{a}:冲突{c}/共{n}{'/已验' if p else ''}"
                    for a, c, n, p in items)
    return Choice(name="probe:which", options=opts,
                  support=support, oppose=oppose, note=note)


def worth_recheck(act: str, noops: int, moved_elsewhere: bool,
                  who: str = "") -> Noul:
    """「换个位置再按一次能拿到新信息」成立吗？

    支持：这个键确实在别处做出过位移，给 2 票 —— 会动的键换个位置往往就能动，
    这是"位置相关"而不是"死键"。
    反对：推不动几次就给几票。

    验算（门槛 0.5，即"证据打平就试一次"）：
        动过 + 推不动 0/1/2 次 → 0.75 / 0.60 / 0.50  → 值得试
        动过 + 推不动 3 次      → 0.43               → 放弃
    和原来那条 `noop_count >= 3 才放弃` 完全等价，但它是从证据算出来的常数。
    """
    support = 2.0 if moved_elsewhere else 0.0
    oppose = float(noops)
    return Noul(name=f"recheck:{act}:{who}",
                statement=f"{act} 在 {who} 换位置值得再试",
                support=support, oppose=oppose,
                note=f"推不动 {noops} 次，别处动过={moved_elsewhere}")
