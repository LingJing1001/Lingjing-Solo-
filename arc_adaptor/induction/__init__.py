"""自动推理层：规则由数据归纳，而不是由人预先写死。

对照一下两套东西的存在方式，差别是本质的：

    旧路径（goalsim / nav / graph）
        人读关卡截图 → 发现"目标是个静止的稀有区域" → 写进代码
        → 换一个游戏，这段代码要么失效，要么开始帮倒忙

    本层
        人只给三样东西：格子的概念、平移的概念、副作用的句式
        → agent 按一次键，看谁动了；动不了，看哪儿变了；过关了，倒推目标
        → 换一个游戏，按键语义作废重学，目标句式作废重推，
          但**推理过程本身**不变，所以不需要改代码

用法：

    from arc_adaptor.induction import Reasoner
    r = Reasoner()
    r.observe(g0, "ACTION1", g1)          # 每一步都喂进去
    a = r.decide(g1, ["ACTION1", ...])    # 它自己决定下一步，并给出理由
    print(r.text())
"""
from .engine import Reasoner, Trial, decompose
from .goals import Goal, induce
from .hypo import EffectBook, KeySemantics, Motion, fit
from .percept import Obj, Perception, guess_block, objects, quantise
from . import world

__all__ = [
    "Reasoner", "Trial", "decompose",
    "Goal", "induce",
    "EffectBook", "KeySemantics", "Motion", "fit",
    "Obj", "Perception", "guess_block", "objects", "quantise",
    "world",
]
