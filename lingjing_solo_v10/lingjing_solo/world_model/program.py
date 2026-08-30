"""
program.py — 可执行世界模型 v0.8（WMP v2）

整合：
    - 关系规则归纳（induction.RelationalInducer）
    - LLM 动力学代码生成（codegen.CodeGenerator）
    - 安全模拟（simulate）
    - 漂移检测 + CEGIS 重建触发

升级点（相对 v0.7）：
    ✅ 规则从「单对象响应」升级为「关系型（push/拾取/联动）」
    ✅ Layer A：LLM 生成 step()；Layer B：关系规则解释执行（离线可用）
    ✅ 归纳采用「假设-检验」：learn() 生成假设，Retrodict 检验并触发 rebuild
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable

from .symbols import SymbolTable
from .relations import (
    RelationGraph, RelationalRule, RelationalFact, Relation, build_relation_graph,
)
from .induction import TransitionEvidence, RelationalInducer
from .codegen import (
    DynamicsProgram, CodeGenerator, FakeLLM, GameState, UnsafeCodeError,
)


@dataclass
class WMPEvidence:
    """喂给 WMP 的转移证据（高层，包装前后两帧符号表）。"""
    action: str
    before: SymbolTable
    after: SymbolTable


class WorldModelProgram:
    """
    可执行世界模型（v0.8）。

    状态机：
        learn(evidence)        → 更新关系规则假设集
        compile(llm=None)      → 生成/重编译 step()（Layer A / Layer B）
        simulate(state, action) → 在模拟器中预测下一状态
        drift(state, action, real_next) → 预测 vs 现实，返回漂移 0~1
        needs_rebuild()         → 漂移超阈值 → CEGIS 重建
    """

    def __init__(
        self,
        llm_client=None,
        min_support: int = 2,
        confidence_threshold: float = 0.6,
        drift_threshold: float = 0.15,
    ):
        self.inducer = RelationalInducer(
            min_support=min_support, confidence_threshold=confidence_threshold
        )
        self.codegen = CodeGenerator(llm_client=llm_client, rules_inducer=self.inducer)
        self.drift_threshold = drift_threshold

        self._program: Optional[DynamicsProgram] = None
        self._drift_accum: List[float] = []
        self._compiled_from_rules: int = 0  # 用于检测「规则是否更新到代码」
        self.provenance: str = "none"

    # ---------- 学习 ----------

    def learn(self, ev: WMPEvidence):
        """从一条真实转移中学习关系规则。"""
        graph = build_relation_graph(ev.before)
        tev = TransitionEvidence(
            action=ev.action, before=ev.before, after=ev.after, graph_before=graph,
        )
        self.inducer.learn(tev)

    def find_conflict(self, ev: WMPEvidence) -> Optional[str]:
        graph = build_relation_graph(ev.before)
        tev = TransitionEvidence(
            action=ev.action, before=ev.before, after=ev.after, graph_before=graph,
        )
        return self.inducer.find_conflict(tev)

    # ---------- 编译 ----------

    def compile(self, llm=None) -> bool:
        """
        把当前已晋升的关系规则编译成可执行 step。
        返回 True 表示成功（可用于 simulate）。
        """
        rules = self.inducer.confident_rules()
        # 若规则集未变化且已有程序，跳过
        if self._program is not None and len(rules) == self._compiled_from_rules:
            return True

        generator = self.codegen if llm is None else CodeGenerator(
            llm_client=llm, rules_inducer=self.inducer
        )
        self._program = generator.generate(rules, traces=[])
        self._compiled_from_rules = len(rules)
        self.provenance = self._program.provenance
        return self._program is not None

    @property
    def is_compiled(self) -> bool:
        return self._program is not None

    # ---------- 模拟 ----------

    def simulate(self, state: Dict, action: str) -> Optional[Dict]:
        """在可执行世界模型中预测 next_state。失败返回 None。"""
        if self._program is None:
            return None
        return self._program.simulate(state, action)

    # ---------- 漂移 & CEGIS ----------

    def drift(self, state: Dict, action: str, real_next: Dict) -> float:
        """
        预测 vs 现实的对象位移差异 → 漂移度 [0, 1]。
        用于 CEGIS：漂移超阈值 → 触发重新归纳/重新生成代码。
        """
        predicted = self.simulate(state, action)
        if predicted is None:
            return 1.0  # 无法预测 = 完全漂移
        return _state_drift(predicted, real_next)

    def record_drift(self, d: float):
        self._drift_accum.append(d)

    def needs_rebuild(self) -> bool:
        """最近窗口内平均漂移超阈值 → 需要 CEGIS 重建。"""
        if not self._drift_accum:
            return False
        window = self._drift_accum[-8:]
        return (sum(window) / len(window)) > self.drift_threshold

    def reset_drift(self):
        self._drift_accum.clear()

    # ---------- 调试/序列化 ----------

    def rule_summary(self) -> str:
        rules = self.inducer.rules()
        if not rules:
            return "(no rules)"
        return "; ".join(
            f"{r.rule_id}(s={self.inducer._supports.get(r.rule_id,0)})"
            for r in rules
        )


# ---------- 工具函数 ----------

def _state_drift(a: Dict, b: Dict) -> float:
    """两状态的差异度：错位对象数 / 总对象数。"""
    objs_a = a.get("objects", {})
    objs_b = b.get("objects", {})
    all_ids = set(objs_a) | set(objs_b)
    if not all_ids:
        return 0.0
    mismatches = 0
    for oid in all_ids:
        oa = objs_a.get(oid)
        ob = objs_b.get(oid)
        if oa is None or ob is None:
            mismatches += 1
            continue
        if (oa.get("x"), oa.get("y")) != (ob.get("x"), ob.get("y")):
            mismatches += 1
    return mismatches / len(all_ids)
