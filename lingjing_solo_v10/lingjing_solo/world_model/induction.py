"""
induction.py — 关系规则归纳（观测 → 假设集，假设-检验前置）

v0.8 新增：从转移证据中归纳「关系型转移规则」。
策略：
    1. 对每个 (action, 关系图) → 下一帧对象位移，构造候选规则
    2. 用后续转移证据做「支持/冲突」计数
    3. 置信度 = support / (support + conflicts)，达到阈值才晋升

这是「假设-检验」的第一步（假设生成）；检验由 CEGIS/Retrodict 负责。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .relations import (
    RelationGraph, RelationalRule, RelationalFact, Relation,
    build_relation_graph, _direction_from,
)
from .symbols import SymbolTable


@dataclass
class TransitionEvidence:
    """一条转移证据：动作前后两帧的符号表 + 关系图。"""
    action: str
    before: SymbolTable
    after: SymbolTable
    graph_before: RelationGraph


class RelationalInducer:
    """
    关系规则归纳器。

    归纳范式（对应 ARC-AGI-3 的「探索中建模型」）：
        观察到 (s, a) → s' 后，若位移涉及「两个对象的相对关系变化」，
        则生成一条关系型规则假设，例如：
            push(box, dir) ⇔ avatar ADJACENT box AND box 前方为 FREE
    """

    def __init__(self, min_support: int = 2, confidence_threshold: float = 0.6):
        self.min_support = min_support
        self.confidence_threshold = confidence_threshold
        self._rules: Dict[str, RelationalRule] = {}
        self._supports: Dict[str, int] = defaultdict(int)
        self._conflicts: Dict[str, int] = defaultdict(int)

    # ---------- 公共 API ----------

    def learn(self, evidence: TransitionEvidence):
        """从一条转移证据中更新规则假设集。"""
        self._record_displacements(evidence)
        self._record_simple_relations(evidence)

    def rules(self) -> List[RelationalRule]:
        return list(self._rules.values())

    def confident_rules(self) -> List[RelationalRule]:
        """达到晋升阈值的规则（可用于生成可执行代码）。"""
        out = []
        for rid, rule in self._rules.items():
            s = self._supports.get(rid, 0)
            c = self._conflicts.get(rid, 0)
            total = s + c
            rule.support = s
            rule.confidence = (s / total) if total > 0 else 0.0
            if s >= self.min_support and rule.confidence >= self.confidence_threshold:
                out.append(rule)
        return out

    def find_conflict(self, evidence: TransitionEvidence) -> Optional[str]:
        """检测新证据是否与已有（已晋升）规则矛盾。返回冲突 rule_id 或 None。"""
        for rule in self.confident_rules():
            if not rule.matches(evidence.graph_before.facts, evidence.action):
                continue
            # 规则声称某对象应位移，但实际没位移 → 冲突
            # （简化：只检查规则 effect 涉及的主语是否位移）
            predicted = self._predict(rule, evidence)
            if predicted is None:
                continue
            for obj_id, new_pos in predicted.items():
                before_pos = self._pos_of(evidence.before, obj_id)
                after_pos = self._pos_of(evidence.after, obj_id)
                if after_pos is not None and after_pos != new_pos:
                    return rule.rule_id
        return None

    # ---------- 内部：归纳 ----------

    def _record_displacements(self, ev: TransitionEvidence):
        """对象位移 → 若是「avatar 推动相邻对象」，归纳 push 规则。"""
        avatar = ev.before.avatar()
        if avatar is None:
            return
        avatar_after = ev.after.avatar()
        if avatar_after is None:
            return

        # 找出所有发生位移的非 avatar 对象
        for obj_id, obj in ev.before.objects.items():
            after_obj = ev.after.objects.get(obj_id)
            if after_obj is None:
                continue
            if (obj.x, obj.y) == (after_obj.x, after_obj.y):
                continue  # 未位移
            # 该对象位移方向
            disp = _direction_from(obj, after_obj)
            if disp is None:
                continue
            # 若位移方向 == avatar 移动方向，且两者相邻 → 候选 push 规则
            avatar_disp = _direction_from(avatar, avatar_after)
            if disp != avatar_disp:
                continue
            if not ev.graph_before.has(
                RelationalFact(Relation.ADJACENT, avatar.id, obj.id)
            ):
                continue

            rid = f"push:{obj_id}:{disp}"
            rule = self._rules.get(rid) or RelationalRule(
                rule_id=rid,
                condition_relations=[
                    RelationalFact(Relation.ADJACENT, avatar.id, obj.id),
                ],
                condition_action=disp,
                effect=f"push({obj_id}, {disp}) when avatar adjacent",
            )
            self._rules[rid] = rule
            self._supports[rid] += 1

    def _record_simple_relations(self, ev: TransitionEvidence):
        """单对象响应（avatar 自身移动、gravity 等）保留为简单规则。"""
        avatar = ev.before.avatar()
        if avatar is None:
            return
        avatar_after = ev.after.avatar()
        if avatar_after is None:
            return
        disp = _direction_from(avatar, avatar_after)
        if disp is None:
            return
        rid = f"move:avatar:{disp}"
        rule = self._rules.get(rid) or RelationalRule(
            rule_id=rid,
            condition_relations=[],
            condition_action=disp,
            effect=f"avatar moves {disp}",
        )
        self._rules[rid] = rule
        self._supports[rid] += 1

    # ---------- 内部：预测（供冲突检测） ----------

    def _predict(self, rule: RelationalRule, ev: TransitionEvidence) -> Optional[Dict]:
        """按规则 effect 预测对象位移。仅处理 push 类。"""
        if not rule.rule_id.startswith("push:"):
            return None
        _, obj_id, direction = rule.rule_id.split(":")
        obj = ev.before.objects.get(obj_id)
        if obj is None:
            return None
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[direction]
        return {obj_id: (obj.x + dx, obj.y + dy)}

    # ---------- 调试/序列化 ----------

    def rule_summary(self) -> str:
        """所有规则（含未晋升）的人类可读摘要，便于测试与日志。"""
        rules = self.rules()
        if not rules:
            return "(no rules)"
        parts = []
        for r in rules:
            s = self._supports.get(r.rule_id, 0)
            c = self._conflicts.get(r.rule_id, 0)
            total = s + c
            conf = (s / total) if total > 0 else 0.0
            parts.append(f"{r.rule_id}(s={s},c={c},conf={conf:.2f})")
        return "; ".join(parts)

    @staticmethod
    def _pos_of(symtab: SymbolTable, obj_id) -> Optional[Tuple[int, int]]:
        obj = symtab.objects.get(obj_id)
        if obj is None:
            return None
        return (obj.x, obj.y)
