"""Layer 2 · 探索与假设引擎 (Exploration & Hypothesis)

对应 ARC-AGI-3 四大能力里的 Exploration + Modeling + Goal-setting。
- 信息增益驱动探索：优先选"预期最大化 Δ 信息量"的动作
- 假设生成与验证：小步探测 → 归纳规则 → 置信度升降
- 目标推断：从"什么状态触发 WIN"反推目标形态

注意：此层不调用 LLM，全部轻量计算，契合 RHAE 步数经济。
"""
from collections import Counter
from math import log2
from typing import Optional

import numpy as np

from ..core import Logger, SoloConfig, hash_grid
from ..exploration.graph_explorer import (
    GraphExplorer,
    GraphExplorerConfig,
    action_int_to_name,
    action_name_to_int,
)


# 默认动作优先级：上下优先级高，左右优先级低
_DEFAULT_ACTION_PRIORITIES = {
    1: 0,  # UP
    2: 0,  # DOWN
    3: 1,  # LEFT
    4: 1,  # RIGHT
    5: 2,  # ROTATE
    6: 2,  # CLICK
}


class ExplorationEngine:
    def __init__(self, cfg: SoloConfig, field, logger: Logger = None):
        self.cfg = cfg
        self.field = field
        self.log = logger or Logger()
        self._probe_budget = 0     # 当前探测剩余步数
        self._probing = False
        self.last_score_details = {}
        self._failure_counts = {}

        # 图探索器：状态空间探索
        self._graph_explorer = GraphExplorer(
            GraphExplorerConfig(
                max_nodes=int(getattr(cfg, "graph_nodes_cap", 800)),
                default_action_priorities=_DEFAULT_ACTION_PRIORITIES,
            )
        ) if getattr(cfg, "graph_explorer_enabled", True) else None
        self._graph_initialized = False

    # ---------- 信息增益估算 ----------
    def info_gain(self, action: str) -> float:
        """估算某动作的预期信息增益。

        启发式：未知 (s,a) 对越多 → 增益越高；已充分探索的动作增益衰减。
        真实实现可用转移表的计数作为不确定性估计。
        """
        current = self.field.grid_state
        if current is not None:
            shash = self.field.current_hash()
            used = len(self.field.transition_index.get((shash, action), []))
            # 未尝试动作优先；重复动作的收益随次数衰减。
            novelty = 1.0 / (1.0 + used)
            transitions = self.field.transition_index.get((shash, action), [])
            if not transitions:
                return novelty
            successor_counts = Counter(t.state_after for t in transitions)
            total = len(transitions)
            entropy = -sum(
                (count / total) * log2(count / total)
                for count in successor_counts.values()
            )
            # 后继越不确定，动作越值得再次探测；权重限制在小范围，
            # 避免不确定性压过未探索动作的基础优先级。
            return novelty + min(0.25, 0.25 * entropy)

        # 若无历史，所有动作等权（随机试探）
        if len(self.field.transition_table) == 0:
            return 1.0
        # 已有转移表的动作视为"已探索"，增益递减
        # 这里用规则置信度均值作为"模型成熟度"代理
        rules = self.field.rules
        maturity = max(0.0, 1.0 - (sum(r.confidence for r in rules) / max(1, len(rules))))
        return maturity

    # ---------- 动作评分（核心）----------
    def score_actions(self, valid_actions) -> list[tuple[str, float]]:
        """对每个候选动作打分：信息增益 × 目标贴近度 × 反循环惩罚。

        这是对 RHAE 平方惩罚的直接对冲 —— 优先选"单位步数信息增益最大"的动作。
        """
        if not valid_actions:
            self.last_score_details = {}
            return []
        scored = []
        self.last_score_details = {}
        recent_actions = [t.action for t in list(self.field.transition_table)[-8:]]
        # 图探索器咨询：当前状态已有实测边时，给导航/新边建议加分
        graph_advice = self.graph_advice()
        graph_bonus = float(getattr(self.cfg, "graph_advice_bonus", 0.3))
        if graph_advice is None:
            graph_bonus = 0.0
        for order, a in enumerate(valid_actions):
            gain = self.info_gain(a)
            # 反循环：若动作近期重复，降低其探索优先级。
            recent_count = recent_actions.count(a)
            loop_penalty = min(0.75, 0.15 * recent_count)
            goal_bonus = 0.0
            current_hash = self.field.current_hash()
            predicted = self.field.predict(current_hash, a) if current_hash else None
            if predicted is not None:
                goal_bonus = max(
                    (
                        float(goal.confidence)
                        for goal in self.field.goals
                        if goal.state_hash and goal.state_hash == predicted
                    ),
                    default=0.0,
                ) * self.cfg.goal_score_bonus
            score = gain - loop_penalty + goal_bonus
            reason = ["information_gain"]
            if recent_count:
                reason.append("loop_penalty")
            if goal_bonus:
                reason.append("goal_successor")
            if graph_bonus and a == graph_advice:
                score += graph_bonus
                reason.append("graph_advice")
            self.last_score_details[a] = {
                "score": score,
                "information_gain": gain,
                "loop_penalty": loop_penalty,
                "goal_bonus": goal_bonus,
                "reason": "+".join(reason),
                "input_order": order,
            }
            scored.append((a, score))
        # Python's stable sort preserves caller order for equal scores.
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ---------- 规则归纳（从转移证据）----------
    def induce_rules(self):
        """扫描转移表，把高频 (s,a)->s' 模式提炼为新规则假设。

        这是"探索中建模型"的核心：Agent 不预知规则，靠观察归纳。
        """
        # 聚合：同一 (shash, action) 的后继分布
        idx = self.field.transition_index
        for (shash, action), transitions in idx.items():
            if len(transitions) < 2:
                continue  # 证据不足
            successors = {}
            for t in transitions:
                successors[t.state_after] = successors.get(t.state_after, 0) + 1
            # 若某一后继占主导（>70%），提炼为确定性规则
            total = sum(successors.values())
            for scc, cnt in successors.items():
                if cnt / total > 0.7:
                    premise = f"at state {shash} do {action}"
                    conclusion = f"-> {scc}"
                    # 避免重复
                    if not any(r.premise == premise for r in self.field.rules):
                        self.field.propose_rule(premise, conclusion)
                    break

    # ---------- 反馈归因 ----------
    def record_action_outcome(self, action: str, delta_pixels: int, progressed: bool) -> None:
        """记录无进展动作，供 R5 符号扩增器避开重复失败。"""
        if delta_pixels <= 0 and not progressed:
            self._failure_counts[action] = self._failure_counts.get(action, 0) + 1
        else:
            self._failure_counts.pop(action, None)

    def rha_penalty(self, action: str) -> float:
        """Repeated Hard Action 惩罚：对反复无进展的动作降权。

        惩罚 = base * fⁿ / (1 + fⁿ)，f=失败次数，n=power。
        失败 0 → 0；失败多 → 趋近 base（默认 0.8）。
        供 StrategicAdvisor.suggest_hypotheses 对候选动作排序时调用。
        """
        failures = self._failure_counts.get(action, 0)
        if failures == 0:
            return 0.0
        base = getattr(self.cfg, "rha_penalty_base", 0.8)
        power = getattr(self.cfg, "rha_penalty_power", 2.0)
        fp = failures ** power
        return float(base) * fp / (1.0 + fp)

    def on_environment_feedback(self, state: str, levels: int, previous_levels: int) -> None:
        """关卡变化后清理上一关的失败计数，并重置状态图（新关卡=新空间）。"""
        if levels > previous_levels:
            self._failure_counts.clear()
            if getattr(self.cfg, "graph_explorer_enabled", True):
                self._graph_explorer = GraphExplorer(
                    GraphExplorerConfig(
                        max_nodes=int(getattr(self.cfg, "graph_nodes_cap", 800)),
                        default_action_priorities=_DEFAULT_ACTION_PRIORITIES,
                    )
                )
                self._graph_initialized = False

    # ---------- 探测模式控制 ----------
    def start_probe(self):
        self._probing = True
        self._probe_budget = self.cfg.probe_max_steps

    def step_probe(self) -> bool:
        if not self._probing:
            return False
        self._probe_budget -= 1
        if self._probe_budget < 0:
            self._probing = False
            return False
        if self._probe_budget == 0:
            self._probing = False
        return True

    # ---------- 目标假设推断 ----------
    def infer_goal(self, win_callback) -> str:
        """把权威 WIN/level 反馈转为可审计目标假设。

        ``win_callback`` 可返回 ``{description, state_hash, confidence, kind}``；
        也可为 ``None``，此时使用 Field 已记录的 WIN 哈希。
        """
        feedback = win_callback(self.field) if callable(win_callback) else None
        if isinstance(feedback, dict):
            state_hash = str(feedback.get("state_hash") or self.field.current_hash())
            description = str(feedback.get("description") or (f"win:{state_hash}" if state_hash else "unknown"))
            confidence = max(0.0, min(1.0, float(feedback.get("confidence", 1.0))))
            kind = str(feedback.get("kind") or "win")
            if state_hash:
                self.field.update_goal(description, confidence, state_hash=state_hash, kind=kind)
            return description
        if self.field.win_hashes:
            state_hash = sorted(self.field.win_hashes)[-1]
            description = f"win:{state_hash}"
            self.field.update_goal(description, 1.0, state_hash=state_hash, kind="win")
            return description
        goals = sorted(self.field.goals, key=lambda goal: goal.confidence, reverse=True)
        return goals[0].description if goals else "unknown"

    # ---------- 图探索器接口（实盘字符串动作 ↔ 图内 int，fail-closed）----------

    @staticmethod
    def _gkey(state) -> str:
        """np 网格 → 确定性哈希串；其余原样（CEAX/field 已给哈希）。"""
        if isinstance(state, np.ndarray):
            return hash_grid(state)
        return str(state) if state is not None else ""

    def _graph_on(self) -> bool:
        return (
            bool(getattr(self.cfg, "graph_explorer_enabled", True))
            and self._graph_explorer is not None
        )

    @staticmethod
    def _to_int_actions(actions) -> list:
        return [i for i in (action_name_to_int(a) for a in (actions or [])) if i is not None]

    def _ensure_graph_initialized(self, state, actions):
        """确保图探索器已初始化（只建起点一次）。"""
        if self._graph_initialized or not self._graph_on():
            return
        ints = self._to_int_actions(actions)
        key = self._gkey(state)
        if not key or not ints:
            return
        try:
            self._graph_explorer.initialize(key, ints)
            self._graph_initialized = True
        except Exception:
            self._graph_explorer = None

    def record_graph_transition(self, from_state, action, success, to_state, to_actions):
        """记录状态转移到图探索器（字符串动作自动转 int；异常静默降级）。"""
        if not self._graph_on():
            return
        if not self._graph_initialized:
            self._ensure_graph_initialized(from_state, to_actions)
            if not self._graph_initialized:
                return
        ai = action_name_to_int(action)
        if ai is None:
            return
        try:
            self._graph_explorer.record_transition(
                self._gkey(from_state), ai, bool(success),
                self._gkey(to_state), self._to_int_actions(to_actions),
            )
        except Exception:
            self._graph_explorer = None

    def get_graph_distance(self, state):
        """获取到最近前沿节点的距离。"""
        if not self._graph_initialized or not self._graph_on():
            return float("inf")
        try:
            return self._graph_explorer.get_distance(self._gkey(state))
        except Exception:
            return float("inf")

    def choose_graph_action(self, state):
        """从图探索器选择动作，返回 "ACTIONn" 字符串或 None。"""
        if not self._graph_initialized or not self._graph_on():
            return None
        try:
            pick = self._graph_explorer.choose_action(self._gkey(state))
            if pick is None:
                return None
            return action_int_to_name(pick[0])
        except Exception:
            return None

    def graph_advice(self) -> Optional[str]:
        """当前状态已有实测边时的图导航建议（避免首访节点的退化建议）。"""
        if not self._graph_initialized or not self._graph_on():
            return None
        current = self.field.current_hash() if self.field is not None else ""
        state = getattr(self.field, "grid_state", None) if self.field is not None else None
        key = self._gkey(state) if state is not None else (current or "")
        if not key:
            return None
        try:
            gsh = self._graph_explorer.hash_state(key)
            node = self._graph_explorer.nodes.get(gsh)
            if node is None or not node.tested_actions:
                return None
            return self.choose_graph_action(key)
        except Exception:
            return None
