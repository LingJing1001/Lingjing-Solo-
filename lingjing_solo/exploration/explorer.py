"""Layer 2 · 探索与假设引擎 (Exploration & Hypothesis)

对应 ARC-AGI-3 四大能力里的 Exploration + Modeling + Goal-setting。
- 信息增益驱动探索：优先选"预期最大化 Δ 信息量"的动作
- 假设生成与验证：小步探测 → 归纳规则 → 置信度升降
- 目标推断：从"什么状态触发 WIN"反推目标形态

注意：此层不调用 LLM，全部轻量计算，契合 RHAE 步数经济。
"""
from ..core import Logger, SoloConfig


class ExplorationEngine:
    def __init__(self, cfg: SoloConfig, field, logger: Logger = None):
        self.cfg = cfg
        self.field = field
        self.log = logger or Logger()
        self._probe_budget = 0     # 当前探测剩余步数
        self._probing = False
        self.last_score_details = {}
        self._failure_counts = {}

    # ---------- 信息增益估算 ----------
    def info_gain(self, action: str) -> float:
        """估算某动作的预期信息增益。

        启发式：未知 (s,a) 对越多 → 增益越高；已充分探索的动作增益衰减。
        真实实现可用转移表的计数作为不确定性估计。
        """
        current = self.field.grid_state
        if current is not None:
            from ..core import hash_grid
            shash = hash_grid(current)
            used = len(self.field.transition_index.get((shash, action), []))
            # 未尝试动作优先；重复动作的收益随次数衰减。
            return 1.0 / (1.0 + used)
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

    def on_environment_feedback(self, state: str, levels: int, previous_levels: int) -> None:
        """关卡变化后清理上一关的失败计数。"""
        if levels > previous_levels:
            self._failure_counts.clear()

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
        """从外部 win 信号反推目标（占位接口，由 harness 注入反馈）。"""
        return "unknown"
