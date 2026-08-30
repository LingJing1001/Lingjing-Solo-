"""Layer 3 · 规划执行层 (Planning & Execution)

短程规划：用转移表 + 目标假设做 BFS/A*，不开 LLM。
长程规划：当短程搜索失效 / 目标模糊 / 新关卡时，才触发 LLM（由 Agent 层节制）。
LLM 调用节制：每关硬预算，直接回应"纯 LLM 烧 token"的失败模式。
"""
from ..core import SoloConfig, Logger, hash_grid


class LightweightPlanner:
    """无 LLM 的短程规划：基于转移表的贪心 + 有限 BFS。

    当世界模型场已积累足够转移时，用它做局部搜索；
    否则退化为 ExplorationEngine 的信息增益评分。
    """
    def __init__(self, cfg: SoloConfig, field, logger: Logger = None):
        self.cfg = cfg
        self.field = field
        self.log = logger or Logger()

    def search(self, goal_hint=None, depth=None, valid_actions=None) -> str | None:
        """Return a legal action leading to a previously unseen successor.

        This is deliberately conservative: it only uses observed transitions and
        never treats an unknown action as a predicted route. The caller can then
        fall back to exploration when no safe learned transition is available.
        """
        if self.field.grid_state is None or not self.field.transition_table:
            return None
        allowed = set(valid_actions) if valid_actions is not None else None
        start = hash_grid(self.field.grid_state)
        candidates = []
        for (state, action), transitions in self.field.transition_index.items():
            if state != start or (allowed is not None and action not in allowed):
                continue
            for transition in transitions:
                if transition.state_after not in self.field.visited_set:
                    candidates.append((transition.t, action))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def greedy_next(self, scored_actions: list) -> str:
        """转移表不足时的贪心：直接取探索评分最高的动作。"""
        if not scored_actions:
            return None
        return scored_actions[0][0]


class LLMPlanner:
    """LLM 战略顾问：仅做"战略重估"，不做微观决策。

    通过 inject_llm() 注入调用函数，保持框架对具体模型解耦，
    便于本地/评测环境切换（评测期无网络，可用规则化 fallback）。
    """
    def __init__(self, cfg: SoloConfig, logger: Logger = None):
        self.cfg = cfg
        self.log = logger or Logger()
        self._llm_fn = None
        self._prompt_llm_fn = None
        self.calls_used = 0

    def inject_llm(self, llm_fn):
        """注入 LLM 调用函数：fn(field_snapshot, valid_actions) -> action_str"""
        self._llm_fn = llm_fn

    def inject_prompt_llm(self, llm_fn):
        """注入 prompt 调用函数：fn(prompt) -> action_str。"""
        self._prompt_llm_fn = llm_fn

    @property
    def budget_left(self) -> int:
        return self.cfg.llm_calls_per_game - self.calls_used

    def can_call(self) -> bool:
        has_callback = self._llm_fn is not None or self._prompt_llm_fn is not None
        return has_callback and self.budget_left > 0

    def build_prompt(self, snapshot, valid_actions=None) -> str:
        """构造 R5 prompt；仅生成文本，不调用模型。"""
        from ..reflection import build_r5_prompt

        return build_r5_prompt(snapshot, valid_actions)

    def plan(self, snapshot, valid_actions) -> str:
        """调用 LLM 做战略决策，自动计入预算。"""
        if not self.can_call():
            return None
        self.calls_used += 1
        self.log.log("LLM", f"call #{self.calls_used}/{self.cfg.llm_calls_per_game}")
        try:
            if self._prompt_llm_fn is not None:
                action = self._prompt_llm_fn(self.build_prompt(snapshot, valid_actions))
            else:
                action = self._llm_fn(snapshot, valid_actions)
        except Exception as e:
            self.log.log("LLM", f"error: {e}")
            return None
        if action is None:
            return None
        if valid_actions is not None and action not in valid_actions:
            self.log.log("LLM", f"invalid action rejected: {action!r}")
            return None
        return action
