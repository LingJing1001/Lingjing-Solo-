"""Layer 3b · advisor.py —— 双模式战略层

开放模式 (use_llm_advisor=True): LLM 战略顾问
封闭模式 (use_llm_advisor=False): 符号猜想扩增器（零 LLM，纯 CPU）

ARC-AGI-3 适配说明：
  本场是**交互游戏**（逐步动作→环境反馈），不是静态 ARC 输入输出网格对。
  符号扩增的对象是「动作效应 / 转移规则 / 动作序列宏」，而非旋转/翻转整张网格。
"""
from __future__ import annotations

import itertools
from collections import Counter

from ..core import SoloConfig, FieldSnapshot, RuleHypothesis, Logger, canonicalize, clamp

# 交互游戏基础「算子」模板（封闭模式算子库）
INTERACTION_OPERATORS = (
    "move_effect",       # 方向键/简单动作引起状态转移
    "click_effect",      # ACTION6 点击效应
    "progress_effect",   # 导致 levels_completed 上升
    "noop_effect",       # 无像素变化
    "combo_seq",         # 动作序列组合
    "perturb_click",     # 点击坐标扰动
)


class SymbolicHypothesisExpander:
    """封闭离线：符号组合 + 剪枝，替代 LLM 创造性猜想。"""

    def __init__(self, cfg: SoloConfig, logger: Logger = None):
        self.cfg = cfg
        self.log = logger or Logger()
        self.rounds_used = 0
        self.exhausted = False
        self._seen_keys: set[str] = set()
        self.last_batch: list[RuleHypothesis] = []

    def can_expand(self) -> bool:
        return not self.exhausted and self.rounds_used < self.cfg.symbolic_advisor_max_rounds

    def expand(
        self,
        snapshot: FieldSnapshot,
        failed_rules: list[RuleHypothesis],
        valid_actions: list[str],
        explorer=None,
    ) -> list[RuleHypothesis]:
        if not self.can_expand():
            self.exhausted = True
            return []

        self.rounds_used += 1
        failed_premises = {r.premise for r in failed_rules}
        blacklist = set(failed_premises)
        if explorer is not None:
            for a, cnt in explorer._failure_counts.items():
                if cnt >= 2:
                    blacklist.add(f"pattern: {canonicalize(a)}")

        candidates: list[tuple[float, RuleHypothesis]] = []

        # 1) 转移历史差分：高 delta 动作 → 新规则
        for t in snapshot.recent_transitions[-16:]:
            kind = "progress_effect" if t.progressed else (
                "noop_effect" if t.delta_pixels <= 0 else "move_effect"
            )
            premise = f"sym:{kind} @{t.state_before[:10]} do {t.action}"
            conclusion = f"-> {t.state_after[:10]} d={t.delta_pixels}"
            key = premise + "|" + conclusion
            if key in self._seen_keys or premise in blacklist:
                continue
            conf = clamp(0.35 + t.delta_pixels / 128.0 + (0.3 if t.progressed else 0), 0, 1)
            candidates.append((conf, RuleHypothesis(premise, conclusion, conf, 1)))

        # 2) 算子组合：两步转移链 A→B→C
        trans = snapshot.recent_transitions[-24:]
        by_from: dict[tuple[str, str], str] = {}
        for t in trans:
            by_from[(t.state_before, t.action)] = t.state_after
        for (s0, a0), s1 in list(by_from.items())[:32]:
            for (s1k, a1), s2 in list(by_from.items())[:32]:
                if s1k != s1:
                    continue
                premise = f"sym:combo_seq {a0};{a1} @ {s0[:8]}"
                conclusion = f"-> {s2[:10]}"
                key = premise + conclusion
                if key in self._seen_keys or premise in blacklist:
                    continue
                candidates.append((0.45, RuleHypothesis(premise, conclusion, 0.45, 1)))

        # 3) 参数扰动：对有效动作生成变体（未尝试方向 / 低 RHA 动作）
        valid = [canonicalize(a) for a in valid_actions]
        tried = {canonicalize(a) for t in trans for a in [t.action]}
        for a in valid:
            if a == "ACTION6":
                continue
            if a not in tried:
                premise = f"sym:perturb untried {a}"
                conclusion = "explore_frontier"
                key = premise
                if key not in self._seen_keys:
                    candidates.append((0.55, RuleHypothesis(premise, conclusion, 0.55, 0)))

        # 4) 动作序列宏：从 Top 效应动作组合短序列
        if explorer is not None and len(valid) >= 2:
            scored = []
            for a in valid:
                if a == "ACTION6":
                    continue
                eff = explorer.field.effect_strength(a) if hasattr(explorer, "field") else 0
                scored.append((eff - explorer.rha_penalty(a), a))
            scored.sort(reverse=True)
            tops = [a for _, a in scored[:4]]
            for seq in itertools.permutations(tops, min(2, len(tops))):
                premise = f"sym:macro {'>'.join(seq)}"
                if premise in blacklist or premise in self._seen_keys:
                    continue
                candidates.append((0.4, RuleHypothesis(premise, "macro_seq", 0.4, 0)))

        # 强剪枝 Top-N
        candidates.sort(key=lambda x: x[0], reverse=True)
        out: list[RuleHypothesis] = []
        for score, rule in candidates:
            key = rule.premise + "|" + rule.conclusion
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            out.append(rule)
            if len(out) >= self.cfg.symbolic_max_hypotheses:
                break

        self.last_batch = out
        if not out:
            self.exhausted = True
            self.log.log("Advisor", "symbolic pool exhausted")
        else:
            self.log.log("Advisor", f"symbolic expand round={self.rounds_used} n={len(out)}")
        return out

    def action_from_hypothesis(self, rule: RuleHypothesis, valid_actions: list[str]) -> str | None:
        """从符号假设中提取一步可执行动作（供 agent 应急）。"""
        valid = {canonicalize(a) for a in valid_actions}
        for tok in rule.premise.replace(";", " ").replace(">", " ").split():
            if tok.startswith("ACTION") and tok in valid:
                return tok
        return None


class StrategicAdvisor:
    """双模式顾问：LLM（开放）或符号扩增（封闭）。"""

    def __init__(self, cfg: SoloConfig, logger: Logger = None):
        self.cfg = cfg
        self.log = logger or Logger()
        self._llm_fn = None
        self._hypothesis_fn = None
        self.calls_used = 0
        self.last_suggestions: list[str] = []
        self.symbolic = SymbolicHypothesisExpander(cfg, logger)

    def inject_llm(self, llm_fn):
        self.cfg.use_llm_advisor = True
        self._llm_fn = llm_fn

    def inject_hypothesis_llm(self, fn):
        self._hypothesis_fn = fn

    def inject_prompt_llm(self, fn):
        """注入只接收 Prompt 文本的模型，并切换到开放顾问模式。"""
        self.cfg.use_llm_advisor = True
        self._llm_fn = lambda snapshot, valid_actions: fn(
            self.build_prompt(snapshot, valid_actions)
        )

    def build_prompt(self, snapshot: FieldSnapshot, valid_actions=None) -> str:
        """构造可审计的 R5 Prompt，不消耗 LLM 预算。"""
        from ..reflection.prompt import build_r5_prompt

        prompt = build_r5_prompt(snapshot, valid_actions)
        return (
            f"{prompt}\n"
            f"【顾问预算】剩余调用次数：{self.budget_left}\n"
            f"【当前触发原因】{', '.join(getattr(snapshot, 'reflection_reasons', [])) or '未记录'}\n"
        )

    @property
    def budget_left(self) -> int:
        return self.cfg.llm_calls_per_game - self.calls_used

    @property
    def symbolic_exhausted(self) -> bool:
        return self.symbolic.exhausted

    def can_call(self) -> bool:
        if self.cfg.use_llm_advisor:
            return self._llm_fn is not None and self.budget_left > 0
        return self.symbolic.can_expand()

    def plan(self, snapshot: FieldSnapshot, valid_actions) -> str | None:
        if self.cfg.use_llm_advisor:
            return self._plan_llm(snapshot, valid_actions)
        return self._plan_symbolic(snapshot, valid_actions)

    def _plan_llm(self, snapshot, valid_actions) -> str | None:
        if not self.can_call():
            return None
        self.calls_used += 1
        self.log.log("Advisor", f"LLM call #{self.calls_used}/{self.cfg.llm_calls_per_game}")
        try:
            result = self._llm_fn(snapshot, valid_actions)
            action = canonicalize(result) if result else ""
            for valid_action in valid_actions or []:
                if canonicalize(valid_action) == action:
                    return valid_action
            return None
        except Exception as e:
            self.log.log("Advisor", f"LLM error: {e}")
            return None

    def _plan_symbolic(self, snapshot, valid_actions) -> str | None:
        rules = self.symbolic.expand(snapshot, [], list(valid_actions))
        for rule in rules:
            act = self.symbolic.action_from_hypothesis(rule, valid_actions)
            if act:
                return act
        return None

    def suggest_hypotheses(
        self,
        snapshot: FieldSnapshot,
        failed_rules: list[RuleHypothesis],
        valid_actions: list[str] | None = None,
        explorer=None,
        field=None,
    ) -> list[str]:
        if self.cfg.use_llm_advisor:
            return self._suggest_llm(snapshot, failed_rules)
        return self._suggest_symbolic(snapshot, failed_rules, valid_actions or [], explorer, field)

    def _suggest_llm(self, snapshot, failed_rules) -> list[str]:
        if self._hypothesis_fn is not None and self.budget_left > 0:
            self.calls_used += 1
            try:
                out = self._hypothesis_fn(snapshot, failed_rules)
                self.last_suggestions = list(out or [])
                return self.last_suggestions
            except Exception as e:
                self.log.log("Advisor", f"hypothesis LLM error: {e}")
                return []
        if not self.can_call():
            return []
        return []

    def _suggest_symbolic(
        self, snapshot, failed_rules, valid_actions, explorer, field
    ) -> list[str]:
        rules = self.symbolic.expand(snapshot, failed_rules, valid_actions, explorer)
        if field is not None:
            for r in rules:
                if not any(x.premise == r.premise for x in field.rules):
                    field.propose_rule(r.premise, r.conclusion, confidence=r.confidence)
        self.last_suggestions = [r.premise for r in rules]
        return self.last_suggestions


LLMPlanner = StrategicAdvisor