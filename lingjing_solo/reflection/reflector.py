"""Layer 4 · 反思触发器 (Reflection Trigger)

监控三类"该请 LLM 出场"的信号（原 ConsciousnessMonitor 的单 Agent 化身）：
1. 循环陷阱：连续 N 步状态重复
2. 规则冲突：新转移与现有假设矛盾（由 Field 标记）
3. 步数预算告急：已用步数 / 人类预估 超过阈值

触发时打包 Φ 场压缩摘要 + 最近转移 + 候选动作，供 LLM 战略重估。
"""
from ..core import SoloConfig, ReflectionSignal, FieldSnapshot, Logger


class ReflectionTrigger:
    def __init__(self, cfg: SoloConfig, field, logger: Logger = None):
        self.cfg = cfg
        self.field = field
        self.log = logger or Logger()
        self._last_reflect_step = -999
        self._last_signal = ReflectionSignal()

    def reset(self):
        """开始新关卡时清空反思节流状态和上一次触发原因。"""
        self._last_reflect_step = -999
        self._last_signal = ReflectionSignal()

    def evaluate(self) -> ReflectionSignal:
        """综合三类信号，产出是否该反思的判定。"""
        sig = ReflectionSignal()
        # 1. 循环陷阱
        sig.loop_trapped = self.field.is_loop()
        # 2. 规则冲突：由世界模型在同一 (state, action) 出现不同后继时标记
        sig.rule_conflict = bool(getattr(self.field, "conflict_flag", False))
        # 3. 步数预算告急
        estimated = self.cfg.human_baseline_estimate
        used = self.field.step
        sig.budget_warning = (used / max(1, estimated)) >= self.cfg.budget_warn_ratio
        return sig

    def should_reflect_now(self) -> bool:
        sig = self.evaluate()
        if not sig.should_reflect:
            return False
        # 节流：两次反思之间至少间隔 N 步
        if (self.field.step - self._last_reflect_step) < self.cfg.reflection_min_interval:
            return False
        self._last_signal = sig
        self._last_reflect_step = self.field.step
        clear_conflict = getattr(self.field, "clear_conflict_flag", None)
        if clear_conflict is not None:
            clear_conflict()
        return True

    def pack_context(self, valid_actions, recent_n=10) -> FieldSnapshot:
        """打包 Φ 场摘要，作为 LLM 上下文。"""
        snap = self.field.snapshot(recent_n=recent_n)
        actions = snap.valid_actions if valid_actions is None else valid_actions
        snap.valid_actions = list(actions or [])
        if not any(
            (self._last_signal.loop_trapped, self._last_signal.rule_conflict,
             self._last_signal.budget_warning)
        ):
            self._last_signal = self.evaluate()
        reasons = []
        if self._last_signal.loop_trapped:
            reasons.append("loop_trapped")
        if self._last_signal.rule_conflict:
            reasons.append("rule_conflict")
        if self._last_signal.budget_warning:
            reasons.append("budget_warning")
        snap.reflection_reasons = reasons
        self.log.log(
            "Reflect",
            f"pack_context: step={snap.step}, rules={len(snap.rules)}, "
            f"visited={snap.visited_count}",
        )
        return snap
