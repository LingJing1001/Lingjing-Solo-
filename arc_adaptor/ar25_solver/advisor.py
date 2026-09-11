"""LLM 反思顾问 — 在步数阈值刹车，触发自我审视。

核心机制:
    1. 步数阈值刹车 — 搜索到达 budget 的 25%/50%/75% 时暂停
    2. 三问反思链:
       "我做错了什么"     → 分析过往决策的失误
       "我接下来怎么做"   → 规划下一步行动
       "如果不这么做会怎样" → 反事实推演后果
    3. 双模式:
       a) 规则模式 (默认) — 基于启发式趋势、覆盖率、死路数等指标
       b) LLM 模式 — 调用外部 LLM 回调进行深度反思

反思 → 行动映射:
    CONTINUE    — 当前方向正确，继续搜索
    ADJUST      — 调整启发权重 / 搜索参数
    SWITCH      — 切换搜索策略 (A* ↔ macro)
    RESTART     — 丢弃当前搜索，用新参数重新开始
    PRUNE       — 剪掉低效分支，聚焦高价值区域
    GIVE_UP     — 认输，避免浪费时间
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════

@dataclass
class SearchSnapshot:
    """搜索状态快照 — 反思的输入。"""
    level: int
    strategy: str
    steps_used: int
    budget: int
    coverage: int
    total_targets: int
    heuristic: float
    nodes: int
    dead_ends: int
    best_h_history: list = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class Reflection:
    """一次完整的反思结果。"""
    trigger_step: int
    threshold: float
    what_wrong: str
    what_next: str
    counterfactual: str
    advice: str
    action: str = "CONTINUE"
    params: dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"\n{'━'*50}\n"
            f"  反思触发 @ {self.trigger_step}步 ({self.threshold:.0%} budget)\n"
            f"{'━'*50}\n"
            f"  ❓ 我做错了什么？\n"
            f"    {self.what_wrong}\n"
            f"  ❓ 我接下来怎么做？\n"
            f"    {self.what_next}\n"
            f"  ❓ 如果不这么做会怎样？\n"
            f"    {self.counterfactual}\n"
            f"{'━'*50}\n"
            f"  → 建议: {self.action} — {self.advice}\n"
            f"{'━'*50}"
        )


# ═══════════════════════════════════════════════════════
# 反思顾问
# ═══════════════════════════════════════════════════════

class Advisor:
    """LLM 反思顾问。

    在搜索过程中按步数阈值触发反思，生成建议。

    Args:
        budget: 总步数预算
        level: 关卡编号
        thresholds: 触发阈值列表 (默认 [0.25, 0.5, 0.75])
        llm_callback: 外部 LLM 回调函数 (snapshot, questions) → reflection_text
        verbose: 打印反思过程
    """

    def __init__(self, budget, level, thresholds=None,
                 llm_callback=None, verbose=True):
        self.budget = budget
        self.level = level
        self.thresholds = thresholds or [0.25, 0.5, 0.75]
        self.llm_callback = llm_callback
        self.verbose = verbose

        self.triggered = set()
        self.reflections = []
        self._h_baseline = None

    def check_and_reflect(self, snap: SearchSnapshot) -> Optional[Reflection]:
        """检查是否到达阈值并触发反思。

        Returns:
            Reflection 或 None (未触发)
        """
        budget_ref = snap.budget if snap.budget > 0 else self.budget
        ratio = snap.steps_used / budget_ref if budget_ref > 0 else 0

        for i, t in enumerate(self.thresholds):
            if i in self.triggered:
                continue
            if ratio >= t:
                self.triggered.add(i)
                r = self._reflect(snap, t)
                self.reflections.append(r)
                if self.verbose:
                    print(r)
                return r
        return None

    def _reflect(self, snap: SearchSnapshot, threshold: float) -> Reflection:
        """执行一次完整的三问反思。"""
        if self.llm_callback:
            return self._llm_reflect(snap, threshold)
        return self._rule_reflect(snap, threshold)

    # ─── 规则模式 ───────────────────────────────────────

    def _rule_reflect(self, snap: SearchSnapshot, threshold: float) -> Reflection:
        """基于规则的反思 — 分析搜索指标生成建议。"""
        coverage_ratio = (snap.coverage / snap.total_targets
                           if snap.total_targets > 0 else 0)
        h = snap.heuristic
        h_trend = self._analyze_trend(snap.best_h_history)

        # ── 问题1: 我做错了什么？ ──
        wrong_parts = []

        if h_trend == "stagnant" and snap.nodes > 1000:
            wrong_parts.append(
                f"启发值停滞在 {h:.0f}，搜索了 {snap.nodes} 节点仍无突破，"
                f"可能陷入了局部最优"
            )
        elif h_trend == "rising":
            wrong_parts.append(
                f"启发值从历史低点回升到 {h:.0f}，之前的路径可能偏离了最优方向"
            )

        if coverage_ratio < 0.3 and threshold > 0.4:
            wrong_parts.append(
                f"已用 {snap.steps_used}步但覆盖率仅 {coverage_ratio:.0%}，"
                f"拼块覆盖策略可能效率不足"
            )

        dead_end_ratio = (snap.dead_ends / max(snap.nodes, 1))
        if dead_end_ratio > 0.6:
            wrong_parts.append(
                f"死路率 {dead_end_ratio:.0%} 过高，"
                f"动作枚举可能包含大量无效分支"
            )

        if not wrong_parts:
            wrong_parts.append(
                f"当前搜索状态正常: {snap.nodes}节点, h={h:.0f}, "
                f"覆盖率 {coverage_ratio:.0%}"
            )

        what_wrong = "; ".join(wrong_parts)

        # ── 问题2: 我接下来怎么做？ ──
        action, what_next, params = self._decide_action(
            snap, threshold, coverage_ratio, h_trend
        )

        # ── 问题3: 如果不这么做会怎样？ ──
        counterfactual = self._counterfactual(
            snap, threshold, coverage_ratio, action
        )

        return Reflection(
            trigger_step=snap.steps_used,
            threshold=threshold,
            what_wrong=what_wrong,
            what_next=what_next,
            counterfactual=counterfactual,
            advice=params.get("advice", ""),
            action=action,
            params=params,
        )

    def _decide_action(self, snap, threshold, coverage_ratio, h_trend):
        """根据当前状态决定行动建议。"""
        remaining = self.budget - snap.steps_used
        params = {}

        # 已找到解 → 建议停止或继续优化
        if coverage_ratio >= 1.0:
            if snap.strategy == "macro":
                action = "GIVE_UP"
                what_next = (
                    f"已找到全覆盖解 (cost={snap.heuristic})，"
                    f"可停止搜索或继续寻找更优解"
                )
                params["advice"] = "停止搜索 (已有最优解)"
            else:
                action = "CONTINUE"
                what_next = (
                    f"覆盖率已达 100%，搜索方向正确，"
                    f"继续直到找到精确解"
                )
                params["advice"] = "保持搜索"
            return action, what_next, params

        # 覆盖率极低 + 已过半 → 切换策略
        if coverage_ratio < 0.3 and threshold >= 0.5:
            alt = "宏动作搜索" if snap.strategy == "astar" else "A*搜索"
            action = "SWITCH"
            params["target_strategy"] = "macro" if snap.strategy == "astar" else "astar"
            what_next = (
                f"覆盖率仅 {coverage_ratio:.0%} 且已过 {threshold:.0%} 阈值，"
                f"切换到{alt}，用不同的搜索空间规避当前困境"
            )
            params["advice"] = f"切换到 {alt}"

        # 启发值停滞 + 节点数多 → 调整权重
        elif h_trend == "stagnant" and snap.nodes > 2000:
            action = "ADJUST"
            new_weight = 12 if snap.strategy == "astar" else 8
            params["weight"] = new_weight
            what_next = (
                f"启发值停滞且 {snap.nodes} 节点未突破，"
                f"提高权重到 {new_weight} 使搜索更贪心，"
                f"加速向目标区域收敛"
            )
            params["advice"] = f"提高启发权重到 {new_weight}"

        # 死路率高 → 剪枝
        elif snap.dead_ends / max(snap.nodes, 1) > 0.5:
            action = "PRUNE"
            what_next = (
                f"死路率过高，应对动作序列做更激进的剪枝: "
                f"跳过启发值 > {snap.heuristic * 1.5:.0f} 的分支，"
                f"聚焦高价值路径"
            )
            params["prune_ratio"] = 0.5
            params["advice"] = "加大剪枝力度"

        # 预算即将耗尽 → 认输或尝试最后冲刺
        elif remaining < self.budget * 0.15:
            if coverage_ratio < 0.5:
                action = "GIVE_UP"
                what_next = (
                    f"仅剩 {remaining} 步且覆盖率 {coverage_ratio:.0%}，"
                    f"已无法在预算内通关，应尽早放弃以节省时间"
                )
                params["advice"] = "放弃当前搜索"
            else:
                action = "CONTINUE"
                what_next = (
                    f"仅剩 {remaining} 步但覆盖率已达 {coverage_ratio:.0%}，"
                    f"保持当前方向做最后冲刺"
                )
                params["advice"] = "保持搜索"

        # 状态正常 → 继续
        else:
            action = "CONTINUE"
            what_next = (
                f"搜索进展正常 ({snap.nodes}节点, h={snap.heuristic:.0f})，"
                f"剩余预算 {remaining} 步，继续当前策略"
            )
            params["advice"] = "继续当前搜索"

        return action, what_next, params

    def _counterfactual(self, snap, threshold, coverage_ratio, action):
        """反事实推演: 如果不这么做会怎样？"""
        remaining = self.budget - snap.steps_used

        if action == "SWITCH":
            return (
                f"如果不切换策略，按当前覆盖率 {coverage_ratio:.0%} 和步速，"
                f"预计在 {remaining} 步内仅能达到 "
                f"{min(coverage_ratio + 0.1, 1.0):.0%} 覆盖，无法通关"
            )
        elif action == "ADJUST":
            return (
                f"如果不调整权重，按当前 {snap.nodes} 节点/停滞的启发值，"
                f"搜索树会继续在低价值区域膨胀，最终耗尽预算"
            )
        elif action == "PRUNE":
            return (
                f"如果不剪枝，死路率会持续累积，"
                f"有效搜索占比进一步下降，浪费剩余 {remaining} 步"
            )
        elif action == "GIVE_UP":
            return (
                f"如果不放弃，会在剩余 {remaining} 步内继续无效搜索，"
                f"浪费时间且无法通关"
            )
        else:
            return (
                f"如果改变方向，可能丢失当前已积累的搜索进展 "
                f"({snap.nodes} 节点, 覆盖 {coverage_ratio:.0%})，"
                f"得不偿失"
            )

    def _analyze_trend(self, history):
        """分析启发值趋势: improving / stagnant / rising / insufficient。"""
        if len(history) < 3:
            return "insufficient"
        recent = history[-5:]
        if all(h == recent[0] for h in recent):
            return "stagnant"
        if recent[-1] > recent[0]:
            return "rising"
        if recent[-1] < recent[0]:
            return "improving"
        return "stagnant"

    # ─── LLM 模式 ───────────────────────────────────────

    def _llm_reflect(self, snap: SearchSnapshot, threshold: float) -> Reflection:
        """使用外部 LLM 进行深度反思。

        将搜索状态格式化为 prompt，调用 llm_callback 获取反思文本，
        再解析为结构化的 Reflection。
        """
        context = self._format_context(snap, threshold)
        questions = (
            "1. 我做错了什么？分析当前搜索状态的问题。\n"
            "2. 我接下来怎么做？给出具体的行动建议。\n"
            "3. 如果不这么做会怎样？推演不采取行动的后果。\n"
            "请按以下格式回复:\n"
            "WHAT_WRONG: <分析>\n"
            "WHAT_NEXT: <建议>\n"
            "COUNTERFACTUAL: <推演>\n"
            "ACTION: CONTINUE|ADJUST|SWITCH|RESTART|PRUNE|GIVE_UP\n"
            "PARAMS: <json格式的参数>"
        )

        prompt = context + "\n\n" + questions
        response = self.llm_callback(prompt)

        parsed = self._parse_llm_response(response)
        return Reflection(
            trigger_step=snap.steps_used,
            threshold=threshold,
            what_wrong=parsed.get("what_wrong", "LLM 未返回分析"),
            what_next=parsed.get("what_next", "LLM 未返回建议"),
            counterfactual=parsed.get("counterfactual", "LLM 未返回推演"),
            advice=parsed.get("advice", ""),
            action=parsed.get("action", "CONTINUE"),
            params=parsed.get("params", {}),
        )

    def _format_context(self, snap, threshold):
        """格式化搜索状态为 LLM 可读的上下文。"""
        coverage_ratio = (snap.coverage / snap.total_targets
                          if snap.total_targets > 0 else 0)
        h_trend = self._analyze_trend(snap.best_h_history)

        return (
            f"[AR25 求解器反思上下文]\n"
            f"关卡: L{snap.level}\n"
            f"策略: {snap.strategy}\n"
            f"步数: {snap.steps_used}/{snap.budget} ({threshold:.0%} 阈值)\n"
            f"覆盖率: {snap.coverage}/{snap.total_targets} ({coverage_ratio:.0%})\n"
            f"启发值: {snap.heuristic:.0f} (趋势: {h_trend})\n"
            f"搜索节点: {snap.nodes}\n"
            f"死路数: {snap.dead_ends}\n"
            f"耗时: {snap.elapsed:.1f}s\n"
            f"启发值历史(最近5): {snap.best_h_history[-5:]}"
        )

    def _parse_llm_response(self, text):
        """解析 LLM 的结构化回复。"""
        import json as _json
        parsed = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("WHAT_WRONG:"):
                parsed["what_wrong"] = line[len("WHAT_WRONG:"):].strip()
            elif line.startswith("WHAT_NEXT:"):
                parsed["what_next"] = line[len("WHAT_NEXT:"):].strip()
            elif line.startswith("COUNTERFACTUAL:"):
                parsed["counterfactual"] = line[len("COUNTERFACTUAL:"):].strip()
            elif line.startswith("ACTION:"):
                parsed["action"] = line[len("ACTION:"):].strip()
            elif line.startswith("PARAMS:"):
                try:
                    parsed["params"] = _json.loads(line[len("PARAMS:"):].strip())
                except Exception:
                    parsed["params"] = {}
                parsed["advice"] = parsed["params"].get("advice", "")
        return parsed

    # ─── 查询 ───────────────────────────────────────────

    @property
    def last_reflection(self):
        """最后一次反思结果。"""
        return self.reflections[-1] if self.reflections else None

    def summary(self):
        """反思历史摘要。"""
        if not self.reflections:
            return "无反思触发"
        lines = [f"共触发 {len(self.reflections)} 次反思:"]
        for r in self.reflections:
            lines.append(f"  @{r.trigger_step}步({r.threshold:.0%}): {r.action}")
        return "\n".join(lines)
