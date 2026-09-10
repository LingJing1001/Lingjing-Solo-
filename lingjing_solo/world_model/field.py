"""Layer 1 · 世界模型场 (World Model Field = Φ 场)

原灵境引擎的"统一信息场"在单 Agent 场景下的化身。
不再是多 Agent 共享介质，而是本 Agent 对当前环境的因果状态张量：
    网格状态 + 已归纳规则假设 + 目标假设 + 转移记忆 + 循环检测集。

设计要点：
- transition_table：原"版本化因果链"的单 Agent 版本，(s,a,s') 追加记录；
  当同一 (s,a) 出现不同 s' 时触发"规则修正"。
- rules / goals：带置信度的假设集 —— 探索中建模型，不做确定性结论。
- visited_states：直接服务 RHAE 效率，避免重复访问。
"""
from collections import deque, defaultdict
from ..core import (
    SoloConfig, Frame, RuleHypothesis, Transition, GoalHypothesis,
    FieldSnapshot, Logger, hash_grid, clamp,
)


class WorldModelField:
    def __init__(self, cfg: SoloConfig, logger: Logger = None, win_detector=None):
        self.cfg = cfg
        self.log = logger or Logger()
        self.win_detector = win_detector
        self.reset()

    # ---------- 生命周期 ----------
    def reset(self):
        self.grid_state = None                       # 当前紧凑网格表示
        self.transition_table = deque(maxlen=self.cfg.field_max_transitions)
        self.transition_index = defaultdict(list)    # (shash, action) -> [Transition]
        self.rules = []                              # List[RuleHypothesis]
        self.goals = []                              # List[GoalHypothesis]
        self.visited = deque(maxlen=self.cfg.state_hash_history)
        self.visited_set = set()
        self.step = 0
        self.last_delta_pixels = 0
        self.roi = []                               # 当前高优先级关注区域
        self._rule_id = 0
        self.conflict_flag = False                 # 最近是否出现规则冲突

    # ---------- 主更新入口 ----------
    def update(self, grid: Frame, prev_grid=None, action=None, objects=None):
        """每帧调用：更新网格、哈希、ROI、转移记录、规则证据。"""
        self.step = grid.t if grid.t is not None else self.step + 1
        prev_hash = hash_grid(self.grid_state) if self.grid_state is not None else None
        self.grid_state = grid.grid.copy()

        if prev_grid is not None and action is not None:
            cur_hash = hash_grid(self.grid_state)
            t = Transition(
                state_before=prev_hash,
                action=action,
                state_after=cur_hash,
                delta_pixels=int((prev_grid.grid != grid.grid).sum()),
                t=self.step,
            )
            self.transition_table.append(t)
            self.transition_index[(prev_hash, action)].append(t)
            self._reconcile_rules(t)                 # 用新转移核对规则

        # 更新 ROI：变化像素 ±pad 邻域
        self._update_roi(prev_grid.grid if prev_grid is not None else None, grid.grid)

        # 循环检测集
        h = hash_grid(self.grid_state)
        self.visited.append(h)
        self.visited_set.add(h)

        return self

    def _update_roi(self, prev, curr):
        pad = 2
        if prev is None:
            self.roi = []
            return
        import numpy as np
        diff = np.argwhere(prev != curr)
        if len(diff) == 0:
            return
        ys, xs = diff[:, 0], diff[:, 1]
        y0, y1 = clamp(int(ys.min()) - pad, 0, self.cfg.grid_size - 1), clamp(int(ys.max()) + pad, 0, self.cfg.grid_size - 1)
        x0, x1 = clamp(int(xs.min()) - pad, 0, self.cfg.grid_size - 1), clamp(int(xs.max()) + pad, 0, self.cfg.grid_size - 1)
        self.roi = [(y0, x0, y1, x1)]

    # ---------- 规则假设管理 ----------
    def propose_rule(self, premise: str, conclusion: str, confidence=None) -> RuleHypothesis:
        """探索引擎提出一条新规则假设。"""
        r = RuleHypothesis(
            premise=premise, conclusion=conclusion,
            confidence=confidence or self.cfg.rule_confidence_init,
        )
        self.rules.append(r)
        self._rule_id += 1
        self.log.log("Field", f"propose_rule: {premise} -> {conclusion} ({r.confidence:.2f})")
        return r

    def _reconcile_rules(self, t: Transition):
        """用新转移证据调整规则置信度；同一 (s,a) 出现矛盾结果 → 冲突降信。"""
        matches = self.transition_index.get((t.state_before, t.action), [])
        if len(matches) <= 1:
            # 唯一证据：温和提升所有"泛化型"规则
            for r in self.rules:
                if "any" in r.premise or r.premise == "default":
                    r.confidence = clamp(r.confidence + self.cfg.rule_confidence_inc * 0.3, 0, 1)
            return
        # 同一前驱+动作对应多个不同后继 → 规则冲突
        successors = {m.state_after for m in matches}
        if len(successors) > 1:
            self.conflict_flag = True
            self.log.log("Field", f"rule_conflict: (s,a) has {len(successors)} successors")
            for r in self.rules:
                r.confidence = clamp(r.confidence - self.cfg.rule_confidence_dec, 0, 1)
            self._gc_rules()

    def _gc_rules(self):
        """淘汰低置信度规则，控制容量。"""
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.confidence >= self.cfg.rule_confidence_min]
        if len(self.rules) > self.cfg.field_max_rules:
            self.rules = sorted(self.rules, key=lambda r: r.confidence, reverse=True)[:self.cfg.field_max_rules]
        if len(self.rules) != before:
            self.log.log("Field", f"rule_gc: {before} -> {len(self.rules)}")

    # ---------- 目标假设 ----------
    def update_goal(self, description: str, confidence: float):
        for g in self.goals:
            if g.description == description:
                g.confidence = clamp(g.confidence + confidence * 0.3, 0, 1)
                return
        self.goals.append(GoalHypothesis(description=description, confidence=confidence))

    # ---------- 查询接口 ----------
    def is_loop(self, window=None) -> bool:
        """连续 N 步状态重复 → 循环陷阱。"""
        w = window or self.cfg.loop_detect_window
        if len(self.visited) < w:
            return False
        recent = list(self.visited)[-w:]
        return len(set(recent)) < w * 0.6   # 去重后数量明显少于窗口

    def best_rules(self, top_k=5) -> list[RuleHypothesis]:
        return sorted(self.rules, key=lambda r: r.confidence, reverse=True)[:top_k]

    def snapshot(self, recent_n=10) -> FieldSnapshot:
        """Φ 场压缩摘要，供 LLM 上下文打包。"""
        return FieldSnapshot(
            grid_summary=f"step={self.step}, shape={self.grid_state.shape if self.grid_state is not None else None}",
            rules=self.best_rules(top_k=8),
            goals=self.goals,
            recent_transitions=list(self.transition_table)[-recent_n:],
            visited_count=len(self.visited_set),
            step=self.step,
        )

    def clear_conflict_flag(self):
        """清除已被反思消费的规则冲突信号。"""
        self.conflict_flag = False

    def detect_win(self, grid) -> bool:
        """Use an injected detector; fail closed when none is configured."""
        if self.win_detector is None:
            return False
        try:
            return bool(self.win_detector(grid))
        except (TypeError, ValueError):
            return False
