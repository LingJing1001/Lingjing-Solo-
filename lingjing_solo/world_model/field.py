"""Layer 1-2 · 世界模型场 Φ（钱学森灵境 · 统一信息场 Solo 化身）

协议映射：
  P1 时空基底      → grid + 状态哈希
  P2 信息密度 Φ    → PhiDensity 粗粒场
  P5 源项写回      → inject_source_term / SourceTerm 历史
  P6 版本化共识    → self.version 单调递增（环境帧=唯一事实源）
  P7 面积律        → BubbleWall 清晰区 vs 迷雾
  P9 迷雾展开      → 新 ROI 并入清晰区统计
"""
from __future__ import annotations

from collections import deque, defaultdict, Counter
import numpy as np
from ..core import (
    SoloConfig, Frame, RuleHypothesis, Transition, GoalHypothesis,
    FieldSnapshot, SourceTerm, BubbleWall, PhiDensity, Logger,
    hash_grid, clamp, canonicalize,
    compute_phi, compute_bubble, curvature_proxy,
)


class WorldModelField:
    def __init__(self, cfg: SoloConfig, logger: Logger = None):
        self.cfg = cfg
        self.log = logger or Logger()
        self._ar25 = None
        self.reset()

    @property
    def ar25(self):
        """Lazy AR25 reflection field (Layer 1 game-specific)."""
        if self._ar25 is None:
            from .ar25_field import Ar25Field
            self._ar25 = Ar25Field()
        return self._ar25

    def reset(self):
        self.grid_state = None
        self.levels = 0
        self.env_state = "NOT_PLAYED"
        self.version = 0                              # 协议 6：事实源版本
        self.transition_table = deque(maxlen=self.cfg.field_max_transitions)
        self.transition_index = defaultdict(list)
        self.predict_graph = {}
        self.state_value = defaultdict(float)
        self.action_counts = Counter()
        self.pair_counts = Counter()
        self._recent_actions = deque(maxlen=16)
        self.source_terms = deque(maxlen=self.cfg.source_term_history)
        self.effect_ema = defaultdict(float)          # action → 实测因果强度 EMA
        self.noop_streak = 0                          # 连续空操作
        self.rules = []
        self.goals = []
        self.win_hashes = set()
        self.progress_hashes = set()
        self.visited = deque(maxlen=self.cfg.state_hash_history)
        self.visited_set = set()
        self.step = 0
        self.last_delta_pixels = 0
        self.roi = []
        self.bubble: BubbleWall | None = None
        self.phi: PhiDensity | None = None
        self.curvature = 0.0
        self.conflict_flag = False
        self._rule_id = 0
        self._last_levels = 0
        self.pending_source: SourceTerm | None = None
        self._ar25 = None

    # ---------- 协议 5：注入源项（决策写回意图）----------
    def inject_source_term(self, action: str, rationale: str = "",
                           delta_phi: float = None) -> SourceTerm:
        """Agent 决策瞬间登记 ΔJ；真实演化由环境 step 完成后再 update()。"""
        action = canonicalize(action)
        if delta_phi is None:
            # 默认扰动：未探索边更大，体现「信息注入」
            h = self.current_hash()
            visits = self.pair_counts.get((h, action), 0) if h else 0
            delta_phi = 1.0 / (1.0 + visits)
            if self.bubble is not None:
                delta_phi += 0.3 * self.bubble.fog_ratio
        st = SourceTerm(
            agent_id=self.cfg.agent_id,
            action=action,
            delta_phi=float(delta_phi),
            tick=self.step,
            version_read=self.version,
            rationale=rationale or "act",
        )
        self.pending_source = st
        self.source_terms.append(st)
        self.log.log("Field", f"inject ΔJ action={action} δΦ={st.delta_phi:.3f} v={self.version}")
        return st

    # ---------- 主更新：环境真相回写 Φ ----------
    def update(self, grid: Frame, prev_grid=None, action=None, objects=None):
        self.step = grid.t if grid.t is not None else self.step + 1
        self.env_state = getattr(grid, "state", None) or self.env_state
        self.levels = int(getattr(grid, "levels_completed", self.levels) or 0)

        levels_for_hash = self.levels if self.cfg.include_levels_in_hash else None
        prev_hash = None
        if self.grid_state is not None:
            prev_levels = self._last_levels if self.cfg.include_levels_in_hash else None
            prev_hash = hash_grid(self.grid_state, prev_levels)

        prev_np = prev_grid.grid if prev_grid is not None and hasattr(prev_grid, "grid") else None
        self.grid_state = grid.grid.copy() if grid.grid is not None else None
        cur_hash = hash_grid(self.grid_state, levels_for_hash) if self.grid_state is not None else ""

        # Φ + 泡壁（协议 2/7）
        if self.grid_state is not None:
            self.phi = compute_phi(self.grid_state, block=self.cfg.phi_block_size)
            self.curvature = curvature_proxy(self.phi)
            self.bubble = compute_bubble(
                prev_np, self.grid_state,
                pad=self.cfg.bubble_pad,
                grid_size=self.cfg.grid_size,
            )
            self.roi = list(self.bubble.clear_rects)

        progressed = self.levels > self._last_levels
        if prev_grid is not None and action is not None and prev_hash:
            action = canonicalize(action)
            delta = int((prev_np != grid.grid).sum()) if (
                prev_np is not None and grid.grid is not None
            ) else 0
            self.last_delta_pixels = delta
            # 协议 6：每完成一次因果转移，版本 +1
            self.version += 1
            t = Transition(
                state_before=prev_hash,
                action=action,
                state_after=cur_hash,
                delta_pixels=delta,
                t=self.step,
                progressed=progressed,
                version=self.version,
            )
            self.transition_table.append(t)
            self.transition_index[(prev_hash, action)].append(t)
            self.action_counts[action] += 1
            self.pair_counts[(prev_hash, action)] += 1
            self._recent_actions.append(action)
            self._rebuild_edge(prev_hash, action)
            self._reconcile_rules(t)
            self._calibrate_effect(action, delta, progressed, prev_hash != cur_hash)

            # 用实际 delta 校准上一笔源项
            if self.pending_source and self.pending_source.action == action:
                realized = min(2.0, delta / 64.0)
                self.pending_source.delta_phi = 0.5 * self.pending_source.delta_phi + 0.5 * realized
                self.pending_source = None

            if progressed:
                self.state_value[cur_hash] += 5.0
                self.state_value[prev_hash] += 1.0
                self.progress_hashes.add(cur_hash)
                self.update_goal(
                    f"progress@L{self.levels}:{cur_hash}",
                    confidence=0.7, state_hash=cur_hash, kind="progress",
                )
            elif prev_hash != cur_hash:
                self.state_value[cur_hash] += 0.05

        if str(self.env_state).upper() == "WIN":
            self.win_hashes.add(cur_hash)
            self.state_value[cur_hash] += 20.0
            self.update_goal(f"win:{cur_hash}", 1.0, state_hash=cur_hash, kind="win")

        if cur_hash:
            self.visited.append(cur_hash)
            self.visited_set.add(cur_hash)

        self._last_levels = self.levels
        return self

    def _calibrate_effect(self, action: str, delta: int, progressed: bool, changed: bool):
        """ΔJ 实测校准：把环境反馈写回动作因果强度 + 规则置信。"""
        action = canonicalize(action)
        # 归一化效果：像素变化 + 关卡进展加权
        raw = min(1.0, delta / 128.0) + (0.5 if progressed else 0.0) + (0.15 if changed else 0.0)
        alpha = self.cfg.effect_ema_alpha
        old = self.effect_ema[action]
        self.effect_ema[action] = (1 - alpha) * old + alpha * raw

        if delta <= 0 and not progressed:
            self.noop_streak += 1
        else:
            self.noop_streak = 0

        # 高效果转移 → 温和抬升匹配规则
        if raw >= 0.25:
            premise_key = f"do {action}"
            for r in self.rules:
                if action in r.premise or premise_key in r.premise:
                    r.confidence = clamp(
                        r.confidence + self.cfg.rule_confidence_inc * 0.5 * raw, 0, 1
                    )
                    r.evidence += 1
        elif raw < 0.05 and not changed:
            # 空操作证据 → 略降「该动作万能」类规则
            for r in self.rules:
                if action in r.premise and r.confidence > self.cfg.rule_confidence_min:
                    r.confidence = clamp(
                        r.confidence - self.cfg.rule_confidence_dec * 0.2, 0, 1
                    )

    def effect_strength(self, action: str) -> float:
        return float(self.effect_ema.get(canonicalize(action), 0.0))

    def _rebuild_edge(self, shash: str, action: str):
        matches = self.transition_index.get((shash, action), [])
        if not matches:
            self.predict_graph.pop((shash, action), None)
            return
        counts = Counter(m.state_after for m in matches)
        nxt, cnt = counts.most_common(1)[0]
        if cnt / len(matches) >= self.cfg.predict_majority_ratio:
            self.predict_graph[(shash, action)] = nxt
        else:
            self.predict_graph.pop((shash, action), None)
            self.conflict_flag = True

    def predict(self, state_hash: str, action: str):
        return self.predict_graph.get((state_hash, canonicalize(action)))

    def successors(self, state_hash: str, actions=None):
        actions = actions or self.cfg.allowed_actions
        out = []
        for a in actions:
            a = canonicalize(a)
            nxt = self.predict(state_hash, a)
            if nxt is not None:
                out.append((a, nxt))
        return out

    def current_hash(self) -> str:
        if self.grid_state is None:
            return ""
        levels = self.levels if self.cfg.include_levels_in_hash else None
        return hash_grid(self.grid_state, levels)

    def propose_rule(self, premise: str, conclusion: str, confidence=None) -> RuleHypothesis:
        r = RuleHypothesis(
            premise=premise, conclusion=conclusion,
            confidence=confidence or self.cfg.rule_confidence_init,
        )
        self.rules.append(r)
        self._rule_id += 1
        self.log.log("Field", f"propose_rule: {premise} -> {conclusion} ({r.confidence:.2f})")
        return r

    def _reconcile_rules(self, t: Transition):
        matches = self.transition_index.get((t.state_before, t.action), [])
        if len(matches) <= 1:
            for r in self.rules:
                if "any" in r.premise or r.premise == "default":
                    r.confidence = clamp(r.confidence + self.cfg.rule_confidence_inc * 0.3, 0, 1)
            return
        if len({m.state_after for m in matches}) > 1:
            self.conflict_flag = True
            self.log.log("Field", "rule_conflict")
            for r in self.rules:
                r.confidence = clamp(r.confidence - self.cfg.rule_confidence_dec, 0, 1)
            self._gc_rules()

    def _gc_rules(self):
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.confidence >= self.cfg.rule_confidence_min]
        if len(self.rules) > self.cfg.field_max_rules:
            self.rules = sorted(self.rules, key=lambda r: r.confidence, reverse=True)[:self.cfg.field_max_rules]
        if len(self.rules) != before:
            self.log.log("Field", f"rule_gc: {before} -> {len(self.rules)}")

    def update_goal(self, description: str, confidence: float,
                    state_hash: str = "", kind: str = "generic"):
        for g in self.goals:
            if g.description == description or (state_hash and g.state_hash == state_hash and g.kind == kind):
                g.confidence = clamp(max(g.confidence, confidence), 0, 1)
                if state_hash:
                    g.state_hash = state_hash
                return
        self.goals.append(GoalHypothesis(
            description=description, confidence=confidence,
            state_hash=state_hash, kind=kind,
        ))

    def goal_hashes(self) -> set:
        hs = set(self.win_hashes)
        for g in self.goals:
            if g.state_hash and g.confidence >= 0.5:
                hs.add(g.state_hash)
        for h, v in self.state_value.items():
            if v >= 3.0:
                hs.add(h)
        return hs

    def consume_conflict(self) -> bool:
        flag = self.conflict_flag
        self.conflict_flag = False
        return flag

    def is_loop(self, window=None) -> bool:
        w = window or self.cfg.loop_detect_window
        if len(self.visited) < w:
            return False
        recent = list(self.visited)[-w:]
        return len(set(recent)) < w * 0.6

    def best_rules(self, top_k=5):
        return sorted(self.rules, key=lambda r: r.confidence, reverse=True)[:top_k]

    def snapshot(self, recent_n=10, c_value: float = 0.0) -> FieldSnapshot:
        goals = sorted(self.goals, key=lambda g: g.confidence, reverse=True)
        best = goals[0].description if goals else ""
        fog = self.bubble.fog_ratio if self.bubble else 0.0
        return FieldSnapshot(
            grid_summary=(
                f"v={self.version}, step={self.step}, levels={self.levels}, "
                f"state={self.env_state}, fog={fog:.2f}, R̃={self.curvature:.2f}"
            ),
            rules=self.best_rules(top_k=8),
            goals=goals[:8],
            recent_transitions=list(self.transition_table)[-recent_n:],
            visited_count=len(self.visited_set),
            step=self.step,
            graph_edges=len(self.predict_graph),
            best_goal=best,
            version=self.version,
            c_value=c_value,
            fog_ratio=fog,
            phi_grad_energy=self.curvature,
        )

    def note_win(self, grid=None, levels: int = None):
        self.env_state = "WIN"
        if grid is None:
            grid = self.grid_state
        if grid is None:
            return
        lv = self.levels if levels is None else levels
        levels_for_hash = lv if self.cfg.include_levels_in_hash else None
        h = hash_grid(grid, levels_for_hash)
        self.win_hashes.add(h)
        self.state_value[h] += 20.0
        self.update_goal(f"win:{h}", 1.0, state_hash=h, kind="win")

    def detect_win(self, grid=None, env_state=None) -> bool:
        state = env_state if env_state is not None else self.env_state
        if str(state).upper() == "WIN":
            return True
        if grid is not None:
            levels = self.levels if self.cfg.include_levels_in_hash else None
            h = hash_grid(grid, levels)
            if h in self.win_hashes:
                return True
        return False
