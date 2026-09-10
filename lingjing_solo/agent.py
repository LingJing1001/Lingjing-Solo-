"""Lingjing-Solo · 单 Agent 顶层编排 (The Agent)

把 Layer 0~4 串成决策闭环，并对接 Kaggle 烟囱口：
    Agent 只需实现 is_done() / choose_action() 两个方法。
    reset() 在每个新游戏开始时由 harness 调用。

决策优先级（每步）：
    1. WIN 检测 → 结束
    2. 反思触发？→ LLM 战略重估（受预算节制）
    3. 短程规划可解？→ 轻量 BFS/A*
    4. 否则 → 探索引擎信息增益评分（贪心）
"""
from .core import SoloConfig, Frame, Logger, Logger as _L
from .perception import PerceptionEncoder
from .world_model import WorldModelField
from .exploration import ExplorationEngine
from .planning import LightweightPlanner, LLMPlanner
from .reflection import ReflectionTrigger


class LingjingSoloAgent:
    """单 Agent 灵境引擎：统一信息场 Φ 的单 Agent 化身。"""

    def __init__(self, cfg: SoloConfig = None, logger: Logger = None, llm_fn=None):
        self.cfg = cfg or SoloConfig()
        self.log = logger or Logger()
        self.step = 0
        self._prev_grid = None
        self._terminal_state = None
        self._levels_completed = None

        # ---- 五层装配 ----
        self.encoder = PerceptionEncoder(self.cfg, self.log)
        self.field = WorldModelField(self.cfg, self.log)
        self.explorer = ExplorationEngine(self.cfg, self.field, self.log)
        self.planner = LightweightPlanner(self.cfg, self.field, self.log)
        self.llm = LLMPlanner(self.cfg, self.log)
        self.reflector = ReflectionTrigger(self.cfg, self.field, self.log)

        if llm_fn is not None:
            self.llm.inject_llm(llm_fn)

    # ---------- Kaggle 烟囱口：reset ----------
    def reset(self, env=None):
        """每个新游戏开始时调用：重置 Φ 场与预算，保留编码器等跨局能力。"""
        self.field.reset()
        self.explorer = ExplorationEngine(self.cfg, self.field, self.log)
        self.planner = LightweightPlanner(self.cfg, self.field, self.log)
        self.llm.calls_used = 0
        self.reflector.reset()
        self.step = 0
        self._prev_grid = None
        self._terminal_state = None
        self._levels_completed = None
        self.log.log("Agent", "reset for new game")

    def observe(self, grid, state=None, levels_completed=None):
        """接收 harness 提供的权威状态，避免仅靠网格猜测终止条件。"""
        if levels_completed is not None and self._levels_completed is not None:
            if levels_completed > self._levels_completed:
                self.field.reset()
                self.explorer = ExplorationEngine(self.cfg, self.field, self.log)
                self.planner = LightweightPlanner(self.cfg, self.field, self.log)
                self._prev_grid = None
                self._last_action = None
        self._terminal_state = state
        self._levels_completed = levels_completed
        return grid

    # ---------- Kaggle 烟囱口：is_done ----------
    def is_done(self, frames, latest_frame) -> bool:
        """判定是否结束当前局。

        official state is authoritative when supplied by the harness.
        """
        state_name = getattr(self._terminal_state, "name", self._terminal_state)
        if state_name in {"WIN", "GAME_OVER"}:
            return True
        grid = self._to_grid(latest_frame)
        if grid is None:
            return False
        if self.field.detect_win(grid):
            return True
        # 硬上限：人类预估的若干倍，超过则强制结束（防止单局失控）
        cap = self.cfg.human_baseline_estimate * 5
        return self.step >= cap

    # ---------- Kaggle 烟囱口：choose_action ----------
    def choose_action(self, frames, latest_frame, valid_actions=None):
        """核心决策：返回动作字符串，如 "UP" / "SPACE" / None。

        Kaggle 评测会传入 frames (历史) 与 latest_frame (当前帧)。
        valid_actions 若未由环境提供，则使用配置默认动作空间。
        """
        self.step += 1
        grid = self._to_grid(latest_frame)
        if grid is None:
            return self._default_action(valid_actions)

        # ---- 构造当前帧对象 ----
        prev_grid_obj = self._prev_grid
        curr_frame = Frame(grid=grid, t=self.step)

        # ---- Layer 0：感知编码 ----
        perception = self.encoder(prev_grid_obj.grid if prev_grid_obj is not None else None, grid)
        _ = perception  # feature / delta_pixels / objects —— 供后续 ROI 规划使用

        # ---- Layer 1：更新 Φ 场（需要上一步动作）→ 见下方 _last_action 机制 ----
        self.field.update(curr_frame, prev_grid_obj, getattr(self, "_last_action", None))

        # ---- 规则归纳 ----
        self.explorer.induce_rules()

        # ---- 候选动作 ----
        valid = valid_actions or self.cfg.allowed_actions

        # ---- Layer 4：反思触发？→ LLM 战略重估 ----
        if self.reflector.should_reflect_now() and self.llm.can_call():
            snap = self.reflector.pack_context(valid)
            action = self.llm.plan(snap, valid)
            if action and action in valid:
                return self._commit(action)

        # ---- Layer 3：短程规划 ----
        plan = self.planner.search(valid_actions=valid)
        if plan and plan in valid:
            return self._commit(plan)

        # ---- Layer 2：探索评分（兜底）----
        scored = self.explorer.score_actions(valid)
        chosen = self.planner.greedy_next(scored)
        if chosen and chosen in valid:
            return self._commit(chosen)

        # ---- 最终兜底：随机合法动作 ----
        return self._commit(valid[0])

    # ---------- 内部辅助 ----------
    def _commit(self, action: str) -> str:
        """记录动作供下一步 Φ 场更新使用。"""
        self._last_action = action
        # 缓存当前网格为"上一步"：包装成带 .grid 的对象，与 Frame 接口一致
        self._prev_grid = type("Grid", (), {"grid": self.field.grid_state})
        return action

    def _default_action(self, valid_actions):
        valid = valid_actions or self.cfg.allowed_actions
        return self._commit(valid[0])

    @staticmethod
    def _to_grid(frame) -> "np.ndarray | None":
        """兼容多种帧输入形式（numpy / dict / 对象）。"""
        if frame is None:
            return None
        import numpy as np
        if isinstance(frame, np.ndarray):
            return frame
        for attr in ("grid", "array", "observation"):
            if hasattr(frame, attr):
                return getattr(frame, attr)
        if isinstance(frame, dict):
            for k in ("grid", "array", "observation"):
                if k in frame:
                    return np.asarray(frame[k])
        return None

    # ---------- 便捷构造 ----------
    @classmethod
    def with_llm(cls, llm_fn, **cfg_kwargs) -> "LingjingSoloAgent":
        """一行构造带 LLM 顾问的 Agent。"""
        cfg = SoloConfig(**cfg_kwargs)
        return cls(cfg=cfg, llm_fn=llm_fn)
