"""
agent.py — Lingjing-Solo v1.0 Kaggle Agent

适配 ARC-AGI-3 官方接口：Agent 只需实现
    - is_done(self, frames, latest_frame) -> bool
    - choose_action(self, frames, latest_frame) -> GameAction

设计：把 v0.9 的五层内核（感知/世界模型/探索/规划/反思）
组装成一个符合官方 harness 的 Agent 类。

决策优先级（v1.0 八级，含模拟器规划）：
    ① WIN 检测           → 直接终止
    ② 反思 / LLM 战略    → 仅在触发信号时（预算节制）
    ③ 宏动作（技能库）    → 若正在执行一个已学技能，连续执行
    ④ 微动作（BFS 规划）  → Planner（转移表 > WMP > 对象 > 评分）
    ⑤ 主动探索            → 信息增益引导（零样本开局建图）
    ⑥ 受控探针            → 预算内试探
    ⑦ WMP 漂移重建        → CEGIS 循环
    ⑧ 兜底随机方向        → 避免卡死

性能（v1.0）：Planner 带增量缓存 + 深度自适应 + 浅拷贝，
    解决 v0.9「每步 5460 节点」的瓶颈。
观测（v1.0）：Telemetry 记录每步决策来源，跑分后 JSONL 喂回分析。
"""
from typing import List, Optional, Any, Dict
import time

from .world_model.symbols import SymbolTable, GameObject
from .world_model.program import WorldModelProgram, WMPEvidence
from .planning.planner import Planner, make_box_goal_evaluator, state_key
from .telemetry import Telemetry


# ---------- 官方 harness 的类型桩（真实环境会提供，此处仅占位）----------

class GameAction:
    """动作占位类型。真实 harness 提供具体实现。"""
    def __init__(self, kind: str = "key", value: Any = None, keys: Any = None):
        self.kind = kind  # "key" | "undo" | "mouse"
        self.value = value
        self.keys = keys

    def to_dict(self):
        return {"kind": self.kind, "value": self.value, "keys": self.keys}


# ---------- 主 Agent ----------

class LingjingAgent:
    """
    Lingjing-Solo v1.0 — 单 Agent 世界模型场 Agent。

    用法（Kaggle Notebook）：
        from lingjing_solo import LingjingAgent
        agent = LingjingAgent(goals=[(4, 1)], telemetry_path="run.jsonl")

        def is_done(frames, latest_frame):
            return agent.is_done(frames, latest_frame)

        def choose_action(frames, latest_frame):
            return agent.choose_action(frames, latest_frame)
    """

    def __init__(
        self,
        goals: Optional[List[tuple]] = None,
        llm_client: Any = None,
        llm_calls_per_game: int = 8,
        max_steps_per_game: int = 500,
        grid_w: int = 64,
        grid_h: int = 64,
        telemetry: Optional[Telemetry] = None,
        telemetry_path: Optional[str] = None,
    ):
        self.goals = list(goals) if goals else []
        self.llm_client = llm_client
        self.llm_budget = llm_calls_per_game
        self.max_steps = max_steps_per_game
        self.grid_w = grid_w
        self.grid_h = grid_h

        # v1.0 观测
        if telemetry is not None:
            self.telemetry = telemetry
        else:
            self.telemetry = Telemetry(log_path=telemetry_path, enabled=True)

        # ---- 五层内核 ----
        self.wmp = WorldModelProgram(
            llm_client=llm_client, min_support=1, confidence_threshold=0.0,
        )
        self.planner = Planner(
            wmp=self.wmp,
            goal_evaluator=(
                make_box_goal_evaluator(self.goals) if self.goals else None
            ),
            max_depth=6,
            adaptive_depth=True,
            telemetry=self.telemetry,
        )
        # 运行时状态
        self.step_count: int = 0
        self._prev_frame = None
        self._macro_remaining: List[str] = []  # 宏动作剩余序列
        self._game_id: Optional[str] = None

    # ---------- 官方接口 ----------

    def reset(self, game_id: Optional[str] = None, goals: Optional[List[tuple]] = None):
        """新一关开始时调用（官方 harness 不保证调用，故 choose_action 内也自检）。"""
        self.step_count = 0
        self._prev_frame = None
        self._macro_remaining = []
        self.planner.clear_cache()
        self.wmp.reset_drift()
        if goals is not None:
            self.goals = list(goals)
            self.planner.set_goal_evaluator(make_box_goal_evaluator(self.goals))
        self.telemetry.reset()
        self._game_id = game_id

    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """检测 WIN 或步数耗尽。"""
        if self._detect_win(latest_frame):
            self.telemetry.end_step(win=True)
            return True
        if self.step_count >= self.max_steps:
            self.telemetry.end_step(win=False)
            return True
        return False

    def choose_action(self, frames: List[Any], latest_frame: Any) -> GameAction:
        """
        八级决策链路（v1.0）。
        返回官方 GameAction。
        """
        self.telemetry.start_step()
        self.step_count += 1

        # ---- 把帧转成内部 state（占位实现，真实环境需对接帧格式）----
        state = self._frame_to_state(latest_frame)

        # ---- ① 宏动作连续执行 ----
        if self._macro_remaining:
            act = self._macro_remaining.pop(0)
            self.telemetry.record(source="macro")
            self.telemetry.end_step(action=act)
            return self._action(act)

        # ---- ② WIN 检测 ----
        if self._detect_win(latest_frame):
            self.telemetry.end_step(win=True)
            return self._action("noop")

        # ---- ③ WMP 漂移重建（CEGIS）----
        if self.wmp.needs_rebuild():
            self.telemetry.record(source="rebuild")
            self.wmp.reset_drift()
            self.planner.clear_cache()

        # ---- ④ 反思 / LLM 战略（预算节制）----
        if self._should_reflect(state) and self.llm_budget > 0:
            self.llm_budget -= 1
            plan = self._llm_plan(state, frames)
            if plan:
                self._macro_remaining = plan[1:]
                self.telemetry.record(source="reflect", llm_budget=self.llm_budget)
                self.telemetry.end_step(action=plan[0])
                return self._action(plan[0])

        # ---- ⑤ 微动作：Planner BFS（四级后继 + 增量缓存）----
        valid = self._valid_actions(state)
        action = self.planner.search(state, valid)
        if action is not None:
            # 记录决策来源（Planner 内部已记 sim_calls/cache）
            self.telemetry.end_step(action=action)
            self.planner.note_visit(state)
            return self._action(action)

        # ---- ⑥ 主动探索 / 探针 ----
        explored = self._active_explore(state, valid)
        if explored is not None:
            self.telemetry.record(source="explore")
            self.telemetry.end_step(action=explored)
            return self._action(explored)

        # ---- ⑦ 兜底：反循环评分（Planner ④ 内部）----
        # search() 已含兜底，能到这里说明连 apply_move 都失败 → 随机合法方向
        fallback = valid[0] if valid else "noop"
        self.telemetry.record(source="fallback")
        self.telemetry.end_step(action=fallback)
        return self._action(fallback)

    # ---------- 帧 ↔ 状态 适配（占位，需按官方帧格式补全）----------

    def _frame_to_state(self, frame: Any) -> Dict:
        """
        把官方帧（64×64×16 网格）转为 Planner 用的 state dict。
        v1.0 占位实现：假设 frame 已是 state-like dict；
        真实接入时在此做像素→对象→符号表的编码（对接 perception/encoder）。
        """
        if isinstance(frame, dict) and "objects" in frame:
            return frame
        # 未知帧格式：返回空状态（触发兜底探索）
        return {"objects": {}, "avatar_id": None, "extras": {}}

    def _valid_actions(self, state: Dict) -> List[str]:
        """从 available_actions 推断。占位：默认四方向。"""
        return ["up", "down", "left", "right"]

    def _action(self, name: str) -> GameAction:
        """动作名 → 官方 GameAction。"""
        if name == "noop":
            return GameAction(kind="key", value="noop")
        return GameAction(kind="key", value=name)

    # ---------- 内部策略 ----------

    def _detect_win(self, frame: Any) -> bool:
        """
        WIN 检测（双层）：
        - 外部权威回调（若有，通过 set_win_callback 注入官方 is_win）
        - 目标假设自举（box 均在 goal 上）
        v1.0 占位：按 goals 判定；真实环境应注入官方判定。
        """
        if hasattr(self, "_win_callback") and self._win_callback is not None:
            try:
                if self._win_callback(frame):
                    return True
            except Exception:
                pass
        state = self._frame_to_state(frame)
        objs = state.get("objects", {})
        boxes = [o for o in objs.values() if o.get("role") in ("box", "pushable")]
        if not boxes or not self.goals:
            return False
        goal_set = set(self.goals)
        return all((b["x"], b["y"]) in goal_set for b in boxes)

    def set_win_callback(self, fn):
        """注入官方 WIN 判定（Kaggle harness 提供时调用）。"""
        self._win_callback = fn

    def _should_reflect(self, state: Dict) -> bool:
        """反思触发信号：循环陷阱 / 假设冲突 / 步数告急。"""
        key = state_key(state.get("objects", {}))
        # 简化：连续访问同一状态 → 触发
        if not hasattr(self, "_recent_states"):
            self._recent_states = []
        self._recent_states.append(key)
        if len(self._recent_states) > 6:
            self._recent_states.pop(0)
        loop = len(self._recent_states) >= 4 and len(set(self._recent_states[-4:])) <= 1
        budget_warn = self.step_count >= int(self.max_steps * 0.3)
        return loop or budget_warn

    def _llm_plan(self, state: Dict, frames: List) -> Optional[List[str]]:
        """LLM 战略顾问：返回动作序列（首动作立即执行，其余入宏队列）。
        无 LLM 时返回 None（降级为轻量规划，符合 Kaggle 无网络约束）。"""
        if self.llm_client is None:
            return None
        # 占位：真实实现把 state + 最近转移打包给 LLM
        return None

    def _active_explore(self, state: Dict, valid: List[str]) -> Optional[str]:
        """主动探索：挑「未访问 / 信息增益最高」的方向。"""
        best, best_val = None, -1.0
        for act in valid:
            nxt = self.planner._apply_move(state, act)  # noqa
            if nxt is None:
                continue
            key = state_key(nxt["objects"])
            val = 0.0
            if key not in self.planner._visited:
                val += 1.0
            if val > best_val:
                best_val, best = val, act
        return best

    # ---------- 学习（真实转移喂给 WMP）----------

    def learn_transition(self, action: str, before_state: Dict, after_state: Dict):
        """
        外部调用：把「真实转移」喂给 WMP 学习。
        应在 choose_action 拿到下一帧后调用（或 Agent 内部自动记录）。
        """
        before = self._state_to_symbols(before_state)
        after = self._state_to_symbols(after_state)
        self.wmp.learn(WMPEvidence(action=action, before=before, after=after))
        self.wmp.compile()
        # 漂移检测（若已有预测）
        predicted = self.wmp.simulate(before_state, action)
        if predicted is not None:
            d = self.wmp.drift(before_state, action, after_state)
            self.wmp.record_drift(d)
            self.telemetry.record(drift=d)

    def _state_to_symbols(self, state: Dict) -> SymbolTable:
        """state dict → 符号表（对接 WMP 学习）。"""
        st = SymbolTable(grid_w=self.grid_w, grid_h=self.grid_h)
        for oid, o in state.get("objects", {}).items():
            st.add_object(o["x"], o["y"], color=o.get("color", 1),
                          role=o.get("role", "?"), obj_id=oid)
        st.avatar_id = state.get("avatar_id")
        return st

    # ---------- 序列化 ----------

    def save_telemetry(self, path: Optional[str] = None):
        """跑分结束调用：落盘观测数据（供后续分析喂回）。"""
        self.telemetry.save(path or self.telemetry.log_path)

    def stats(self) -> Dict:
        """汇总统计。"""
        s = dict(self.telemetry.summary())
        s.update({
            "step_count": self.step_count,
            "llm_budget_remaining": self.llm_budget,
            "wmp_rules": len(self.wmp.inducer.confident_rules())
                if hasattr(self.wmp, "inducer") else 0,
            "wmp_drift": (
                sum(self.wmp._drift_accum[-8:]) / len(self.wmp._drift_accum[-8:])
                if self.wmp._drift_accum else 0.0
            ),
        })
        return s
