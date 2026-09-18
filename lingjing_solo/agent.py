"""ZovaX · 单 Agent 编排（6 角色分工闭环）

Tick：感知 encoder → 场 field → 探索 explorer → 搜索 search → 顾问 advisor → 反思 reflector
"""
from __future__ import annotations

from .core import (
    SoloConfig, Frame, Logger, MANIFESTO,
    extract_grid, extract_state, extract_levels, extract_available_actions,
    is_win_state, needs_reset, canonicalize, normalize_valid,
)
from .perception import PerceptionEncoder
from .world_model import WorldModelField
from .exploration import ExplorationEngine
from .exploration.click_sweep import BubbleClickPlanner
from .planning import SearchEngine, StrategicAdvisor
from .planning.discrete_nav import DiscreteNavPlanner
from .planning.ls20_solver import Ls20Solver, looks_like_ls20
from .reflection import ReflectionTrigger
from .transfer import TransferLayer


def _try_game_action(name: str):
    try:
        from arcengine import GameAction
        key = canonicalize(name)
        if hasattr(GameAction, key):
            return getattr(GameAction, key)
        if hasattr(GameAction, "from_name"):
            return GameAction.from_name(key)
    except Exception:
        pass
    return name


class LingjingSoloAgent:
    """ZovaX / 灵境 Solo 化身（统一信息场 Φ + 6 模块编排）。"""

    def __init__(self, cfg: SoloConfig = None, logger: Logger = None, llm_fn=None):
        self.cfg = cfg or SoloConfig()
        self.log = logger or Logger()
        self.step = 0
        self._prev_frame: Frame | None = None
        self._last_action = None
        self._levels_seen = 0
        self.last_perception = None
        self.last_rationale = ""
        self.last_click = None
        self._last_was_click = False

        self.encoder = PerceptionEncoder(self.cfg, self.log)
        self.field = WorldModelField(self.cfg, self.log)
        self.explorer = ExplorationEngine(self.cfg, self.field, self.log)
        self.clicks = BubbleClickPlanner(self.cfg, self.log)
        self.search = SearchEngine(self.cfg, self.field, self.explorer, self.log)
        self.advisor = StrategicAdvisor(self.cfg, self.log)
        self.reflector = ReflectionTrigger(self.cfg, self.field, self.log)
        self.discrete_nav = DiscreteNavPlanner(self.cfg, self.log)
        self.discrete_nav.bind_hasher(lambda: self.field.current_hash())
        self.ls20 = Ls20Solver(self.cfg, self.log)
        self.transfer = TransferLayer(self.cfg, field=self.field, logger=self.log)

        # 向后兼容别名
        self.planner = self.search
        self.llm = self.advisor

        if llm_fn is not None:
            self.advisor.inject_llm(llm_fn)
        self.log.log("Agent", MANIFESTO[:40] + "…")

    def reset(self, env=None):
        exported = None
        if getattr(self, "transfer", None) is not None:
            exported = self.transfer.on_episode_end(self.field)
        self.field.reset()
        self.explorer = ExplorationEngine(self.cfg, self.field, self.log)
        self.clicks = BubbleClickPlanner(self.cfg, self.log)
        self.search = SearchEngine(self.cfg, self.field, self.explorer, self.log)
        self.advisor.calls_used = 0
        self.reflector = ReflectionTrigger(self.cfg, self.field, self.log)
        self.discrete_nav = DiscreteNavPlanner(self.cfg, self.log)
        self.discrete_nav.bind_hasher(lambda: self.field.current_hash())
        self.ls20 = Ls20Solver(self.cfg, self.log)
        self.planner = self.search
        self.llm = self.advisor
        # L6：新游戏清空局内假设；仅按机制族跨游戏导入技能
        if not hasattr(self, "transfer") or self.transfer is None:
            self.transfer = TransferLayer(self.cfg, field=self.field, logger=self.log)
        else:
            self.transfer.bind_field(self.field)
            self.transfer.reset_episode(keep_skills=False, import_cross_game=exported)
        game_id = getattr(env, "game_id", None) or getattr(env, "id", None)
        self.transfer.set_game_id(str(game_id) if game_id else None)
        self.step = 0
        self._prev_frame = None
        self._last_action = None
        self._levels_seen = 0
        self.last_perception = None
        self.last_rationale = ""
        self.last_click = None
        self._last_was_click = False
        self.log.log("Agent", "reset · new game field version=0")

    def _parse_frame(self, latest_frame) -> Frame | None:
        grid = extract_grid(latest_frame)
        if grid is None:
            return None
        return Frame(
            grid=grid,
            t=self.step,
            state=extract_state(latest_frame),
            levels_completed=extract_levels(latest_frame),
            available_actions=extract_available_actions(
                latest_frame, fallback=self.cfg.allowed_actions
            ),
            raw=latest_frame,
        )

    def is_done(self, frames, latest_frame) -> bool:
        state = extract_state(latest_frame)
        if is_win_state(state):
            self.field.note_win(
                grid=extract_grid(latest_frame),
                levels=extract_levels(latest_frame),
            )
            return True
        cap = int(self.cfg.human_baseline_estimate * self.cfg.hard_step_multiplier)
        return self.step >= cap

    def _advisor_blocked(self) -> bool:
        return self.reflector.should_give_up(self.advisor)

    def choose_action(self, frames, latest_frame, valid_actions=None):
        curr = self._parse_frame(latest_frame)

        if curr is None:
            return self._emit("RESET" if needs_reset(latest_frame) else self.cfg.allowed_actions[0])
        if needs_reset(curr.state):
            self.field.inject_source_term("RESET", rationale="reset")
            return self._emit("RESET")

        if self._advisor_blocked() and self.field.noop_streak >= 8:
            # 封闭模式：符号猜想耗尽且长期无进展 → 提前结束本局
            self.last_rationale = "symbolic_give_up"
            return self._emit(self.cfg.allowed_actions[0])

        self.step += 1
        curr.t = self.step

        prev_grid = self._prev_frame.grid if self._prev_frame is not None else None
        snap = self.encoder.perceive(
            prev_grid, curr.grid,
            version=self.field.version,
            tick=self.step,
            levels=curr.levels_completed,
            env_state=curr.state,
        )
        self.last_perception = snap

        if self._prev_frame is not None and self._last_action is not None:
            self.clicks.observe_effect(
                delta_pixels=len(snap.delta_pixels) if snap.delta_pixels else 0,
                progressed=curr.levels_completed > self._levels_seen,
                was_click=self._last_was_click,
            )
            self.explorer.record_action_outcome(
                self._last_action,
                len(snap.delta_pixels) if snap.delta_pixels else 0,
                curr.levels_completed > self._levels_seen,
            )
            self.discrete_nav.observe(
                self._prev_frame.raw if self._prev_frame else None,
                curr.raw,
                self._last_action,
                self.field.current_hash(),
                curr.levels_completed,
            )
            self.ls20.observe(
                self._prev_frame.grid if self._prev_frame else None,
                curr.grid,
                self._last_action,
                curr.levels_completed,
            )

        self.field.update(curr, self._prev_frame, self._last_action)

        delta_px = 0
        progressed = curr.levels_completed > self._levels_seen
        if self._prev_frame is not None and self.last_perception is not None:
            delta_px = len(self.last_perception.delta_pixels) if self.last_perception.delta_pixels else 0
        self.transfer.observe(
            grid=curr.grid,
            objects=getattr(self.last_perception, "objects", None) if self.last_perception else None,
            actions=valid_actions or curr.available_actions,
            level=curr.levels_completed,
            grid_hash=self.field.current_hash(),
            last_action=self._last_action,
            delta_pixels=delta_px,
            progressed=progressed,
        )

        if curr.levels_completed > self._levels_seen:
            self.search.clear_plan()
            self.discrete_nav.reset_level()
            self.explorer.on_environment_feedback(
                curr.state, curr.levels_completed, self._levels_seen
            )
            self.clicks.reset()
            # L6：关卡升级 — Φ/探索局部重置由各模块负责；技能库持久
            self.transfer.on_level_up(self.field)
            self._levels_seen = curr.levels_completed

        self.explorer.induce_rules()
        valid = self._valid(curr, valid_actions)

        if self.field.is_loop():
            self.search.clear_plan()

        action, rationale = None, "fallback"
        click_xy = None
        discrete = DiscreteNavPlanner.is_discrete_game(valid)
        ls20_like = discrete and looks_like_ls20(curr.grid)

        # ---- ls20：BFS + 旋转台 + 目标 ----
        if ls20_like:
            ls = self.ls20.plan(valid)
            layout_wait = getattr(self.ls20, "_layout_wait", 0)
            if (
                ls is None
                and not self.ls20._await_level_layout
                and not layout_wait
                and not self.ls20.active
            ):
                self.ls20.reset_level(curr.grid)
                ls = self.ls20.plan(valid)
            if ls is None and layout_wait and "ACTION1" in valid:
                ls, rationale = "ACTION1", "ls20_layout_wait"
            if ls and ls in valid:
                action, rationale = ls, "ls20_solver"

        # ---- 其他离散关：4 向 blob 导航 ----
        if action is None and discrete and self.discrete_nav.steps_on_level >= 2:
            nav = self.discrete_nav.plan(valid, self.field.current_hash(), curr.levels_completed)
            if nav and nav in valid:
                action, rationale = nav, "discrete_nav"

        if (
            action is None
            and not discrete
            and self.clicks.should_click(valid)
            and "ACTION6" in valid
            and not (self.reflector.evaluate().should_reflect and self.advisor.can_call())
        ):
            self.clicks.refill(
                curr.grid,
                bubble=self.field.bubble or snap.bubble,
                objects=snap.objects,
                phi=self.field.phi or snap.phi,
            )
            click_xy = self.clicks.next_xy(
                curr.grid,
                bubble=self.field.bubble or snap.bubble,
                objects=snap.objects,
                phi=self.field.phi or snap.phi,
            )
            if click_xy is not None:
                action, rationale = "ACTION6", "bubble_click"

        if action is None and self.reflector.should_reflect_now(self.advisor) and self.advisor.can_call():
            ctx = self.reflector.pack_context(valid)
            # R5：注入迁移技能上下文
            try:
                ctx.skills_context = self.transfer.r5_skill_context()  # type: ignore[attr-defined]
            except Exception:
                pass
            failed = [r for r in self.field.rules if r.confidence < self.cfg.rule_prune_threshold]
            hints = self.advisor.suggest_hypotheses(
                ctx, failed, valid_actions=valid, explorer=self.explorer, field=self.field,
            )
            self.reflector.mark_symbolic_exhausted(self.advisor)
            for hint in hints:
                if "ACTION" in hint:
                    for tok in hint.split():
                        if tok.startswith("ACTION") and tok in valid:
                            action = tok
                            rationale = "advisor_hint"
                            break
                if action:
                    break
            if action is None:
                action = self.advisor.plan(ctx, valid)
                if action and action in valid:
                    rationale = "reflect"

        if action is None:
            plan_valid = [a for a in valid if a != "ACTION6"] or valid
            plan = self.search.search(valid_actions=plan_valid)
            if plan and plan in valid:
                action, rationale = plan, "search"

        # ---- L6 CEAX：技能先验 + 反事实×explore 融合（探索兜底前）----
        if action is None and getattr(self.cfg, "transfer_priority_before_explore", True):
            scored = self.explorer.score_actions(
                [a for a in valid if a != "ACTION6"] if not discrete else valid
            )
            suggestion = self.transfer.suggest_action(
                valid,
                explore_scored=scored,
                state_hash=self.field.current_hash(),
            )
            if suggestion and suggestion[0] in valid:
                action, rationale = suggestion

        if action is None:
            scored = self.explorer.score_actions(
                [a for a in valid if a != "ACTION6"] if not discrete else valid
            )
            best = scored[0][1] if scored else -999
            if (
                not discrete
                and best < 0.2
                and "ACTION6" in valid
                and self.clicks.should_click(valid, force=True)
            ):
                click_xy = self.clicks.next_xy(
                    curr.grid,
                    bubble=self.field.bubble or snap.bubble,
                    objects=snap.objects,
                    phi=self.field.phi or snap.phi,
                )
                if click_xy is not None:
                    action, rationale = "ACTION6", "bubble_click_lowgain"
            if action is None:
                chosen = self.search.greedy_next(scored)
                if chosen and chosen in valid:
                    action, rationale = chosen, "explore"

        if action is None:
            action = valid[0]

        return self._commit(action, curr, rationale=rationale, click_xy=click_xy)

    def _valid(self, curr: Frame, valid_actions):
        if valid_actions:
            valid = normalize_valid(valid_actions)
        elif curr.available_actions:
            valid = normalize_valid(curr.available_actions)
        else:
            valid = list(self.cfg.allowed_actions)
        if not self.cfg.enable_undo:
            valid = [a for a in valid if a != "ACTION7"]
        if not (self.cfg.enable_mouse or self.cfg.enable_click_sweep):
            valid = [a for a in valid if a != "ACTION6"]
        elif "ACTION6" not in valid and (self.cfg.enable_mouse or self.cfg.enable_click_sweep):
            if not curr.available_actions:
                valid = list(valid) + ["ACTION6"]
        return valid or list(self.cfg.allowed_actions)

    def _commit(self, action: str, curr: Frame = None, rationale: str = "act",
                click_xy=None):
        action = canonicalize(action)
        self.last_rationale = rationale
        self._last_was_click = action == "ACTION6"
        if action == "ACTION6":
            self.last_click = click_xy or self.clicks.last_xy or (32, 32)
        else:
            self.last_click = None
            self.clicks.note_simple_action()
        self.field.inject_source_term(action, rationale=rationale)
        self._last_action = action
        if curr is not None:
            self._prev_frame = curr
        return self._emit(action)

    def _emit(self, action: str):
        action = canonicalize(action)
        if self.cfg.return_game_action:
            return _try_game_action(action)
        return action

    @staticmethod
    def _to_grid(frame):
        return extract_grid(frame)

    @classmethod
    def with_llm(cls, llm_fn, **cfg_kwargs) -> "LingjingSoloAgent":
        return cls(cfg=SoloConfig(**cfg_kwargs), llm_fn=llm_fn)
