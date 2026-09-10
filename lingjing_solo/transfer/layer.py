"""L6 CEAX Transfer 门面：接入 Solo tick，关卡隔离下技能库持久化。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .counterfactual import CounterfactualPlanner
from .families import FamilyTransfer
from .hypotheses import CompetingHypotheses, HypothesisKind
from .rule_primitives import RuleComposer
from .rule_transfer import RuleTransfer
from .signatures import StateSignature, build_signature
from .skill_library import SkillLibrary
from .world_model import CounterfactualWorldModel


class TransferLayer:
    """
    Tick 钩子：
      1. observe_transition → 更新本地 WM + 结构化假设
      2. suggest_action → 技能先验区分实验 + 反事实×explore 融合
      3. on_level_up → Φ 局部由 agent reset；技能库保留并抽取
      4. on_episode_end → 抽取技能；可 export 跨局
    """

    def __init__(self, cfg=None, field=None, logger=None):
        self.cfg = cfg
        self.log = logger
        min_conf = float(getattr(cfg, "transfer_min_confidence", 0.75)) if cfg else 0.75
        self.enabled = bool(getattr(cfg, "enable_transfer", True)) if cfg else True
        self.cf_blend = float(getattr(cfg, "transfer_cf_blend", 0.55)) if cfg else 0.55
        self.skills = SkillLibrary(min_confidence=min_conf)
        self.hyps = CompetingHypotheses()
        self.model = CounterfactualWorldModel(field=field)
        self.planner = CounterfactualPlanner()
        self.transfer = RuleTransfer(min_confidence=min_conf)
        self.composer = RuleComposer(min_confidence=min_conf)
        self.family = FamilyTransfer(self.skills)
        self._last_sig: Optional[StateSignature] = None
        self._experiments = 0
        self._transfer_suggests = 0
        self._game_id: Optional[str] = None
        self._register_compositional_predictors()

    def bind_field(self, field) -> None:
        self.model.bind_field(field)

    def set_game_id(self, game_id: Optional[str]) -> None:
        self._game_id = game_id

    # ---- 生命周期：整局 reset 清空 Φ 侧，但可选择保留技能 ----
    def reset_episode(self, *, keep_skills: bool = True, import_cross_game: Optional[dict] = None):
        self.hyps = CompetingHypotheses()
        self.model = CounterfactualWorldModel(field=self.model.field)
        self._register_compositional_predictors()
        self._last_sig = None
        self._experiments = 0
        self._transfer_suggests = 0
        if not keep_skills:
            self.skills.clear()
        if import_cross_game:
            self.family.import_for_game(import_cross_game, self._game_id)

    def on_level_up(self, field=None):
        """关卡升级：抽取当前技能，不清空 SkillLibrary。"""
        self._extract_into_library(field)
        self.hyps = CompetingHypotheses()
        # 本地签名转移可清；Φ 场由 agent 负责 reset
        self.model.transitions.clear()
        self._last_sig = None
        if self.log:
            self.log.log(
                "Transfer",
                f"level_up skills={len(self.skills.skills)} hit_rate={self.skills.transfer_hit_rate:.2f}",
            )

    def on_episode_end(self, field=None) -> Dict[str, dict]:
        self._extract_into_library(field)
        return self.skills.export()

    # ---- 观测 ----
    def observe(
        self,
        *,
        grid=None,
        objects=None,
        actions=None,
        level: int = 0,
        grid_hash: str = "",
        last_action: Optional[str] = None,
        prev_grid=None,
        delta_pixels: int = 0,
        progressed: bool = False,
    ) -> StateSignature:
        sig = build_signature(
            grid,
            objects=objects,
            actions=actions,
            level=level,
            grid_hash=grid_hash,
        )
        if self._last_sig is not None and last_action is not None:
            prev = self._last_sig
            key = prev.grid_hash or prev
            predicted = self.model.predict(key, last_action)
            if predicted is None:
                self._experiments += 1
            self.model.observe(key, last_action, sig.grid_hash or sig)
            self._learn_from_delta(
                last_action, delta_pixels=delta_pixels, progressed=progressed,
                changed=(prev.grid_hash != sig.grid_hash) if prev.grid_hash and sig.grid_hash else delta_pixels > 0,
            )
        self._last_sig = sig
        return sig

    def _learn_from_delta(self, action: str, *, delta_pixels: int, progressed: bool, changed: bool):
        action_u = str(action).upper()
        if not changed and delta_pixels <= 0:
            if "ACTION6" in action_u or action_u in ("SPACE", "CLICK"):
                self.hyps.support(HypothesisKind.NOOP, evidence=action)
            else:
                self.hyps.support(HypothesisKind.BLOCKED_TRANSITION, evidence=action)
        elif changed or delta_pixels > 0:
            self.hyps.support(HypothesisKind.MOVEMENT, evidence=(action, delta_pixels))
            if "ACTION6" in action_u or action_u in ("SPACE", "CLICK"):
                self.hyps.support(HypothesisKind.CLICK_EFFECT, evidence=(action, delta_pixels))
            # 中等变化 + 非纯移动 → 开关/切换候选
            if 0 < delta_pixels < 200 and progressed is False:
                self.hyps.support(HypothesisKind.TOGGLE_ON_CONTACT, evidence=(action, delta_pixels), amount=0.08)
        if progressed:
            self.hyps.support(HypothesisKind.LEVEL_PROGRESS, evidence=action, amount=0.15)

    # ---- 决策建议 ----
    def suggest_action(
        self,
        valid_actions: Sequence[str],
        *,
        explore_scored: Optional[Sequence[Tuple[str, float]]] = None,
        state_hash: str = "",
    ) -> Optional[Tuple[str, str]]:
        """返回 (action, rationale) 或 None（让上层继续 ls20/search）。"""
        if not self.enabled or not valid_actions:
            return None

        skill = self.skills.prioritize()
        state_key = state_hash or (self._last_sig.grid_hash if self._last_sig else "") or self._last_sig

        # 技能先验：优先尝试未充分观测的区分动作
        if skill:
            self._transfer_suggests += 1
            prefer = self._actions_for_skill(skill, valid_actions)
            if prefer:
                # 在 prefer 集合里选 epistemic 最高者
                ranked = self.planner.rank(state_key, prefer, self.model)
                if ranked:
                    return ranked[0][1], f"transfer_skill:{skill}"

        if explore_scored:
            fused = self.planner.fuse_with_explore(
                state_key, explore_scored, self.model, cf_blend=self.cf_blend
            )
            if fused:
                return fused[0][0], "transfer_cf_fuse"
        else:
            ranked = self.planner.rank(state_key, list(valid_actions), self.model)
            if ranked and ranked[0][0] > 0.5:
                return ranked[0][1], "transfer_epistemic"
        return None

    def _actions_for_skill(self, skill: str, valid: Sequence[str]) -> List[str]:
        v = list(valid)
        su = skill.lower()
        if "click" in su:
            clicks = [a for a in v if "ACTION6" in str(a).upper() or str(a).upper() in ("SPACE", "CLICK")]
            return clicks or v
        if "toggle" in su or "blocked" in su or "gate" in su or "switch" in su:
            # 非点击方向/交互优先做区分
            dirs = [a for a in v if str(a).upper() not in ("ACTION7",)]
            return dirs or v
        if "movement" in su or "navigation" in su:
            return [a for a in v if str(a).upper().startswith("ACTION") and "ACTION6" not in str(a).upper()] or v
        return v

    def _extract_into_library(self, field=None):
        from_hyps = self.transfer.extract(self.hyps.ranked())
        self.skills.import_many(from_hyps, cross_game=False)
        if field is not None:
            from_field = self.transfer.extract_from_field(field)
            self.skills.import_many(from_field, cross_game=False)
        # 注册组合预测器
        self._register_compositional_predictors()

    def _register_compositional_predictors(self):
        """Phase3：未见 (s,a) 时用技能族做弱预测占位（None=强制实验）。"""
        self.model.clear_rule_predictors()
        skills = self.skills.export()

        def _predict(state_sig, action):
            # 有 click_effect 技能且动作为点击 → 不假装知道后继，返回 None 促实验
            # 有 movement 且动作为方向 → 若本地无边，仍返回 None（epistemic）
            # 组合技能存在时，标记「应优先测 toggle」但不伪造 next hash
            _ = (state_sig, action, skills)
            return None

        self.model.register_rule_predictor(_predict)

    # ---- 指标 / R5 上下文 ----
    def metrics(self) -> Dict[str, Any]:
        return {
            "skills": len(self.skills.skills),
            "experiments": self._experiments,
            "transfer_suggests": self._transfer_suggests,
            "transfer_hit_rate": self.skills.transfer_hit_rate,
            "hypotheses": [(h.name, round(h.confidence, 2)) for h in self.hyps.ranked()],
        }

    def r5_skill_context(self) -> str:
        if not self.skills.skills:
            return "（暂无迁移技能）"
        lines = []
        for name, val in sorted(
            self.skills.skills.items(),
            key=lambda kv: float(kv[1].get("confidence", 0)),
            reverse=True,
        )[:8]:
            lines.append(
                f"- {name} family={val.get('family')} "
                f"conf={float(val.get('confidence', 0)):.2f} :: {val.get('representation')}"
            )
        conflict = [h for h in self.hyps.ranked() if h.contradictions]
        if conflict:
            lines.append("冲突假设：" + ", ".join(h.name for h in conflict[:4]))
        skill = self.skills.prioritize()
        if skill:
            lines.append(f"建议区分实验技能：{skill}")
        return "\n".join(lines)
