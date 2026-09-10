"""SSA × CEAX 嫁接层：把谱代数认知内核注入假设-实验-利用闭环。

SpectralCeaxController(CeaxController) 只做加法、不动基类语义：
  1) observe_outcome 后把 (prev_grid, grid, action, progressed) 喂给 SpectralMind
     —— 四子空间 / 进度轴 / 特征技能 / 周期检测全部在线积累。
  2) _ingest_objects 后为未尝试点击目标计算谱先验（区域杠杆+特征轴+活块），
     _next_experiment_click 按谱先验排序未尝试目标 —— 探索预算花在
     "可控杠杆高 / 从未试过 / 非死区" 的对象上。
  3) _rank_simple 融合谱动作评分（epistemic + 预测进度 + 杠杆）。
  4) 所有谱注入均异常护栏包裹：谱内核任何故障都退化为基类行为。

理论：SSA 白皮书 §3/§5.3；数学：[S6] Ch3/4/7/9。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import os

import numpy as np

from ..lincore import MatrixMemory, SpectralMind
from ..lincore.features import block_of
from ..lincore.hypothesis_lab import HypothesisLab
from ..lincore.ttt import SharedBasis
from .ceax_controller import CeaxController, ClickTarget, _obj_centroid


class SpectralCeaxController(CeaxController):
    """CEAX 控制器 + SpectralMind 谱认知内核（SSA-L1..L5 在线）。"""

    def __init__(self, cfg=None, logger=None, game_sig: str = "unknown"):
        super().__init__(cfg=cfg, logger=logger)
        self.game_sig = str(game_sig or "unknown")
        self.mind = SpectralMind(cfg=cfg, game_sig=self.game_sig, memory=MatrixMemory.shared())
        # SSA-E3：环境变量 LINCORE_SHARED_BASIS 指向共享基时自动热启动（谱-TTT 测试时适配）
        try:
            _sb_path = os.environ.get("LINCORE_SHARED_BASIS", "")
            if _sb_path:
                self.mind.warm_start(SharedBasis.load(_sb_path))
        except Exception:
            pass
        self.lab = HypothesisLab()
        self._spectral_prior: Dict[Tuple, float] = {}
        self._spectral_step = 0
        self._hyp_grid_shape = (64, 64)
        self._steps_since_progress = 0

    # ---------- 生命周期 ----------

    def reset_game(self, *, import_skills: Optional[dict] = None):
        super().reset_game(import_skills=import_skills)
        # 新局 = 新算子：谱世界模型重置；跨游戏矩阵记忆保留（shared 单例）
        try:
            self.mind = SpectralMind(
                cfg=self.cfg, game_sig=self.game_sig, memory=MatrixMemory.shared()
            )
        except Exception:
            pass
        self._spectral_prior.clear()
        self._spectral_step = 0
        self.lab = HypothesisLab()

    def on_level_up(self):
        super().on_level_up()
        # 换关 = 动力学切换：目标已由基类清空；谱先验、假设与子空间一并重置
        try:
            self.mind = SpectralMind(
                cfg=self.cfg, game_sig=self.game_sig, memory=MatrixMemory.shared()
            )
        except Exception:
            pass
        self.lab = HypothesisLab()
        self._spectral_prior.clear()

    # ---------- 观测 ----------

    def observe_outcome(
        self,
        *,
        delta_pixels: int,
        progressed: bool,
        grid_hash: str = "",
        levels: int = 0,
        prev_grid=None,
        grid=None,
    ):
        super().observe_outcome(
            delta_pixels=delta_pixels,
            progressed=progressed,
            grid_hash=grid_hash,
            levels=levels,
            prev_grid=prev_grid,
            grid=grid,
        )
        try:
            self.mind.observe(
                prev_grid,
                grid,
                self._last_action,
                levels=levels,
                progressed=progressed,
            )
        except Exception:
            pass
        # SSA-E2：若上一步是假设实验，用效应块支撑集裁决
        try:
            blocks = self._delta_blocks(prev_grid, grid)
            if blocks is not None:
                self.lab.verify(blocks)
        except Exception:
            pass
        # SSA-E2 自动假设生成：持续无进度且假设库枯竭时，把谱信号升级为显式假设
        try:
            if progressed:
                self._steps_since_progress = 0
            else:
                self._steps_since_progress += 1
            pending = self.lab.snapshot().get("pending", 0)
            if (
                self._steps_since_progress >= 24
                and self._steps_since_progress % 24 == 0
                and pending < 2
            ):
                self.auto_register_from_mind()
        except Exception:
            pass
        self._spectral_step += 1
        if self._spectral_step % 16 == 0:
            try:
                self.mind.maintenance()
            except Exception:
                pass

    def _delta_blocks(self, prev_grid, grid) -> Optional[List[int]]:
        """(prev, curr) 网格 → 发生变化的块索引列表；无法计算返回 None。"""
        if prev_grid is None or grid is None:
            return None
        p, c = np.asarray(prev_grid), np.asarray(grid)
        if p.shape != c.shape or p.ndim != 2:
            return None
        self._hyp_grid_shape = c.shape
        ys, xs = np.where(p != c)
        if xs.size == 0:
            return []
        caps = xs[:200], ys[:200]
        return sorted(
            {block_of(int(x), int(y), c.shape, self.mind.blocks)
             for x, y in zip(*caps)}
        )

    def register_hypotheses(self, payload) -> int:
        """LLM JSON 建议批量注册为待验证假设（畸形输入安全忽略）。返回注册数。"""
        n0 = len(self.lab.hyps)
        for h in HypothesisLab.from_llm_json(payload, blocks=self.mind.blocks,
                                             shape=self._hyp_grid_shape):
            self.lab.register(
                action=h.action,
                text=h.text,
                xy=h.xy,
                expect_blocks=h.expect_blocks,
                prior=h.prior,
                hid=h.hid,
                source="llm",
            )
        return len(self.lab.hyps) - n0

    def auto_register_from_mind(self, top_k: int = 2) -> int:
        """谱信号 → 显式可证伪假设（LLM 缺席时的自动 System 2）。"""
        return self.lab.auto_from_mind(
            self.mind, shape=self._hyp_grid_shape, top_k=top_k
        )

    # ---------- 决策 ----------

    def choose(
        self,
        *,
        valid_actions: Sequence[str],
        objects: Optional[Sequence[Any]] = None,
        grid=None,
        grid_hash: str = "",
    ) -> Tuple[str, Optional[Tuple[int, int]], str]:
        try:
            self.mind.set_state_context(grid)
        except Exception:
            pass
        if grid is not None:
            try:
                self._hyp_grid_shape = np.asarray(grid).shape
            except Exception:
                pass
        act, xy, why = super().choose(
            valid_actions=valid_actions, objects=objects, grid=grid, grid_hash=grid_hash
        )
        try:
            shape = np.asarray(grid).shape if grid is not None else None
            self.mind.note_choice(act, xy, shape)
            self.mind.note_action_taken(act)
        except Exception:
            pass
        return act, xy, why

    # ---------- 谱注入点 ----------

    def _ingest_objects(self, objects: Sequence[Any]) -> None:
        super()._ingest_objects(objects)
        try:
            cands = [
                (key, t.xy)
                for key, t in self.targets.items()
                if t.trials == 0 and t.xy is not None
            ]
            scores = self.mind.score_click_candidates(cands)
            self._spectral_prior = {
                key: float(scores.get(key, 0.0)) for key, _xy in cands
            }
        except Exception:
            self._spectral_prior = {}

    def _next_experiment_click(self) -> Optional[Tuple[Tuple[int, int], Tuple]]:
        """未尝试目标改按谱先验排序（杠杆×未尝试×活块），退化到基类次序。

        优先级：待验证假设实验 > 谱先验排序 > 基类次序。
        """
        try:
            planned = self.lab.design_experiment()
            if planned is not None:
                action, xy, _why = planned
                if action == "ACTION6" and xy is not None:
                    return xy, ("hyp", xy[0] // 8, xy[1] // 8)
            untried = [t for t in self.targets.values() if t.trials == 0]
            if len(untried) >= 2 and self._spectral_prior:
                def sort_key(t: ClickTarget):
                    sp = float(self._spectral_prior.get(t.key, 0.0))
                    pref = 0 if t.color in self._preferred_colors else 1
                    return (-sp, pref, t.area, t.color)
                untried.sort(key=sort_key)
                t = untried[0]
                return t.xy, t.key
        except Exception:
            pass
        return super()._next_experiment_click()

    def _rank_simple(self, actions: Sequence[str], grid_hash: str) -> List[str]:
        """简单动作排序：基类证据 + 谱评分（权重随探索门伸缩，SSA-E3 决策入口）。

        谱权重 w = 0.3·(0.5 + explore_gate)/1.5 ∈ [0.1, 0.3]：
        近期效应对世界先验越新奇 → 谱信号（含未尝试/世界新奇度）话语权越大。
        """
        base = super()._rank_simple(actions, grid_hash)
        try:
            gate = self.mind.explore_gate()
            w = max(0.10, min(0.30, 0.3 * (0.5 + gate) / 1.5))
            spectral = self.mind.score_actions(base)
            if spectral:
                vals = sorted(spectral.values())
                lo, hi = vals[0], vals[-1]
                span = (hi - lo) or 1.0

                def blend(i: int, name: str) -> tuple:
                    sp = (float(spectral.get(name, 0.0)) - lo) / span
                    return (i + w * sp, name)

                ranked = sorted(
                    (blend(i, n) for i, n in enumerate(base)), reverse=True
                )
                return [n for _s, n in ranked]
        except Exception:
            pass
        return base

    # ---------- 诊断 ----------

    def snapshot(self) -> dict:
        snap = super().snapshot()
        try:
            snap["spectral"] = self.mind.snapshot()
        except Exception:
            snap["spectral"] = {"error": True}
        try:
            snap["lab"] = self.lab.snapshot()
        except Exception:
            snap["lab"] = {"error": True}
        return snap
