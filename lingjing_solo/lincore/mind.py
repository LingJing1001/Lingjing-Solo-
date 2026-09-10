"""SSA-L3/L4 · SpectralMind：三信号评估器与决策融合的编排层。

统一子系统（SSA 白皮书 §3）：
  SubspaceModel  —— 四子空间世界模型 + 进度轴 p* + 惊讶/可控/进度三信号
  EigenSkillMap  —— 区域×动作杠杆图 + 特征选项
  LogisticSGD    —— 进度概率的流式预测器（动作 onehot + 状态摘要）
  DualBudgetScheduler —— 探索预算对偶调度
  detect_period  —— 变化序列周期检测（闪烁门/周期机制）
  MatrixMemory   —— 跨游戏低秩迁移先验（进程级共享）

对外三问（CEAX 控制器在每步询问）：
  score_click_candidates(objects) —— 点击候选的谱评分
  score_actions(valid, grid)       —— 键盘/简单动作的谱评分
  observe(...)                     —— 每步反馈（效应、进度、区域杠杆）
全部方法异常安全（返回中性值），保证不击落上层决策链。
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .eigen import EigenSkillMap
from .features import activity_mask, block_center, block_of, encode_grid, grid_summary
from .gradient import DualBudgetScheduler, LogisticSGD
from .memory import MatrixMemory
from .spectral import detect_period
from .subspace import SubspaceModel
from .ttt import SharedBasis, TransitionReplay, combined_row_space, residual_novelty

DEFAULT_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7")


class SpectralMind:
    """谱认知内核编排器（线程不共享；每 agent 一例）。"""

    def __init__(
        self,
        cfg=None,
        blocks: int = 8,
        actions: Sequence[str] = DEFAULT_ACTIONS,
        history: int = 256,
        game_sig: str = "unknown",
        memory: Optional[MatrixMemory] = None,
        shared_basis: Optional[SharedBasis] = None,
        replay: Optional[TransitionReplay] = None,
    ) -> None:
        self.blocks = max(2, int(blocks))
        self.dim = 2 * self.blocks * self.blocks
        self.sub = SubspaceModel(self.dim, k=8, max_rows=400)
        self.eigen = EigenSkillMap(self.blocks * self.blocks, actions or DEFAULT_ACTIONS)
        self.z_dim = len(self.eigen.actions) + 4
        self.sgd = LogisticSGD(self.z_dim, lr=0.08, momentum=0.9, l2=1e-4)
        self.scheduler = DualBudgetScheduler(horizon=800, explore_share=0.25)
        self.memory = memory if memory is not None else MatrixMemory.shared()
        self.game_sig = str(game_sig or "unknown")
        # SSA-E3：世界先验基（谱-TTT 的"预训练"侧）与转移重放（"重训数据"侧）
        self.shared_basis = shared_basis
        self.replay = replay if replay is not None else TransitionReplay()
        self._prior_Q: Optional[np.ndarray] = None

        self.activity = None  # 像素级"曾经变化"掩码（N(E) 支撑集）
        self._change_counts: deque = deque(maxlen=int(history))
        self._period: Optional[int] = None
        self._z_pending: Optional[np.ndarray] = None
        self._pending_region: Optional[int] = None
        self._pending_action: Optional[str] = None
        self._recent_actions: deque = deque(maxlen=12)
        self._gain_ema: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._prev_feat: Optional[np.ndarray] = None
        self._ctx_summary: Optional[np.ndarray] = None
        self._step = 0
        # 世界先验的决策接入口：每动作效应签名（对先验的新奇度 → 探索加权）
        self._action_sig: Dict[str, np.ndarray] = {}
        self._recent_novelty: float = 0.0
        self._novelty_hist: deque = deque(maxlen=16)

    # ---------- 每步观测 ----------

    def observe(
        self,
        prev_grid,
        grid,
        action: Optional[str],
        levels: int = 0,
        progressed: bool = False,
    ) -> Dict[str, Any]:
        """反馈一步 (x_t, a_t, x_{t+1}, 进度)。返回诊断 dict（异常安全）。"""
        self._step += 1
        out: Dict[str, Any] = {}
        curr_feat = None
        try:
            prev_feat = encode_grid(prev_grid, self.blocks)
            curr_feat = encode_grid(grid, self.blocks)
            delta = None
            if prev_feat is not None and curr_feat is not None and prev_feat.shape == curr_feat.shape:
                delta = curr_feat - prev_feat
                mag = float(np.linalg.norm(delta))
                self.sub.update(delta, progressed=progressed)
                # 世界先验决策信号：该动作的效应签名 vs 先验行空间的新奇度
                act = str(action or "").upper()
                if mag > 1e-9:
                    sig = self._action_sig.get(act)
                    self._action_sig[act] = delta.copy() if sig is None else sig + delta
                    try:
                        self._novelty_hist.append(self.world_novelty(delta))
                    except Exception:
                        pass
                # SSA-E3：转移入库（离线重放重训共享基的数据源）
                if mag > 1e-9:
                    self.replay.append(self.game_sig, delta)
                self.sub.note_action(act, mag)
                self._gain_ema[act] = 0.8 * self._gain_ema.get(act, 0.0) + 0.2 * mag
                self._counts[act] = self._counts.get(act, 0) + 1
                # 像素活跃掩码（不变量支撑集）
                m = activity_mask(prev_grid, grid)
                if self.activity is None or self.activity.shape != m.shape:
                    self.activity = m.astype(bool)
                else:
                    self.activity |= m
                # 变化量序列（周期检测输入）
                changed = int((np.asarray(prev_grid) != np.asarray(grid)).sum()) if (
                    prev_grid is not None and grid is not None
                    and np.asarray(prev_grid).shape == np.asarray(grid).shape
                ) else 0
                self._change_counts.append(float(changed))
                # 区域杠杆：把效应量记账到上一动作的作用区域
                if self._pending_region is not None and self._pending_action:
                    self.eigen.note(self._pending_action, self._pending_region, min(1.0, mag / 4.0))
                # 进度预测器（SGD 在线一步）
                if self._z_pending is not None:
                    self.sgd.partial_fit(self._z_pending, 1.0 if progressed else 0.0)
                # 跨游戏记忆累积（每 32 步采样一次，避免过写）
                if self._step % 32 == 0 and self._counts:
                    self.memory.record(self.game_sig, {
                        f"a{int(k[-1])}_gain" if k.startswith("ACTION") else k: v
                        for k, v in self._gain_ema.items()
                    })
        except Exception:
            pass
        finally:
            self._prev_feat = curr_feat
            self._z_pending = None
            self._pending_region = None
            self._pending_action = None
        out["step"] = self._step
        out["rank"] = self.sub.effective_rank
        out["period"] = self._period
        return out

    def note_choice(self, action: str, xy: Optional[Tuple[int, int]] = None, grid_shape=None) -> None:
        """决策前登记：构建 SGD 训练样本 z 并记账区域杠杆（observe 时消费）。"""
        try:
            act = str(action or "").upper()
            self._pending_action = act
            if xy is not None and grid_shape is not None:
                self._pending_region = block_of(xy[0], xy[1], grid_shape, self.blocks)
            onehot = [1.0 if a == act else 0.0 for a in self.eigen.actions]
            z_extra = self._ctx_summary if self._ctx_summary is not None else np.zeros(4)
            z = np.concatenate([np.asarray(onehot, dtype=np.float64),
                                np.asarray(z_extra, dtype=np.float64)])[: self.z_dim]
            self._z_pending = z if z.size == self.z_dim else None
        except Exception:
            pass

    def set_state_context(self, grid) -> None:
        """为当前帧构建状态摘要（score_actions / note_choice 的公共输入）。"""
        try:
            s = grid_summary(grid, self.blocks)
            self._ctx_summary = None if s is None else s
        except Exception:
            self._ctx_summary = None

    # ---------- 决策评分 ----------

    def warm_start(self, shared_basis: Optional[SharedBasis]) -> bool:
        """注入世界先验基（谱-TTT 测试时适配的起点；无先验则 False）。"""
        if shared_basis is None or not shared_basis.ready:
            return False
        self.shared_basis = shared_basis
        self._prior_Q = None
        return True

    def world_novelty(self, delta) -> float:
        """带世界先验的新奇度：残差对「本局行空间 ∪ 先验行空间」的合并正交基。

        先验里见过的效应方向不再算新奇 —— 这是跨局迁移在感知层的落点。
        """
        try:
            if self._prior_Q is None:
                prior_V = self.shared_basis.V if self.shared_basis is not None else None
                self._prior_Q = combined_row_space([self.sub.row_basis(), prior_V])
            return residual_novelty(delta, self._prior_Q)
        except Exception:
            return self.sub.novelty(delta)

    def _progress_block_scores(self) -> Dict[int, float]:
        """块级进度对齐：p* 向块坐标平均（仅取占用维），归一化到 [0,1]。"""
        w = self.sub.progress_axis()
        if w is None:
            return {}
        b2 = self.blocks * self.blocks
        w_occ = np.abs(w[:b2])
        if float(w_occ.max()) <= 1e-12:
            return {}
        w_occ = w_occ / float(w_occ.max())
        return {i: float(v) for i, v in enumerate(w_occ)}

    def score_click_candidates(
        self, candidates: Sequence[Tuple[int, Tuple[int, int]]]
    ) -> Dict[int, float]:
        """candidates: [(id, (x,y))] → {id: score}。

        分数 = EigenSkill 排序分 ×（死块 ×0.3，活块 ×1.0）。
        """
        out: Dict[int, float] = {}
        try:
            shape = self.activity.shape if self.activity is not None else (64, 64)
            regions = [block_of(x, y, shape, self.blocks) for _i, (x, y) in candidates]
            ranked = self.eigen.rank_candidates(regions, self._progress_block_scores())
            score_by_region = dict(ranked)
            live_mask = None
            if self.activity is not None:
                ever_changed = self.activity
                live_mask = self._block_live(ever_changed)
            for (cid, (x, y)) in candidates:
                r = block_of(x, y, shape, self.blocks)
                s = score_by_region.get(r, 0.0)
                if live_mask is not None:
                    bi, bj = divmod(r, self.blocks)
                    if bi < live_mask.shape[0] and bj < live_mask.shape[1] and not live_mask[bi, bj]:
                        s *= 0.3
                out[cid] = float(s)
        except Exception:
            pass
        return out

    def _block_live(self, ever_changed: np.ndarray) -> np.ndarray:
        """像素活跃掩码 → 块级活性（任一像素变过 = 活块）。"""
        H, W = ever_changed.shape
        bh = max(1, -(-H // self.blocks))
        bw = max(1, -(-W // self.blocks))
        ph, pw = bh * self.blocks, bw * self.blocks
        padded = np.zeros((ph, pw), dtype=bool)
        padded[:H, :W] = ever_changed
        cells = padded.reshape(self.blocks, bh, self.blocks, bw).transpose(0, 2, 1, 3)
        return cells.reshape(self.blocks * self.blocks, bh * bw).any(axis=1).reshape(self.blocks, self.blocks)

    def score_actions(self, valid: Sequence[str], grid=None) -> Dict[str, float]:
        """动作谱评分 ∈ [0, ~1.7]：
        0.30·未尝试 + 0.25·预测进度 + 0.20·历史杠杆 + 0.15·新奇奖励 − 重复惩罚。

        世界新奇度（SSA-E3 决策入口）**只奖不罚**：效应签名落在
        「本局 ∪ 世界先验」行空间之外（真新机制）才加分；
        已解释动作（如移动）回到中性 —— 解释度高 ≠ 无生产力，不可减分。
        """
        scores: Dict[str, float] = {}
        try:
            s_sum = getattr(self, "_ctx_summary", None)
            for a in valid:
                act = str(a).upper()
                cnt = self._counts.get(act, 0)
                untried = 1.0 if cnt == 0 else 1.0 / (1.0 + 0.25 * cnt)
                p_prog = 0.5
                if s_sum is not None:
                    onehot = [1.0 if x == act else 0.0 for x in self.eigen.actions]
                    z = np.concatenate([np.asarray(onehot), s_sum])[: self.z_dim]
                    if z.size == self.z_dim:
                        p_prog, _unc = self.sgd.predict(z)
                gain = self._gain_ema.get(act, 0.0)
                sig = self._action_sig.get(act)
                if sig is not None and float(np.linalg.norm(sig)) > 1e-9:
                    wnov_bonus = 0.15 * max(0.0, self.world_novelty(sig) - 0.5) * 2.0
                else:
                    wnov_bonus = 0.0  # 无历史：不加不减
                repeat = list(self._recent_actions).count(act)
                score = (
                    0.30 * untried
                    + 0.25 * max(0.0, p_prog - 0.5) * 2.0
                    + 0.20 * min(1.0, self.eigen.action_leverage(act))
                    + wnov_bonus
                    - 0.10 * repeat
                )
                scores[act] = float(score)
        except Exception:
            pass
        return scores

    def explore_gate(self) -> float:
        """探索门 = 对偶预算调度 × 世界新奇度加成。

        近期效应普遍新奇（发现新机制）→ 抬高门限维持探索；
        一切皆已辨识（对先验无残差）→ 收敛到利用。
        """
        try:
            if self._novelty_hist:
                self._recent_novelty = float(
                    sum(self._novelty_hist) / len(self._novelty_hist)
                )
            base = 0.5 + 0.5 * float(min(1.0, max(0.0, self._recent_novelty)))
            return float(self.scheduler.explore_gate(base=base))
        except Exception:
            return 0.5

    def note_action_taken(self, action: str) -> None:
        try:
            self._recent_actions.append(str(action or "").upper())
            self.scheduler.step(False)
        except Exception:
            pass

    # ---------- 诊断 ----------

    def maintenance(self) -> Dict[str, Any]:
        """低频维护（每 ~16 步由控制器调用）：周期检测 + 谱缓存刷新。"""
        out: Dict[str, Any] = {}
        try:
            p = detect_period(list(self._change_counts), min_p=2, max_p=32)
            if p is not None:
                self._period = p
            self.sub.fit()
            self.eigen.eigen_decompose()
            self._prior_Q = combined_row_space(
                [self.sub.row_basis(),
                 self.shared_basis.V if self.shared_basis is not None else None]
            )
        except Exception:
            pass
        out["period"] = self._period
        out["rank"] = self.sub.effective_rank
        return out

    def snapshot(self) -> Dict[str, Any]:
        try:
            sb = self.shared_basis
            return {
                "step": self._step,
                "rank": self.sub.effective_rank,
                "n_samples": self.sub.n_samples,
                "period": self._period,
                "novel_center": round(self.sub.novelty(np.ones(min(16, self.dim))), 3),
                "gains": {k: round(v, 3) for k, v in sorted(self._gain_ema.items())},
                "dual_pressure": round(self.scheduler.pressure, 3),
                "explore_gate": round(self.explore_gate(), 3),
                "recent_novelty": round(self._recent_novelty, 3),
                "sgd_n": self.sgd.n,
                "eigen": bool(self.eigen.eigen_decompose()),
                "prior_k": int(sb.V.shape[0]) if sb is not None and sb.ready else 0,
                "prior_games": int(sb.n_games) if sb is not None else 0,
            }
        except Exception:
            return {"step": self._step}
