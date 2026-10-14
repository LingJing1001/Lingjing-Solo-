"""关卡无关的「动作效果画像」：把 action_diff 的差分能力推向所有关卡。

背景 —— 仓库里已有四条"动作效果"信号，强弱差三档
------------------------------------------------
1. ``perception/encoder.compute_delta``              只回答"哪些像素变了"（坐标列表）
2. ``world_model/field._calibrate_effect``           把变化量折成 ``delta/128`` 的标量 EMA
3. ``planning/discrete_nav._centroid_delta``         拿**变化像素**的质心当角色位移（近似），
                                                     且只在离散方向键关卡启用
4. ``exploration/action_diff.analyze_observation``   用**指定颜色**的质心求真实位移方向，
                                                     外加过关触发与多采样一致性

第 4 条最准，但它此前**只被离线回放脚本 ``evidence/ls20.py`` 调用**。在线链路上，
25 局全部只用 1/2 两条弱信号 —— 连"这一步把角色推向哪个方向"都无从得知，
更别说"这个动作每次效果都一样吗"。这就是"动作差分只服务固定关卡"的实际后果。

本模块把第 4 条做成**关卡无关、在线可用**的组件：任何关卡、任何动作（含 ACTION6），
每步记一条 ``ActionObservation``，聚合成可查询的效果画像，供决策层使用。

设计约束
--------
* **单关内聚**：位移一致性只在同一关内有意义 → 过关时清空单关样本。
* **跨关先验**：按键语义由 harness 固定，"无效键"往往跨关通用，所以另存一份
  跨关累计，供新关冷启动时避开已验证无效的动作。
* **绝不抛出**：在线链路任何一步异常都会毁掉整局。空帧/形状不匹配一律静默跳过，
  只计入 ``failures`` 计数（可观测，不致命）。
* **口径对齐**：复用 ``action_diff`` 的差分实现，方向阈值与
  ``discrete_nav.axis_actions`` 保持一致，保证两条链路对同一观测得出同一结论。
* **修正上游语义**：``ActionDelta.triggered_level_change`` 定义成
  ``levels_completed_after > 0``，在"累计关卡数"语义下几乎恒真 —— 离线 ls20 回放里
  ``level_trigger_samples`` 因此 ≈ 样本数，是个死指标。本模块改由调用方显式传入
  ``progressed``（"本次动作是否让关卡 +1"），只用这个更严的口径。
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .action_diff import ActionDelta, ActionObservation, analyze_observation

__all__ = [
    "ActionEffect",
    "ActionEffectModel",
    "decision_effects_enabled",
    "EFFECT_MIN_MOVE",
    "EFFECT_MAX_SAMPLES",
    "EFFECT_NOOP_MIN_SAMPLES",
]

#: 位移方向判定阈值（像素）。与 ``planning/discrete_nav.py::axis_actions`` 一致，
#: 否则同一条观测在两条链路上会给出不同的"方向"。
EFFECT_MIN_MOVE = 0.3

#: 每动作保留的最大样本数（滚动窗口）。48 ≈ 一局 400 步的八分之一，
#: 足够估一致性，又不会被开局噪声长期污染。
EFFECT_MAX_SAMPLES = 48

#: 判定"无效动作"所需的最少样本。样本太少时"0 变化"可能只是偶然
#: （例如点击了空白格、或走到了边界墙），不足以断定动作本身是死的。
EFFECT_NOOP_MIN_SAMPLES = 3


@dataclass(frozen=True)
class ActionEffect:
    """一个动作的效果画像（只读快照）。"""

    action: str
    samples: int
    changed_pixels_mean: float
    #: 只有**多次采样位移完全一致**时才给出；否则为 None（不可靠）。
    displacement: Optional[tuple[float, float]]
    consistent: bool
    confidence: float
    level_trigger_samples: int
    no_effect_samples: int
    #: "up"/"down"/"left"/"right"；无有效位移时为空串。
    direction: str

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "samples": self.samples,
            "changed_pixels_mean": self.changed_pixels_mean,
            "displacement": self.displacement,
            "consistent": self.consistent,
            "confidence": self.confidence,
            "level_trigger_samples": self.level_trigger_samples,
            "no_effect_samples": self.no_effect_samples,
            "direction": self.direction,
        }


def _direction(displacement: Optional[tuple[float, float]]) -> str:
    """把位移向量翻成方向名。阈值/优先级与 discrete_nav.axis_actions 一致。"""
    if displacement is None:
        return ""
    dx, dy = float(displacement[0]), float(displacement[1])
    if abs(dy) >= abs(dx) and abs(dy) > EFFECT_MIN_MOVE:
        return "up" if dy < 0 else "down"
    if abs(dx) > EFFECT_MIN_MOVE:
        return "left" if dx < 0 else "right"
    return ""


def decision_effects_enabled() -> bool:
    """效果画像是否参与动作排序。

    默认 **关**：不设 ``LINGJING_ACTION_EFFECT`` 时，链路行为与接入前逐位一致
    （与 ``AOP`` 的接入规矩相同）。想启用就设成 ``on``/``1``/``true``。

    注意：**观测本身不受此开关影响** —— 画像始终在累积，只是不参与决策。
    这样"先收集、后启用"不需要改代码。
    """
    raw = (os.environ.get("LINGJING_ACTION_EFFECT") or "").strip().lower()
    return raw not in ("", "0", "off", "false", "no")


class ActionEffectModel:
    """跨关卡在线累积「动作 → 效果」画像。

    用法（agent 主循环，**所有关卡**都一样）::

        self.action_effect.observe(
            self._last_action, prev_grid, curr_grid,
            progressed=curr.levels_completed > self._levels_seen,
            level=curr.levels_completed,
        )

    画像分两个作用域：
      * ``local()``   —— 当前关卡内的一致性（过关即清空）
      * ``overall()`` —— 跨关累计先验（滚动窗口，不清空）
    """

    def __init__(
        self,
        cfg=None,
        logger=None,
        player_color: int = 1,
        max_samples: int = EFFECT_MAX_SAMPLES,
    ):
        self.cfg = cfg
        self.log = logger
        #: 角色颜色。ACTION 游戏里玩家几乎恒为 1（与 action_diff/离线 ls20 一致）。
        #: 设 0 表示"不做颜色假设"，退回变化像素质心（近似 discrete_nav 口径）。
        self.player_color = int(player_color)
        self.max_samples = int(max_samples)
        self.level = 0
        self.observations = 0
        self.failures = 0
        self._cur: dict[str, list[ActionDelta]] = defaultdict(list)
        self._all: dict[str, list[ActionDelta]] = defaultdict(list)
        self._progress: dict[str, int] = defaultdict(int)

    # ---------- 写入 ----------
    def observe(
        self,
        action: Optional[str],
        before_grid,
        after_grid,
        *,
        progressed: bool = False,
        level: Optional[int] = None,
    ) -> Optional[ActionDelta]:
        """记一条 (动作, 前帧, 后帧) 观测。**任何异常都吞掉**，绝不上抛。

        返回算出的 ``ActionDelta``（成功时），供调用方直接取用，避免二次计算。
        """
        if not action or before_grid is None or after_grid is None:
            return None
        try:
            sample = ActionObservation(
                action=str(action),
                before=np.asarray(before_grid),
                after=np.asarray(after_grid),
                # 只编码"本次是否过关"：上游把该字段当作"累计关卡数 > 0"来判触发，
                # 那在在线语义下恒真，所以这里传 0/1 收紧语义。
                levels_completed_after=1 if progressed else 0,
            )
            delta = analyze_observation(sample, player_color=self.player_color)
        except Exception:
            # 形状不匹配、空帧、动画栈层数不一致…… 统计模块不该拖垮决策链路。
            self.failures += 1
            return None

        self.observations += 1
        key = str(action)
        for store in (self._cur, self._all):
            bucket = store[key]
            bucket.append(delta)
            if len(bucket) > self.max_samples:
                del bucket[: len(bucket) - self.max_samples]
        if progressed:
            self._progress[key] += 1
        if level is not None and int(level) > self.level:
            self.level_up(int(level))
        return delta

    def level_up(self, level: int) -> None:
        """进入新关卡：清空单关一致性样本，保留跨关先验。"""
        if level <= self.level:
            return
        self.level = int(level)
        self._cur.clear()

    # ---------- 聚合 ----------
    def _aggregate(
        self, store: dict[str, list[ActionDelta]], *, with_progress: bool
    ) -> dict[str, ActionEffect]:
        out: dict[str, ActionEffect] = {}
        for action, deltas in store.items():
            n = len(deltas)
            if n == 0:
                continue
            displacements = [d.displacement for d in deltas if d.displacement is not None]
            unique = set(displacements)
            consistent = bool(displacements) and len(unique) == 1
            disp = displacements[0] if consistent else None
            moved = sum(1 for d in deltas if d.displacement not in (None, (0.0, 0.0)))
            # 与 action_diff.summarize_actions 同式：位移一致才算满置信，
            # 否则减半（方向不可靠但"会动"仍是有用信息）。
            confidence = round((moved / n) * (1.0 if consistent else 0.5), 3)
            out[action] = ActionEffect(
                action=action,
                samples=n,
                changed_pixels_mean=round(sum(d.changed_pixels for d in deltas) / n, 3),
                displacement=disp,
                consistent=consistent,
                confidence=confidence,
                level_trigger_samples=int(self._progress.get(action, 0)) if with_progress else 0,
                no_effect_samples=sum(1 for d in deltas if d.changed_pixels == 0),
                direction=_direction(disp),
            )
        return out

    def local(self) -> dict[str, ActionEffect]:
        """当前关卡内的效果画像（过关后为空）。"""
        return self._aggregate(self._cur, with_progress=False)

    def overall(self) -> dict[str, ActionEffect]:
        """跨关累计的效果画像。"""
        return self._aggregate(self._all, with_progress=True)

    def effect(self, action: str, *, scope: str = "overall") -> Optional[ActionEffect]:
        table = self.overall() if scope == "overall" else self.local()
        return table.get(str(action))

    # ---------- 查询（决策层消费的入口）----------
    def no_effect_actions(
        self,
        *,
        min_samples: int = EFFECT_NOOP_MIN_SAMPLES,
        scope: str = "overall",
    ) -> set[str]:
        """返回"执行过多次、但画面**每一次**都没变"的动作。

        这是"哪些动作是死的"的最小判定：样本数达标 + 全部零变化 + 从未触发过关。
        混杂样本（有时有效）**不在此列** —— 那多半是位置相关（撞墙 vs 走动），
        不是动作本身无效。
        """
        table = self.overall() if scope == "overall" else self.local()
        return {
            action
            for action, effect in table.items()
            if effect.samples >= min_samples
            and effect.no_effect_samples == effect.samples
            and effect.level_trigger_samples == 0
        }

    def directional_actions(self, *, scope: str = "overall") -> dict[str, str]:
        """方向 → 动作名（只收位移一致的动作）。与 discrete_nav.axis_actions 同形。"""
        out: dict[str, str] = {}
        table = self.overall() if scope == "overall" else self.local()
        for action, effect in table.items():
            if effect.direction and effect.direction not in out:
                out[effect.direction] = action
        return out

    def useful_actions(self, *, min_samples: int = 2, scope: str = "overall") -> set[str]:
        """返回"确实会改变画面"的动作（样本足够且至少半数有位移/过关）。"""
        table = self.overall() if scope == "overall" else self.local()
        out: set[str] = set()
        for action, effect in table.items():
            if effect.level_trigger_samples > 0:
                out.add(action)
                continue
            if effect.samples >= min_samples and effect.no_effect_samples < effect.samples:
                out.add(action)
        return out

    # ---------- 观测 ----------
    def summary(self, *, top: int = 6) -> dict:
        """供日志/证据落盘的紧凑摘要。"""
        overall = self.overall()
        ranked = sorted(
            overall.values(),
            key=lambda e: (e.level_trigger_samples, e.changed_pixels_mean),
            reverse=True,
        )
        return {
            "observations": self.observations,
            "failures": self.failures,
            "level": self.level,
            "actions_seen": len(overall),
            "no_effect": sorted(self.no_effect_actions()),
            "directional": self.directional_actions(),
            "top": [e.as_dict() for e in ranked[:top]],
        }
