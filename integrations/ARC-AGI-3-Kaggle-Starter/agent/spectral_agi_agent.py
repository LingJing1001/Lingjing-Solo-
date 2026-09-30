"""SSA spectral agent — 谱-子空间认知架构的未知能力主入口（无插件、无罐头脚本）。

决策链：
  1) RESET when required
  2) SpectralCeaxController = CEAX 假设-实验闭环 × SpectralMind 谱认知内核
     （四子空间世界模型 · 惊讶/可控/进度三信号 · 特征选项 · 对偶探索预算 ·
       周期机制检测 · 跨游戏低秩矩阵记忆）
  3) 永不 ScriptBank / PluginRegistry / canned solvers

理论：docs/SSA_谱子空间认知架构_理论白皮书_v1.md
数学：Strang《线性代数（原书第6版）》Ch1–Ch10（lincore 模块逐一映射）
上游：ceax_unknown_agent.py（同一决策契约，谱层只加不改）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

_FILE = Path(__file__).resolve()
ROOT = _FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingjing_solo.core import SoloConfig, extract_grid, hash_grid
from lingjing_solo.perception import PerceptionEncoder
from lingjing_solo.transfer import SpectralCeaxController
from lingjing_solo.lincore.hypothesis_lab import extract_suggestions_json
from lingjing_solo.plugins import _norm_game

BUILD_TAG = "ssa-spectral-v1"

# Short-L1 first (human baseline L1 steps)
PRIORITY_GAMES = (
    "lp85", "vc33", "sb26", "tu93", "s5i5", "r11l", "su15", "bp35",
    "tn36", "cd82", "ft09", "sc25",
)

# SSA-E2 LLM 顾问：SSA_LLM_CONSULT=1 时在卡点向 LLM 征集假设（默认关闭，Kaggle 离线安全）
LLM_CONSULT_STEPS = (60, 240)
LLM_CONSULT_MAX = 2


def _canon(name: str) -> str:
    return str(name or "ACTION1").strip().upper()


def _as_game_action(action: Any) -> GameAction:
    if isinstance(action, GameAction):
        return action
    name = _canon(str(action or "ACTION1"))
    if hasattr(GameAction, name):
        return getattr(GameAction, name)
    mapping = {
        "RESET": 0,
        "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
        "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
    }
    return GameAction.from_id(mapping.get(name, 1))


def _valid_names(frame: FrameData) -> list[str]:
    raw = getattr(frame, "available_actions", None) or []
    out: list[str] = []
    for a in raw:
        if hasattr(a, "name"):
            name = str(a.name).upper()
        elif hasattr(a, "id"):
            name = f"ACTION{int(a.id)}"
        else:
            name = _canon(str(a))
        if name.isdigit():
            name = f"ACTION{int(name)}"
        out.append(name)
    return out


class MyAgent(Agent):
    """SSA spectral unknown-capability agent."""

    MAX_ACTIONS: int = 800

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = SoloConfig(
            use_llm_advisor=False,
            enable_transfer=True,
            enable_mouse=True,
            enable_click_sweep=True,
            return_game_action=False,
        )
        self.encoder = PerceptionEncoder(self.cfg)
        gid = _norm_game(getattr(self, "game_id", ""))
        self.ceax = SpectralCeaxController(self.cfg, game_sig=gid or "unknown")
        self._prev_grid: Optional[np.ndarray] = None
        self._last_was_click = False
        self._levels_seen = 0
        self._llm_consults = 0
        print(
            f"[{BUILD_TAG}] game={self.game_id} gid={gid} "
            f"priority={gid in PRIORITY_GAMES} max_actions={self.MAX_ACTIONS} "
            f"core=lincore(4-subspace+eigenoptions+dual-budget)",
            flush=True,
        )

    @property
    def name(self) -> str:
        return f"{super().name}.ssa-spectral.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return self.action_counter >= self.MAX_ACTIONS

    # ---------- SSA-E2：LLM 假设顾问（卡点有限次征集，默认关闭） ----------

    def _maybe_consult_llm(self, grid, levels: int, valid: list[str]) -> None:
        if os.environ.get("SSA_LLM_CONSULT", "0") != "1":
            return
        if self._llm_consults >= LLM_CONSULT_MAX:
            return
        if self.action_counter not in LLM_CONSULT_STEPS:
            return
        if levels != self._levels_seen:
            return  # 仍有进展，不打扰
        try:
            from lingjing_solo.llm_client import chat_completion

            snap = self.ceax.snapshot()
            objs = [
                {"color": t["color"], "area": t["area"], "xy": list(t["xy"])}
                for t in (snap.get("best_targets") or [])[:8]
            ]
            prompt = (
                "你是 ARC-AGI-3 交互游戏的策略顾问。"
                f"游戏 {getattr(self, 'game_id', '?')} 第 {levels} 关，"
                f"已 {self.action_counter} 步无进展。\n"
                f"可用动作: {valid}\n"
                f"屏幕显著目标(颜色/面积/中心xy): {objs}\n"
                "请猜测最可能推动关卡进化的 2-3 个假设，只输出 JSON 数组：\n"
                '[{"action":"ACTION6","x":32,"y":12,"expect":"点击后门变色","confidence":0.7}]'
            )
            text = chat_completion(prompt, timeout=30.0, max_tokens=400)
            payload = extract_suggestions_json(text)
            if payload is None:
                return
            n = self.ceax.register_hypotheses(payload)
            self._llm_consults += 1
            print(f"[{BUILD_TAG}] llm-consult#{self._llm_consults} -> {n} hypotheses", flush=True)
        except Exception as exc:  # noqa: BLE001 — LLM 故障绝不影响决策链
            print(f"[{BUILD_TAG}] llm-consult skipped: {type(exc).__name__}", flush=True)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        gid = _norm_game(
            getattr(self, "game_id", "") or getattr(latest_frame, "game_id", "")
        )

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.NOT_PLAYED:
                self.ceax.reset_game()
                self._prev_grid = None
                self._levels_seen = 0
            action = GameAction.RESET
            action.reasoning = f"{BUILD_TAG}:reset"
            return action

        grid = extract_grid(latest_frame)
        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
        valid = _valid_names(latest_frame)

        # Observe previous action outcome
        delta_px = 0
        if self._prev_grid is not None and grid is not None:
            delta_px = int((self._prev_grid != grid).sum())
        progressed = levels > self._levels_seen
        ghash = hash_grid(grid) if grid is not None else ""
        if self.action_counter > 0:
            self.ceax.observe_outcome(
                delta_pixels=delta_px,
                progressed=progressed,
                grid_hash=ghash,
                levels=levels,
                prev_grid=self._prev_grid,
                grid=grid,
            )
        if levels > self._levels_seen:
            self._levels_seen = levels

        # Full-frame objects for CEAX click experiments (not delta-only)
        objects = []
        if grid is not None:
            try:
                objects = self.encoder.segment(grid, delta_pixels=None)
            except Exception:
                objects = []

        # SSA-E2：卡点时向 LLM 征集假设（默认关闭；故障安全）
        try:
            self._maybe_consult_llm(grid, levels, valid)
        except Exception:
            pass

        act_name, xy, why = self.ceax.choose(
            valid_actions=valid or ["ACTION1"],
            objects=objects,
            grid=grid,
            grid_hash=ghash,
        )
        action = _as_game_action(act_name)
        self._last_was_click = act_name == "ACTION6"
        if action.is_complex():
            if xy is not None:
                action.set_data({"x": int(xy[0]), "y": int(xy[1])})
            else:
                action.set_data({"x": 32, "y": 32})

        if grid is not None:
            self._prev_grid = grid.copy()

        action.reasoning = f"{BUILD_TAG}:{why} L{levels}"
        if self.action_counter < 3 or self.action_counter % 25 == 0 or progressed:
            snap = self.ceax.snapshot()
            spec = snap.get("spectral") or {}
            print(
                f"[{BUILD_TAG}] step={self.action_counter} L={levels} "
                f"{act_name} {why} pc={snap.get('kb_pc')} g={snap.get('kb_goal')} "
                f"corr={snap.get('kb_corr')} blocked={snap.get('kb_blocked')} "
                f"skills={snap['skills']} "
                f"rank={spec.get('rank')} period={spec.get('period')} "
                f"dual={spec.get('dual_pressure')}",
                flush=True,
            )
        return action
