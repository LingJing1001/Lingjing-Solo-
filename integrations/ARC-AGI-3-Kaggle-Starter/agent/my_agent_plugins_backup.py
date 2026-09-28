"""Submission agent — general Solo skeleton + game plugins.

Decision order (P2 architecture):
  1) PluginRegistry (ls20/ar25 canned)  — 辅线：已摸清游戏
  2) CoverageBootstrap                  — 主线：无插件局先抠 ≥1 关
  3) LingjingSoloAgent                  — 通用骨架

BUILD_TAG documents P0 score target (~8 if ls20+ar25 bind on Phase B).
Class name MUST remain `MyAgent`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

_FILE = Path(__file__).resolve()


def _bootstrap_paths() -> None:
    candidates: list[Path] = []
    if _FILE.parent.name == "agent":
        candidates.append(_FILE.parents[1])
    if _FILE.parent.name == "templates":
        candidates.append(_FILE.parents[2])
    candidates.extend([_FILE.parents[1], Path.cwd()])
    seen: set[str] = set()
    for c in candidates:
        try:
            if (c / "lingjing_solo").is_dir():
                s = str(c)
                if s not in seen:
                    seen.add(s)
                    sys.path.insert(0, s)
        except OSError:
            continue


_bootstrap_paths()

from lingjing_solo import SoloConfig  # noqa: E402
from lingjing_solo.agent import LingjingSoloAgent  # noqa: E402
from lingjing_solo.plugins.builtin_canned import build_default_registry  # noqa: E402
from lingjing_solo.plugins.coverage import CoverageBootstrap  # noqa: E402
from lingjing_solo.plugins import _norm_game  # noqa: E402

BUILD_TAG = "p0p1p2-skel+plugins-ls20x7+ar25x8"
AGENT_BRAND = "Lingjing EtherealRealm-Solo"
TEAM_SLUG = "lingjing-skel-v8"
# P0 expect ≈8 if Phase B binds; P1 = raise non-zero games via coverage.


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
            out.append(str(a.name).upper())
        else:
            out.append(_canon(str(a)))
    # Empty → do not filter plugins (avoid false rejects)
    return out


class MyAgent(Agent):
    """General skeleton + plugins. See module docstring."""

    MAX_ACTIONS: int = 800

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.plugins = build_default_registry()
        self.coverage = CoverageBootstrap(max_probe_steps=120, click_grid=6)
        self.plugins.reset(_norm_game(getattr(self, "game_id", "")))
        self.brain = None
        try:
            self.brain = LingjingSoloAgent(
                cfg=SoloConfig(
                    use_llm_advisor=False,
                    llm_calls_per_game=0,
                    enable_undo=False,
                    enable_mouse=True,
                    enable_click_sweep=True,
                    return_game_action=False,
                    human_baseline_estimate=120,
                    hard_step_multiplier=6.0,
                    lightweight_search_depth=10,
                    loop_detect_window=5,
                    click_stall_threshold=3,
                    click_warmup=10,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{AGENT_BRAND}] Solo init failed: {exc}", flush=True)

        gid = _norm_game(getattr(self, "game_id", ""))
        print(
            f"[{AGENT_BRAND}/{BUILD_TAG}] game={self.game_id} gid={gid} "
            f"plugins={self.plugins.known_games()} "
            f"max_actions={self.MAX_ACTIONS} "
            f"P0_expect≈8 P1_coverage_on",
            flush=True,
        )

    @property
    def name(self) -> str:
        return f"{super().name}.{TEAM_SLUG}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return self.action_counter >= self.MAX_ACTIONS

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        gid = _norm_game(
            getattr(self, "game_id", "") or getattr(latest_frame, "game_id", "")
        )

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.plugins.reset(gid)
            # Keep coverage memory across GAME_OVER so probes diversify
            if latest_frame.state is GameState.NOT_PLAYED:
                self.coverage.reset()
            if self.brain is not None:
                try:
                    self.brain.reset()
                except Exception:  # noqa: BLE001
                    pass
            action = GameAction.RESET
            action.reasoning = f"{BUILD_TAG}:reset gid={gid}"
            print(f"[{BUILD_TAG}] RESET gid={gid}", flush=True)
            return action

        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
        self.coverage.note_progress(levels)
        valid = _valid_names(latest_frame)

        # 1) Plugin short-circuit (ls20 / ar25)
        dec = self.plugins.choose(gid, levels, valid)
        if dec is not None:
            action = _as_game_action(dec.action)
            action.reasoning = f"{BUILD_TAG}:{dec.rationale}"
            if (
                self.action_counter < 2
                or dec.remaining == 0
                or levels != getattr(self, "_last_log_level", -1)
            ):
                print(
                    f"[{BUILD_TAG}] plugin step={self.action_counter} "
                    f"L={levels} {action.name} rem={dec.remaining} gid={gid}",
                    flush=True,
                )
                self._last_log_level = levels
            return action

        # 2) Coverage bootstrap — first-level rush for unknown games
        cov = self.coverage.suggest(valid, levels)
        if cov is not None:
            act_name, xy, why = cov
            action = _as_game_action(act_name)
            if xy is not None and action.is_complex():
                action.set_data({"x": int(xy[0]), "y": int(xy[1])})
            action.reasoning = f"{BUILD_TAG}:{why}"
            if self.action_counter < 3 or self.action_counter % 16 == 0:
                print(
                    f"[{BUILD_TAG}] coverage step={self.action_counter} "
                    f"L={levels} {action.name} gid={gid}",
                    flush=True,
                )
            return action

        # 3) General Solo skeleton
        if self.brain is not None:
            try:
                raw = self.brain.choose_action(frames, latest_frame)
                action = _as_game_action(raw)
            except Exception as exc:  # noqa: BLE001
                print(f"[{BUILD_TAG}] solo error: {exc}", flush=True)
                action = GameAction.ACTION1
        else:
            action = GameAction.ACTION1

        if action.is_complex() and not getattr(action, "data", None):
            action.set_data({"x": 32, "y": 32})
        action.reasoning = f"{BUILD_TAG}:solo L{levels} {action.name}"
        return action
