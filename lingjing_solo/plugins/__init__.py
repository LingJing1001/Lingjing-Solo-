"""Game plugins under the Solo general skeleton.

Architecture:
  MyAgent / LingjingSoloAgent  (general skeleton)
      ├─ PluginRegistry
      │     ├─ CannedRoutePlugin(ls20)
      │     └─ CannedRoutePlugin(ar25)
      └─ Solo coverage bootstrap (first-level rush)

Plugins never replace the skeleton; they only short-circuit when they
have a confident next action for the current game/level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class PluginDecision:
    action: str
    rationale: str
    plugin_id: str
    remaining: int = 0


class GamePlugin(Protocol):
    plugin_id: str
    game_ids: tuple[str, ...]

    def reset(self, game_id: str) -> None: ...

    def next_action(
        self,
        game_id: str,
        levels_completed: int,
        valid_actions: list[str] | None = None,
    ) -> Optional[PluginDecision]: ...


@dataclass
class CannedRoutePlugin:
    """Per-game canned ACTION sequences (ls20 / ar25, …)."""

    plugin_id: str
    game_ids: tuple[str, ...]
    levels: dict[str, list[str]]
    _gid: str = ""
    _level: int = -1
    _queue: list[str] = field(default_factory=list)
    _idx: int = 0

    def reset(self, game_id: str) -> None:
        self._gid = _norm_game(game_id)
        self._level = -1
        self._queue = []
        self._idx = 0

    def _load(self, levels_completed: int) -> bool:
        lv = int(levels_completed)
        if self._gid not in self.game_ids:
            return False
        if self._level != lv:
            self._queue = list(self.levels.get(str(lv)) or [])
            self._idx = 0
            self._level = lv
        return self._idx < len(self._queue)

    def next_action(
        self,
        game_id: str,
        levels_completed: int,
        valid_actions: list[str] | None = None,
    ) -> Optional[PluginDecision]:
        del valid_actions  # canned plans are absolute; do not desync by filtering
        gid = _norm_game(game_id)
        if gid != self._gid:
            self.reset(gid)
        if gid not in self.game_ids:
            return None
        if not self._load(levels_completed):
            return None
        if self._idx >= len(self._queue):
            return None
        act = self._queue[self._idx]
        self._idx += 1
        rem = max(0, len(self._queue) - self._idx)
        return PluginDecision(
            action=act,
            rationale=f"plugin:{self.plugin_id}:L{levels_completed}:rem={rem}",
            plugin_id=self.plugin_id,
            remaining=rem,
        )


def _norm_game(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    return s.split("-")[0]


class PluginRegistry:
    """Ordered plugin lookup; first hit wins."""

    def __init__(self) -> None:
        self._plugins: list[GamePlugin] = []

    def register(self, plugin: GamePlugin) -> None:
        self._plugins.append(plugin)

    def reset(self, game_id: str) -> None:
        for p in self._plugins:
            p.reset(game_id)

    def choose(
        self,
        game_id: str,
        levels_completed: int,
        valid_actions: list[str] | None = None,
    ) -> Optional[PluginDecision]:
        gid = _norm_game(game_id)
        for p in self._plugins:
            if gid not in p.game_ids:
                continue
            dec = p.next_action(gid, levels_completed, valid_actions)
            if dec is not None:
                return dec
        return None

    def known_games(self) -> list[str]:
        out: list[str] = []
        for p in self._plugins:
            out.extend(p.game_ids)
        return sorted(set(out))


__all__ = [
    "PluginDecision",
    "GamePlugin",
    "CannedRoutePlugin",
    "PluginRegistry",
    "_norm_game",
]
