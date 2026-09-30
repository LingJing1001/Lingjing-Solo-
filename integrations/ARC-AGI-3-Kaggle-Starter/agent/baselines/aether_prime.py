"""Aether-Prime — ARC-AGI-3 agent with world model + planner (+ optional LLM).

Self-contained for Kaggle (stdlib + arcengine + agents.agent only).
When OPENAI_API_KEY is set (local/online), a compact LLM advisor can override
the planner on hard decisions. On Kaggle (no net / no key) the symbolic stack
runs alone.

Pillars:
  1. Perception   — grid hash, blobs, change loci, salient clicks
  2. WorldModel   — action effects, transition graph, mover axes, goals
  3. Planner      — (state,action) novelty, spatial seek, recipe replay, BFS
  4. LLM (opt)    — short JSON action advice with compact grid summary
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


# ═══════════════════════════════════════════════════════════════════════════
# Perception
# ═══════════════════════════════════════════════════════════════════════════

def layers_of(frame: FrameData) -> list[list[list[int]]]:
    raw = frame.frame or []
    if not raw:
        return []
    first = raw[0]
    if first and isinstance(first[0], list):
        return raw  # type: ignore[return-value]
    return [raw]  # type: ignore[list-item]


def top_grid(frame: FrameData) -> list[list[int]]:
    ls = layers_of(frame)
    return ls[-1] if ls else []


def grid_hash(frame: FrameData) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(int(frame.levels_completed or 0)).encode())
    for layer in layers_of(frame):
        for row in layer:
            h.update(bytes(int(c) & 0xFF for c in row))
            h.update(b"|")
        h.update(b"||")
    return h.hexdigest()


def guess_bg(grid: list[list[int]]) -> int:
    if not grid:
        return 0
    return Counter(c for row in grid for c in row).most_common(1)[0][0]


def bbox(grid: list[list[int]], bg: int) -> tuple[int, int, int, int]:
    h, w = len(grid), len(grid[0]) if grid else 0
    xs, ys = [], []
    for y in range(h):
        for x in range(w):
            if grid[y][x] != bg:
                xs.append(x)
                ys.append(y)
    if not xs:
        return 0, 0, max(0, w - 1), max(0, h - 1)
    return min(xs), min(ys), max(xs), max(ys)


def changed_cells(
    a: list[list[int]], b: list[list[int]]
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for y in range(min(len(a), len(b))):
        for x in range(min(len(a[y]), len(b[y]))):
            if a[y][x] != b[y][x]:
                out.append((x, y, a[y][x], b[y][x]))
    return out


def change_centroid(
    a: list[list[int]], b: list[list[int]]
) -> Optional[tuple[float, float, int]]:
    ch = changed_cells(a, b)
    if not ch:
        return None
    cx = sum(t[0] for t in ch) / len(ch)
    cy = sum(t[1] for t in ch) / len(ch)
    return cx, cy, len(ch)


def connected_blobs(
    grid: list[list[int]], bg: int, max_blobs: int = 40
) -> list[dict[str, Any]]:
    """4-connected components excluding background."""
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    blobs: list[dict[str, Any]] = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0] or grid[y0][x0] == bg:
                continue
            color = grid[y0][x0]
            stack = [(x0, y0)]
            seen[y0][x0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx]:
                        if grid[ny][nx] == color:
                            seen[ny][nx] = True
                            stack.append((nx, ny))
            if not cells:
                continue
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            blobs.append(
                {
                    "color": color,
                    "size": len(cells),
                    "cx": sum(xs) / len(xs),
                    "cy": sum(ys) / len(ys),
                    "x0": min(xs),
                    "y0": min(ys),
                    "x1": max(xs),
                    "y1": max(ys),
                }
            )
            if len(blobs) >= max_blobs:
                return blobs
    blobs.sort(key=lambda b: b["size"])
    return blobs


def salient_clicks(frame: FrameData, limit: int = 36) -> list[tuple[int, int]]:
    grid = top_grid(frame)
    if not grid:
        return [(32, 32)]
    h, w = len(grid), len(grid[0])
    bg = guess_bg(grid)
    hist = Counter(c for row in grid for c in row)
    rare = [c for c, _ in sorted(hist.items(), key=lambda kv: kv[1]) if c != bg]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(x: int, y: int) -> None:
        x = max(0, min(w - 1, min(63, int(x))))
        y = max(0, min(h - 1, min(63, int(y))))
        if (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))

    for color in rare[:8]:
        coords = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == color]
        step = max(1, len(coords) // 6)
        for i in range(0, len(coords), step):
            add(*coords[i])
            if len(out) >= limit:
                return out

    for bl in connected_blobs(grid, bg):
        add(bl["cx"], bl["cy"])
        if len(out) >= limit:
            return out

    x0, y0, x1, y1 = bbox(grid, bg)
    for p in (
        (x0, y0),
        (x1, y0),
        (x0, y1),
        (x1, y1),
        ((x0 + x1) // 2, (y0 + y1) // 2),
        (w // 2, h // 2),
    ):
        add(*p)
    return out[:limit] or [(w // 2, h // 2)]


def compact_grid_text(frame: FrameData, max_side: int = 16) -> str:
    """Downsample grid to a short text for LLM context."""
    grid = top_grid(frame)
    if not grid:
        return "(empty)"
    h, w = len(grid), len(grid[0])
    step_y = max(1, h // max_side)
    step_x = max(1, w // max_side)
    lines = []
    for y in range(0, h, step_y):
        row = []
        for x in range(0, w, step_x):
            row.append(f"{grid[y][x]:X}")
        lines.append("".join(row))
    return f"{h}x{w} step={step_x},{step_y}\n" + "\n".join(lines)


def resolve_actions(available: Optional[list[Any]]) -> list[GameAction]:
    if not available:
        return [a for a in GameAction if a is not GameAction.RESET]
    out: list[GameAction] = []
    for a in available:
        if isinstance(a, GameAction):
            if a is not GameAction.RESET:
                out.append(a)
            continue
        try:
            wanted = int(a)
            matched = next((m for m in GameAction if m.value == wanted), None)
            if matched is not None and matched is not GameAction.RESET:
                out.append(matched)
        except (TypeError, ValueError):
            try:
                act = GameAction.from_name(str(a).upper())
                if act is not GameAction.RESET:
                    out.append(act)
            except Exception:
                continue
    seen: set[str] = set()
    uniq: list[GameAction] = []
    for a in out:
        if a.name not in seen:
            seen.add(a.name)
            uniq.append(a)
    return uniq or [a for a in GameAction if a is not GameAction.RESET]


# ═══════════════════════════════════════════════════════════════════════════
# World model
# ═══════════════════════════════════════════════════════════════════════════

class ActionKey:
    __slots__ = ("name", "x", "y")

    def __init__(self, name: str, x: Optional[int] = None, y: Optional[int] = None):
        self.name = name
        self.x = x
        self.y = y

    def fp(self) -> str:
        return self.name if self.x is None else f"{self.name}:{self.x},{self.y}"


class WorldModel:
    """Learned dynamics + goals for one game episode."""

    def __init__(self) -> None:
        self.visits: Counter = Counter()
        self.pair_visits: Counter = Counter()
        self.graph: dict[str, dict[str, str]] = defaultdict(dict)  # s -> a -> s'
        self.effects: dict[str, dict[str, float]] = defaultdict(
            lambda: {"noop": 0.0, "change": 0.0, "progress": 0.0, "trials": 0.0}
        )
        # Estimated move vector per simple action (dx, dy) in screen space
        self.move_vec: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        # action -> list of observed (dx,dy)
        self._move_samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.recent: deque = deque(maxlen=64)
        self.recipes: list[list[ActionKey]] = []
        self.best_levels: int = 0
        self.player_xy: Optional[tuple[float, float]] = None
        self.goals: list[tuple[float, float]] = []  # interesting targets
        self.tried_clicks: set[tuple[int, int]] = set()
        self.click_q: deque = deque()
        self.stall: int = 0
        self.last_hash: Optional[str] = None
        self.last_action: Optional[str] = None
        self.streak: int = 0
        self.steps_level: int = 0
        self.deaths: int = 0
        self.notes: list[str] = []  # short textual findings for LLM
        self.replaying: Optional[list[ActionKey]] = None
        self.recipe_i: int = 0
        self.patrol_i: int = 0
        self.seek_target: Optional[tuple[float, float]] = None
        self.seek_ttl: int = 0

    def record_transition(
        self,
        before_h: str,
        after_h: str,
        action: ActionKey,
        before_g: list[list[int]],
        after_g: list[list[int]],
        levels_before: int,
        levels_after: int,
    ) -> None:
        progressed = levels_after > levels_before
        ch = changed_cells(before_g, after_g) if before_g and after_g else []
        n_ch = len(ch)
        noop = before_h == after_h and not progressed
        st = self.effects[action.name]
        st["trials"] += 1.0
        if noop:
            st["noop"] += 1.0
        if n_ch:
            st["change"] += 1.0
        if progressed:
            st["progress"] += 1.0
            self.notes.append(f"PROGRESS via {action.fp()} L{levels_before}->{levels_after}")

        self.graph[before_h][action.name] = after_h
        self.recent.append((action, before_h, after_h, n_ch, progressed))

        cen = change_centroid(before_g, after_g) if before_g and after_g else None
        if cen and action.x is None:  # simple action
            cx, cy, n = cen
            if self.player_xy is not None:
                dx = cx - self.player_xy[0]
                dy = cy - self.player_xy[1]
                # Only trust moderate displacements as "movement"
                if 0.5 < (dx * dx + dy * dy) ** 0.5 < 20:
                    self._move_samples[action.name].append((dx, dy))
                    samples = self._move_samples[action.name][-30:]
                    self._move_samples[action.name] = samples
                    ax = sum(s[0] for s in samples) / len(samples)
                    ay = sum(s[1] for s in samples) / len(samples)
                    self.move_vec[action.name] = [ax, ay, float(len(samples))]
            self.player_xy = (cx, cy)
        elif cen:
            self.player_xy = (cen[0], cen[1])

        for x, y, _, _ in ch[:20]:
            if (x, y) not in self.tried_clicks:
                self.click_q.appendleft((x, y))

        if progressed:
            window = [t[0] for t in list(self.recent)[-20:]]
            if window:
                self.recipes.append(window)
                self.recipes = self.recipes[-24:]
            self.best_levels = levels_after
            self.on_level_up()

    def on_level_up(self) -> None:
        self.tried_clicks.clear()
        self.click_q.clear()
        self.stall = 0
        self.replaying = None
        self.steps_level = 0
        self.pair_visits = Counter()
        self.goals = []
        self.seek_target = None
        self.notes.append(f"LEVEL_UP to {self.best_levels}")

    def refresh_goals(self, frame: FrameData) -> None:
        grid = top_grid(frame)
        if not grid:
            return
        bg = guess_bg(grid)
        blobs = connected_blobs(grid, bg)
        # Prefer small rare-colored blobs as interaction targets
        hist = Counter(c for row in grid for c in row)
        goals: list[tuple[float, float]] = []
        for bl in blobs:
            rarity = 1.0 / max(1, hist[bl["color"]])
            if bl["size"] <= 80 and rarity > 0:
                goals.append((bl["cx"], bl["cy"]))
        # Dedup near-duplicates
        cleaned: list[tuple[float, float]] = []
        for g in goals:
            if all((g[0] - c[0]) ** 2 + (g[1] - c[1]) ** 2 > 9 for c in cleaned):
                cleaned.append(g)
        self.goals = cleaned[:20]
        if self.seek_target is None and self.goals and self.player_xy:
            # Pick farthest unexplored-ish goal
            self.seek_target = max(
                self.goals,
                key=lambda g: (g[0] - self.player_xy[0]) ** 2
                + (g[1] - self.player_xy[1]) ** 2,  # type: ignore[index]
            )
            self.seek_ttl = 40

    def axis_actions(self) -> dict[str, Optional[str]]:
        """Map cardinal directions to learned action names."""
        best = {"up": (None, 0.0), "down": (None, 0.0), "left": (None, 0.0), "right": (None, 0.0)}
        for name, (dx, dy, n) in self.move_vec.items():
            if n < 2:
                continue
            if abs(dy) >= abs(dx) and abs(dy) > 0.3:
                key = "up" if dy < 0 else "down"
                if abs(dy) > best[key][1]:
                    best[key] = (name, abs(dy))
            elif abs(dx) > 0.3:
                key = "left" if dx < 0 else "right"
                if abs(dx) > best[key][1]:
                    best[key] = (name, abs(dx))
        return {k: v[0] for k, v in best.items()}

    def bfs_unexplored(self, start: str, depth: int = 8) -> Optional[str]:
        """Return first action toward a less-visited / unknown edge."""
        q: deque = deque([(start, [])])
        seen = {start}
        while q:
            node, path = q.popleft()
            if len(path) >= depth:
                continue
            edges = self.graph.get(node, {})
            # Prefer unknown actions later via caller; here walk known graph
            for aname, nxt in edges.items():
                if nxt not in seen:
                    if self.visits[nxt] <= 1 or aname not in edges:
                        return path[0] if path else aname
                    seen.add(nxt)
                    q.append((nxt, path + [aname]))
            # If this node has few known edges, signal to explore here
            if path and len(edges) < 3:
                return path[0]
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Optional LLM advisor (urllib — no openai package required)
# ═══════════════════════════════════════════════════════════════════════════

class LLMAdvisor:
    def __init__(self) -> None:
        self.key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.model = os.environ.get("AETHER_LLM_MODEL", "gpt-4o-mini").strip()
        self.base = os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.enabled = bool(self.key) and os.environ.get("AETHER_LLM", "0").strip() == "1"
        self.fail_streak = 0
        self.calls = 0
        self.max_calls = int(os.environ.get("AETHER_LLM_MAX_CALLS", "40"))

    def suggest(
        self,
        frame: FrameData,
        legal: list[GameAction],
        wm: WorldModel,
        game_id: str,
    ) -> Optional[tuple[str, Optional[int], Optional[int], str]]:
        if not self.enabled or self.fail_streak >= 3 or self.calls >= self.max_calls:
            return None
        # Only ask LLM when stuck or every N steps early
        if wm.stall < 2 and wm.steps_level not in (5, 15, 30, 60, 100):
            if wm.steps_level > 10 and wm.steps_level % 25 != 0:
                return None

        legal_names = [a.name for a in legal]
        axis = wm.axis_actions()
        notes = "; ".join(wm.notes[-8:]) or "none"
        prompt = {
            "game_id": game_id,
            "state": frame.state.name if hasattr(frame.state, "name") else str(frame.state),
            "levels_completed": frame.levels_completed,
            "win_levels": getattr(frame, "win_levels", None),
            "legal_actions": legal_names,
            "learned_axes": axis,
            "player_xy": wm.player_xy,
            "seek_target": wm.seek_target,
            "stall": wm.stall,
            "notes": notes,
            "grid": compact_grid_text(frame),
            "instruction": (
                "You play an unfamiliar grid game. Infer controls and goals from "
                "evidence. Reply JSON only: "
                '{"action":"ACTION1","x":null,"y":null,"why":"..."}. '
                "For ACTION6 provide integer x,y in [0,63]. "
                "Prefer legal_actions. Do not RESET unless GAME_OVER."
            ),
        }
        body = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "Return one JSON object only. No markdown.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        try:
            req = urllib.request.Request(
                f"{self.base}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            text = raw["choices"][0]["message"]["content"]
            self.calls += 1
            self.fail_streak = 0
            return self._parse(text, legal_names)
        except Exception as e:  # noqa: BLE001
            self.fail_streak += 1
            wm.notes.append(f"LLM_FAIL:{type(e).__name__}")
            return None

    @staticmethod
    def _parse(
        text: str, legal_names: list[str]
    ) -> Optional[tuple[str, Optional[int], Optional[int], str]]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        # Find first {...}
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        name = str(obj.get("action", "")).upper().strip()
        if name not in legal_names and name != "RESET":
            return None
        x = obj.get("x")
        y = obj.get("y")
        try:
            xi = int(x) if x is not None else None
            yi = int(y) if y is not None else None
        except (TypeError, ValueError):
            xi = yi = None
        why = str(obj.get("why", "llm"))[:200]
        return name, xi, yi, why


# ═══════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════

class MyAgent(Agent):
    """Aether-Prime: world-model explorer with optional LLM advisor."""

    MAX_ACTIONS = 800
    STALL_UNDO = 5
    CLICK_REFRESH = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        self.rng = random.Random(seed)
        self.wm = WorldModel()
        self.llm = LLMAdvisor()
        self._pending: Optional[tuple[str, ActionKey, int, list[list[int]]]] = None
        self._click_age = 0

    @property
    def name(self) -> str:
        tag = "llm" if self.llm.enabled else "sym"
        return f"{super().name}.aether-prime.{tag}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ── observe ────────────────────────────────────────────────────────────

    def _observe(self, latest: FrameData) -> None:
        h = grid_hash(latest)
        self.wm.visits[h] += 1
        if self.wm.last_hash == h:
            self.wm.stall += 1
        else:
            self.wm.stall = 0
        self.wm.last_hash = h

        if self._pending is not None:
            bh, action, lb, bg = self._pending
            self._pending = None
            self.wm.record_transition(
                bh,
                h,
                action,
                bg,
                top_grid(latest),
                lb,
                int(latest.levels_completed or 0),
            )

        if self.wm.steps_level % 7 == 0:
            self.wm.refresh_goals(latest)

    def _commit(
        self,
        action: GameAction,
        latest: FrameData,
        reason: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> GameAction:
        key = ActionKey(action.name, x, y)
        if action.is_complex():
            xx = 32 if x is None else int(x)
            yy = 32 if y is None else int(y)
            action.set_data({"x": xx, "y": yy})
            key = ActionKey(action.name, xx, yy)
            action.reasoning = {
                "policy": "aether-prime",
                "why": reason,
                "xy": [xx, yy],
                "L": latest.levels_completed,
                "stall": self.wm.stall,
                "llm": self.llm.enabled,
            }
        else:
            action.reasoning = (
                f"aether-prime:{reason}|L{latest.levels_completed}"
                f"|S{self.wm.stall}|k{self.wm.streak}"
            )

        self._pending = (
            grid_hash(latest),
            key,
            int(latest.levels_completed or 0),
            [row[:] for row in top_grid(latest)],
        )
        if key.name == self.wm.last_action:
            self.wm.streak += 1
        else:
            self.wm.streak = 1
            self.wm.last_action = key.name
        self.wm.steps_level += 1
        self.wm.pair_visits[(grid_hash(latest), key.name)] += 1
        if self.wm.seek_ttl > 0:
            self.wm.seek_ttl -= 1
            if self.wm.seek_ttl == 0:
                self.wm.seek_target = None
        return action

    def _pick_click(self, latest: FrameData) -> tuple[int, int]:
        self._click_age += 1
        if not self.wm.click_q or self._click_age >= self.CLICK_REFRESH:
            self._click_age = 0
            for pt in salient_clicks(latest):
                if pt not in self.wm.tried_clicks:
                    self.wm.click_q.append(pt)
        while self.wm.click_q:
            pt = self.wm.click_q.popleft()
            if pt not in self.wm.tried_clicks:
                self.wm.tried_clicks.add(pt)
                return pt
        grid = top_grid(latest)
        if grid:
            bg = guess_bg(grid)
            x0, y0, x1, y1 = bbox(grid, bg)
            return self.rng.randint(x0, x1), self.rng.randint(y0, y1)
        return self.rng.randint(0, 63), self.rng.randint(0, 63)

    # ── planning modules ───────────────────────────────────────────────────

    def _by_name(self, legal: list[GameAction], name: str) -> Optional[GameAction]:
        return next((a for a in legal if a.name == name), None)

    def _spatial_seek(
        self, legal: list[GameAction], latest: FrameData
    ) -> Optional[GameAction]:
        if not self.wm.player_xy or not self.wm.seek_target:
            return None
        axis = self.wm.axis_actions()
        if not any(axis.values()):
            return None
        px, py = self.wm.player_xy
        tx, ty = self.wm.seek_target
        dx, dy = tx - px, ty - py
        order: list[str] = []
        if abs(dx) >= abs(dy):
            order = (["right", "left"] if dx > 0 else ["left", "right"]) + (
                ["down", "up"] if dy > 0 else ["up", "down"]
            )
        else:
            order = (["down", "up"] if dy > 0 else ["up", "down"]) + (
                ["right", "left"] if dx > 0 else ["left", "right"]
            )
        # Near target → try interact-ish actions / clicks
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 3.5:
            for name in ("ACTION5", "ACTION7"):
                a = self._by_name(legal, name)
                if a and a.is_simple():
                    return self._commit(a, latest, "seek_interact")
            for a in legal:
                if a.is_complex():
                    return self._commit(
                        a, latest, "seek_click", int(tx), int(ty)
                    )
            self.wm.seek_target = None
            return None

        for cardinal in order:
            aname = axis.get(cardinal)
            if not aname:
                continue
            a = self._by_name(legal, aname)
            if a is None:
                continue
            # Avoid known self-loops at current state
            sh = grid_hash(latest)
            nxt = self.wm.graph.get(sh, {}).get(aname)
            if nxt == sh:
                continue
            if aname == self.wm.last_action and self.wm.streak >= 6:
                continue
            return self._commit(a, latest, f"seek_{cardinal}")
        return None

    def _score(
        self, action: GameAction, state_h: str, simples: list[GameAction]
    ) -> float:
        name = action.name
        pair_n = self.wm.pair_visits[(state_h, name)]
        st = self.wm.effects[name]
        trials = max(st["trials"], 1.0)
        score = 3.2 / (1.0 + pair_n)
        nxt = self.wm.graph.get(state_h, {}).get(name)
        if nxt is not None:
            score += 2.2 / (1.0 + self.wm.visits[nxt])
            if nxt == state_h:
                score -= 3.0
        else:
            score += 1.4
        score += 5.0 * (st["progress"] / trials)
        score += 0.9 * (st["change"] / trials)
        score -= 1.4 * (st["noop"] / trials)
        if name == self.wm.last_action:
            score -= 0.45 * self.wm.streak

        if action.is_complex():
            productive = any(self.wm.effects[a.name]["change"] > 0 for a in simples)
            avg_noop = 0.0
            n = 0
            for a in simples:
                s = self.wm.effects[a.name]
                if s["trials"] > 0:
                    avg_noop += s["noop"] / s["trials"]
                    n += 1
            avg_noop = avg_noop / n if n else 0.4
            if productive and avg_noop < 0.55:
                score -= 2.2
            else:
                score += 0.15 * max(0, 24 - len(self.wm.tried_clicks))
            if self.wm.steps_level < 10:
                score -= 1.8
        score += 0.04 * self.rng.random()
        return score

    def _recipe(
        self, legal: list[GameAction], latest: FrameData
    ) -> Optional[GameAction]:
        if self.wm.replaying is None:
            if not self.wm.recipes or (self.wm.stall < 2 and self.rng.random() > 0.25):
                return None
            self.wm.replaying = list(self.wm.recipes[-1])
            self.wm.recipe_i = 0
        assert self.wm.replaying is not None
        if self.wm.recipe_i >= len(self.wm.replaying):
            self.wm.replaying = None
            return None
        step = self.wm.replaying[self.wm.recipe_i]
        self.wm.recipe_i += 1
        a = self._by_name(legal, step.name)
        if a is None:
            self.wm.replaying = None
            return None
        if a.is_complex():
            x = step.x if step.x is not None else self._pick_click(latest)[0]
            y = step.y if step.y is not None else self._pick_click(latest)[1]
            return self._commit(a, latest, "recipe", x, y)
        return self._commit(a, latest, "recipe")

    # ── main policy ────────────────────────────────────────────────────────

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self._observe(latest_frame)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.GAME_OVER:
                self.wm.deaths += 1
                self.wm.notes.append("GAME_OVER")
            self.wm.replaying = None
            self.wm.stall = 0
            self.wm.streak = 0
            self.wm.steps_level = 0
            self.wm.seek_target = None
            self._pending = None
            action = GameAction.RESET
            action.reasoning = "aether-prime:reset"
            return action

        legal = resolve_actions(latest_frame.available_actions)
        state_h = grid_hash(latest_frame)
        simples = [a for a in legal if a.is_simple()]
        complex_acts = [a for a in legal if a.is_complex()]

        if complex_acts and not simples:
            x, y = self._pick_click(latest_frame)
            return self._commit(complex_acts[0], latest_frame, "click_sweep", x, y)
        if complex_acts and simples and all(a.name == "ACTION7" for a in simples):
            if self.wm.stall >= 3:
                return self._commit(simples[0], latest_frame, "undo")
            x, y = self._pick_click(latest_frame)
            return self._commit(complex_acts[0], latest_frame, "click_sweep", x, y)

        # 1) LLM advisor (optional)
        advice = self.llm.suggest(latest_frame, legal, self.wm, self.game_id)
        if advice is not None:
            name, xi, yi, why = advice
            if name == "RESET":
                # ignore RESET mid-game
                pass
            else:
                a = self._by_name(legal, name)
                if a is not None:
                    if a.is_complex():
                        if xi is None or yi is None:
                            xi, yi = self._pick_click(latest_frame)
                        return self._commit(a, latest_frame, f"llm:{why}", xi, yi)
                    return self._commit(a, latest_frame, f"llm:{why}")

        # 2) Undo when stalled
        undo = self._by_name(legal, "ACTION7")
        if self.wm.stall >= self.STALL_UNDO and undo is not None:
            if self.rng.random() < 0.7:
                return self._commit(undo, latest_frame, "undo")

        # 3) Recipe replay after progress history
        rec = self._recipe(legal, latest_frame)
        if rec is not None:
            return rec

        # 4) Spatial seek toward hypothesized goals
        if self.wm.steps_level >= 6:
            seek = self._spatial_seek(legal, latest_frame)
            if seek is not None and self.rng.random() < 0.75:
                return seek

        # 5) Untried edges (shuffled, anti-streak)
        untried = [a for a in simples if self.wm.pair_visits[(state_h, a.name)] == 0]
        if untried:
            if self.wm.streak >= 2 and self.wm.last_action:
                alt = [a for a in untried if a.name != self.wm.last_action]
                if alt:
                    untried = alt
            return self._commit(self.rng.choice(untried), latest_frame, "edge")

        # 6) Graph BFS hint
        hint = self.wm.bfs_unexplored(state_h)
        if hint:
            a = self._by_name(legal, hint)
            if a is not None and not (
                a.name == self.wm.last_action and self.wm.streak >= 5
            ):
                return self._commit(a, latest_frame, "bfs")

        # 7) Softmax over scored actions
        scored: list[tuple[float, GameAction]] = []
        for a in simples:
            scored.append((self._score(a, state_h, simples), a))
        for a in complex_acts:
            scored.append((self._score(a, state_h, simples), a))
        if not scored:
            fb = self.rng.choice(legal)
            if fb.is_complex():
                x, y = self._pick_click(latest_frame)
                return self._commit(fb, latest_frame, "fallback", x, y)
            return self._commit(fb, latest_frame, "fallback")

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: min(5, len(scored))]
        temp = 0.5 + 0.05 * min(self.wm.stall, 10)
        weights = [math.exp(s / max(temp, 0.15)) for s, _ in top]
        pick = self.rng.choices(top, weights=weights, k=1)[0][1]
        if pick.is_complex():
            x, y = self._pick_click(latest_frame)
            return self._commit(pick, latest_frame, "explore_click", x, y)
        return self._commit(pick, latest_frame, "explore")
