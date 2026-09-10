"""CEAX-primary controller for unknown ARC games (no plugins / no canned solvers).

Closed loop (CEAX-AGI-Explorer-v0.4):
  Observation → Hypotheses → Experiment → World Model → Counterfactuals
  → Planning → Verification → Skill Extraction → Transfer

v3 unknown-capability upgrades:
  - Visual-match arrange: top reference colors → mid slots from bottom palette → ACTION5
  - Lattice keyboard nav: rare-agent + corridor halfway BFS (step 6/3) + motion lock + blocked edges
  - Undo only after failed submit in arrange mode (do not wipe placements)
  - Keep click experiment/exploit for button/gate games

v3.4 CEAX-chassis push (no game-id hardcoding):
  - Sub-object clicks along elongated blobs (slider / bar ends)
  - Structure/pose effect credit (aspect + occupancy), not only delta_pixels
  - Revert/noop click → block key + undo search when ACTION7 available
  - Hybrid: after effective click, walk L/R with learned motion gain

v3.5 align/slider push:
  - Past-midpoint end samples (chrome frames often have centroid == slider mid → noop)
  - Rare dual-marker pair-gap as alignment progress signal
  - Sticky repeat of clicks that shrink marker gap (extend/rotate-to-align)

v3.6 attract/pull + hybrid climb:
  - Click-only: leash-pull rare agent blob toward goal box (settle physics)
  - Hybrid: prefer mutating mid-board nodes then walk L/R toward rare goal speck
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hypotheses import CompetingHypotheses, HypothesisKind
from .rule_transfer import RuleTransfer
from .skill_library import SkillLibrary
from .world_model import CounterfactualWorldModel
from .counterfactual import CounterfactualPlanner


def _obj_centroid(obj) -> Optional[Tuple[int, int]]:
    pixels = getattr(obj, "pixels", None) or []
    if pixels:
        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        return int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))
    bbox = getattr(obj, "bbox", None)
    if bbox and len(bbox) == 4:
        a, b, c, d = bbox
        if abs(c - a) <= 64 and abs(d - b) <= 64:
            return int((a + c) / 2), int((b + d) / 2)
    return None


def _obj_area(obj) -> int:
    pixels = getattr(obj, "pixels", None) or []
    if pixels:
        return len(pixels)
    bbox = getattr(obj, "bbox", None)
    if bbox and len(bbox) == 4:
        a, b, c, d = bbox
        return max(1, abs(c - a) * abs(d - b))
    return 9999


def _obj_key(obj) -> Tuple:
    color = int(getattr(obj, "color", -1))
    area = min(_obj_area(obj) // 4, 64)
    return (color, area)


def _obj_sample_points(obj) -> List[Tuple[int, int]]:
    """Centroid + major-axis samples for elongated / chrome blobs (sliders, bars).

    Critical: many chrome frames have centroid exactly on the widget midline;
    mid clicks are noops when the env uses strict past-midpoint lengthen.
    Always bias end samples past 0.5 along the long axis.
    """
    pts: List[Tuple[int, int]] = []
    c = _obj_centroid(obj)
    if c is not None:
        pts.append(c)
    pixels = getattr(obj, "pixels", None) or []
    if len(pixels) < 12:
        return pts
    ys = [int(p[0]) for p in pixels]
    xs = [int(p[1]) for p in pixels]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    # 13x7 chrome rings etc.: aspect ~1.8 and area ~36 — must still get ends
    elongated = max(w, h) >= int(1.45 * min(w, h)) or max(w, h) >= 9
    if not elongated and len(pixels) < 40:
        return pts
    cy = int(round(sum(ys) / len(ys)))
    cx = int(round(sum(xs) / len(xs)))
    extras: List[Tuple[int, int]] = []
    # Far-end first (lengthen); near-end second (shorten) — saves budget on align puzzles
    fracs = (0.88, 0.12, 0.75, 0.25) if elongated else (0.8, 0.2)
    if w >= h:
        for q in fracs:
            tx = int(round(x0 + q * (x1 - x0)))
            cand = [(x, y) for x, y in zip(xs, ys) if abs(x - tx) <= 1]
            if not cand:
                cand = [(x, y) for x, y in zip(xs, ys) if abs(x - tx) <= 2]
            if cand:
                extras.append(min(cand, key=lambda p: abs(p[1] - cy)))
    else:
        for q in fracs:
            ty = int(round(y0 + q * (y1 - y0)))
            cand = [(x, y) for x, y in zip(xs, ys) if abs(y - ty) <= 1]
            if not cand:
                cand = [(x, y) for x, y in zip(xs, ys) if abs(y - ty) <= 2]
            if cand:
                extras.append(min(cand, key=lambda p: abs(p[0] - cx)))
    for p in extras:
        if p not in pts and 0 <= p[0] < 64 and 0 <= p[1] < 64:
            pts.append(p)
    return pts[:6]


def _dual_marker_gap(grid) -> Optional[float]:
    """Pair-gap for plus-like rare markers (equal tiny CCs, usually 2 or 4)."""
    if not isinstance(grid, np.ndarray) or grid.size == 0:
        return None
    play = grid[:58] if grid.shape[0] >= 58 else grid
    hist = Counter(int(c) for c in play.flatten().tolist())
    if not hist:
        return None
    bg = hist.most_common(1)[0][0]
    H, W = play.shape
    best_gap: Optional[float] = None
    best_rank = -1.0
    for color, n in hist.items():
        if color == bg or color == 0 or n < 6 or n > 48:
            continue
        visited = np.zeros_like(play, dtype=bool)
        cents: List[Tuple[float, float]] = []
        sizes: List[int] = []
        ys, xs = np.where(play == color)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((x, y))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and int(play[ny, nx]) == color:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if 3 <= len(cells) <= 6:
                cents.append(
                    (sum(p[0] for p in cells) / len(cells), sum(p[1] for p in cells) / len(cells))
                )
                sizes.append(len(cells))
        k = len(cents)
        if k not in (2, 4, 6):
            continue
        if max(sizes) - min(sizes) > 2:
            continue
        rem = list(range(k))
        gap = 0.0
        while len(rem) >= 2:
            bi, bj, bd = rem[0], rem[1], 1e9
            for i in range(len(rem)):
                for j in range(i + 1, len(rem)):
                    a, b = cents[rem[i]], cents[rem[j]]
                    d = abs(a[0] - b[0]) + abs(a[1] - b[1])
                    if d < bd:
                        bi, bj, bd = rem[i], rem[j], d
            gap += bd
            rem = [r for r in rem if r != bi and r != bj]
        rank = (10.0 if k == 4 else 5.0) + gap * 0.01
        if rank > best_rank:
            best_rank = rank
            best_gap = gap
    return best_gap


def _cc_centroids(
    play: np.ndarray, color: int, min_sz: int, max_sz: int
) -> List[Tuple[int, int, int]]:
    """Return (cx, cy, size) for CCs of color in size band."""
    H, W = play.shape
    visited = np.zeros_like(play, dtype=bool)
    out: List[Tuple[int, int, int]] = []
    ys, xs = np.where(play == color)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((x, y))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and int(play[ny, nx]) == color:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        n = len(cells)
        if min_sz <= n <= max_sz:
            cx = int(round(sum(p[0] for p in cells) / n))
            cy = int(round(sum(p[1] for p in cells) / n))
            out.append((cx, cy, n))
    return out


def _find_pull_roles(
    grid,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """Rare small agent + mid/large goal box → attract/pull puzzle (su15-like)."""
    if not isinstance(grid, np.ndarray) or grid.size == 0:
        return None
    play = grid[:58] if grid.shape[0] >= 58 else grid
    field = play.copy()
    field[:10, :] = -1
    hist = Counter(int(c) for c in field.flatten().tolist() if c >= 0)
    if not hist:
        return None
    bg = hist.most_common(1)[0][0]
    agents: List[Tuple[int, int, int, int]] = []
    goals: List[Tuple[int, int, int, int]] = []
    for color, n in hist.items():
        if color == bg or color == 0:
            continue
        if 4 <= n <= 40:
            for cx, cy, sz in _cc_centroids(field, color, 3, 20):
                if cy >= 10:
                    agents.append((cx, cy, sz, color))
        if 25 <= n <= 220:
            for cx, cy, sz in _cc_centroids(field, color, 20, 220):
                if cy >= 10:
                    goals.append((cx, cy, sz, color))
    if not agents or not goals:
        return None
    # Prefer smallest rare agent
    agents.sort(key=lambda t: (t[2], hist.get(t[3], 99)))
    ax, ay, _, ac = agents[0]
    best_g = None
    best_d = -1.0
    for gx, gy, sz, col in goals:
        if col == ac:
            continue
        d = float(abs(gx - ax) + abs(gy - ay))
        if d > best_d:
            best_d = d
            best_g = (gx, gy)
    if best_g is None or best_d < 8:
        return None
    return (ax, ay), best_g, best_d


def _structure_delta(prev, curr) -> float:
    """Pose / occupancy change beyond raw pixel count (rotate/extend signal)."""
    if not isinstance(prev, np.ndarray) or not isinstance(curr, np.ndarray):
        return 0.0
    if prev.shape != curr.shape or prev.size == 0:
        return 0.0
    p0 = prev[:58] if prev.shape[0] >= 58 else prev
    p1 = curr[:58] if curr.shape[0] >= 58 else curr
    if p0.shape != p1.shape:
        return 0.0
    bg0 = int(Counter(p0.flatten().tolist()).most_common(1)[0][0])
    bg1 = int(Counter(p1.flatten().tolist()).most_common(1)[0][0])
    m0 = p0 != bg0
    m1 = p1 != bg1

    def _aspect(m):
        ys, xs = np.where(m)
        if len(xs) < 2:
            return 1.0
        w = float(xs.max() - xs.min() + 1)
        h = float(ys.max() - ys.min() + 1)
        return w / max(h, 1.0)

    a0, a1 = _aspect(m0), _aspect(m1)
    aspect_chg = abs(a0 - a1) / max(a0, a1, 1e-3)
    u0 = len({int(c) for c in p0[m0].tolist()}) if m0.any() else 0
    u1 = len({int(c) for c in p1[m1].tolist()}) if m1.any() else 0
    color_chg = abs(u0 - u1) / 8.0
    flip = float(np.logical_xor(m0, m1).mean()) if m0.size else 0.0
    return float(min(1.0, 0.55 * aspect_chg + 0.35 * color_chg + 1.2 * flip))


@dataclass
class ClickTarget:
    xy: Tuple[int, int]
    key: Tuple
    area: int
    color: int
    trials: int = 0
    effect_sum: float = 0.0
    progress_hits: int = 0

    @property
    def score(self) -> float:
        if self.trials <= 0:
            # Untried: honor ingest priors (chrome ends > mid / solid rods)
            return 0.45 + float(self.effect_sum)
        return (self.effect_sum / self.trials) + 2.5 * self.progress_hits


def _norm_action(name: str) -> str:
    s = str(name or "").strip().upper()
    if s.startswith("ACTION"):
        return s
    if s.isdigit():
        return f"ACTION{int(s)}"
    mapping = {
        "RESET": "RESET", "UP": "ACTION1", "DOWN": "ACTION2",
        "LEFT": "ACTION3", "RIGHT": "ACTION4", "SPACE": "ACTION5",
        "CLICK": "ACTION6", "UNDO": "ACTION7",
    }
    return mapping.get(s, s)


class CeaxController:
    """Primary decision brain for unknown environments."""

    def __init__(self, cfg=None, logger=None):
        self.cfg = cfg
        self.log = logger
        self.hyps = CompetingHypotheses()
        self.skills = SkillLibrary(min_confidence=0.55)
        self.model = CounterfactualWorldModel()
        self.planner = CounterfactualPlanner()
        self.transfer = RuleTransfer(min_confidence=0.55)
        self.targets: Dict[Tuple, ClickTarget] = {}
        self._grid_i = 0
        self._last_click: Optional[Tuple[int, int]] = None
        self._last_key: Optional[Tuple] = None
        self._last_action: Optional[str] = None
        self._prev_hash: Optional[str] = None
        self._levels = 0
        self._step = 0
        self._exploit_streak = 0
        self._stale_exploits = 0
        self._clicks_since_submit = 0
        self._effective_clicks = 0
        self._submit_trials = 0
        self._submit_success = 0
        self._alt_i = 0
        self._preferred_colors: List[int] = []
        self._recent_actions: deque = deque(maxlen=12)
        self._action_effects: Dict[str, List[float]] = defaultdict(list)
        self._combo_phase = 0
        self._arrange_mode = False
        self._arrange_plan: List[Tuple[str, Optional[Tuple[int, int]]]] = []
        self._arrange_i = 0
        self._arrange_variant = 0
        self._need_arrange_undo = False
        self._post_level_budget = 0
        self._kb_blocked: set = set()
        self._kb_prev_pos: Optional[Tuple[int, int]] = None
        self._kb_last: Optional[str] = None
        self._kb_step = 6
        self._kb_player_colors: List[int] = []
        self._kb_corridor: Optional[int] = None
        self._kb_goal: Optional[int] = None
        self._kb_failed_pc: set = set()
        self._kb_motion_pos: Optional[Tuple[int, int]] = None
        self._kb_no_move = 0
        self._kb_verified = False
        self._kb_hint_pc: List[int] = []
        self._kb_recent_pos: deque = deque(maxlen=10)
        self._kb_dist_hist: deque = deque(maxlen=8)
        self._kb_plan: List[str] = []
        self._need_click_undo = False
        self._click_fail_streak = 0
        self._blocked_keys: set = set()
        self._hybrid_walk_budget = 0
        self._align_gap: Optional[float] = None
        self._sticky_clicks: List[Tuple[int, int]] = []
        self._sticky_i = 0
        self._sticky_budget = 0
        self._pull_goal: Optional[Tuple[int, int]] = None
        self._pull_agent: Optional[Tuple[int, int]] = None
        self._pull_dist: Optional[float] = None
        self._pull_hits = 0
        self._pull_mode = False
        self._pull_eligible = False
        self.metrics = {
            "experiments": 0,
            "exploits": 0,
            "submits": 0,
            "undos": 0,
            "skills": 0,
            "level_ups": 0,
            "arrange_steps": 0,
            "nav_steps": 0,
            "align_hits": 0,
            "pull_hits": 0,
        }

    def reset_game(self, *, import_skills: Optional[dict] = None):
        exported = self.skills.export()
        self.hyps = CompetingHypotheses()
        self.model = CounterfactualWorldModel()
        self.targets.clear()
        self._grid_i = 0
        self._last_click = None
        self._last_key = None
        self._last_action = None
        self._prev_hash = None
        self._levels = 0
        self._step = 0
        self._exploit_streak = 0
        self._stale_exploits = 0
        self._clicks_since_submit = 0
        self._effective_clicks = 0
        self._submit_trials = 0
        self._submit_success = 0
        self._alt_i = 0
        self._preferred_colors = []
        self._recent_actions.clear()
        self._action_effects.clear()
        self._combo_phase = 0
        self._arrange_mode = False
        self._arrange_plan = []
        self._arrange_i = 0
        self._arrange_variant = 0
        self._need_arrange_undo = False
        self._post_level_budget = 0
        self._kb_blocked.clear()
        self._kb_prev_pos = None
        self._kb_last = None
        self._kb_step = 6
        self._kb_player_colors = []
        self._kb_corridor = None
        self._kb_goal = None
        self._kb_motion_pos = None
        self._kb_no_move = 0
        self._kb_hint_pc = []
        self._kb_recent_pos.clear()
        self._kb_dist_hist.clear()
        self._kb_plan = []
        self._need_click_undo = False
        self._click_fail_streak = 0
        self._blocked_keys.clear()
        self._hybrid_walk_budget = 0
        self._align_gap = None
        self._sticky_clicks = []
        self._sticky_i = 0
        self._sticky_budget = 0
        self._pull_goal = None
        self._pull_agent = None
        self._pull_dist = None
        self._pull_hits = 0
        self._pull_mode = False
        self._pull_eligible = False
        # Keep _kb_failed_pc across NOT_PLAYED/lose resets so decoy roles stay banned
        if import_skills:
            self.skills.clear()
            self.skills.import_many(import_skills, cross_game=True)
        elif exported:
            self.skills.import_many(exported, cross_game=True)
        self._preferred_colors = self._colors_from_skills()
        self.metrics = {
            "experiments": 0,
            "exploits": 0,
            "submits": 0,
            "undos": 0,
            "skills": len(self.skills.skills),
            "level_ups": 0,
            "arrange_steps": 0,
            "nav_steps": 0,
            "align_hits": 0,
            "pull_hits": 0,
        }

    def _colors_from_skills(self) -> List[int]:
        colors = []
        for name, val in self.skills.skills.items():
            c = val.get("color")
            if c is not None:
                colors.append(int(c))
            elif "click_target_c" in name:
                try:
                    colors.append(int(name.split("_c")[1].split("_")[0]))
                except Exception:
                    pass
        return colors[:6]

    def on_level_up(self):
        self.metrics["level_ups"] += 1
        extracted = self.transfer.extract(self.hyps.ranked())
        self.skills.import_many(extracted, cross_game=False)
        ranked = sorted(self.targets.values(), key=lambda t: t.score, reverse=True)
        for t in ranked[:4]:
            if t.trials >= 1 and t.score >= 0.12:
                self.skills.add(
                    f"click_target_c{t.color}_a{t.area}",
                    f"IF click color={t.color} area~{t.area} THEN effect",
                    min(0.99, 0.5 + t.score),
                    family="click_interact",
                    cross_game=True,
                    kind="click_effect",
                    xy=t.xy,
                    color=t.color,
                )
        if self._submit_success > 0 or (
            self._submit_trials > 0 and self._levels > 0
        ):
            self.skills.add(
                "submit_after_arrange",
                "IF several effective clicks THEN try ACTION5 submit",
                0.85,
                family="click_interact",
                cross_game=True,
                kind="level_progress",
            )
        if self._kb_corridor is not None:
            self.skills.add(
                "lattice_nav_corridor",
                f"IF keyboard THEN BFS on corridor color={self._kb_corridor}",
                0.8,
                family="movement",
                cross_game=True,
                kind="movement",
                color=self._kb_corridor,
            )
        self.metrics["skills"] = len(self.skills.skills)
        self._preferred_colors = [t.color for t in ranked[:3] if t.progress_hits or t.score > 0.2]
        self._preferred_colors.extend(self._colors_from_skills())
        self.targets.clear()
        self._grid_i = 0
        self._exploit_streak = 0
        self._stale_exploits = 0
        self._clicks_since_submit = 0
        self._effective_clicks = 0
        self._alt_i = 0
        self._arrange_plan = []
        self._arrange_i = 0
        self._arrange_variant = 0
        self._need_arrange_undo = False
        self._post_level_budget = 48
        self._kb_hint_pc = list(self._kb_player_colors) if self._kb_player_colors else list(self._kb_hint_pc)
        self._kb_blocked.clear()
        self._kb_prev_pos = None
        self._kb_last = None
        self._kb_player_colors = []
        self._kb_corridor = None
        self._kb_goal = None
        self._kb_step = 6
        self._kb_failed_pc = set()
        self._kb_motion_pos = None
        self._kb_no_move = 0
        self._kb_verified = False
        self._kb_recent_pos.clear()
        self._kb_dist_hist.clear()
        self._kb_plan = []
        self._need_click_undo = False
        self._click_fail_streak = 0
        self._blocked_keys.clear()
        self._hybrid_walk_budget = 0
        self._align_gap = None
        self._sticky_clicks = []
        self._sticky_i = 0
        self._sticky_budget = 0
        self._pull_goal = None
        self._pull_agent = None
        self._pull_dist = None
        self._pull_hits = 0
        self._pull_mode = False
        self._pull_eligible = False
        self.hyps = CompetingHypotheses()

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
        if levels > self._levels:
            self._levels = levels
            self.on_level_up()
            progressed = True

        effect = min(1.0, delta_pixels / 64.0) + (0.8 if progressed else 0.0)
        # Structure/pose credit: rotate/extend may change few pixels but reshape blobs
        struct = _structure_delta(prev_grid, grid)
        if struct > 0.04:
            effect = min(1.0, effect + 0.55 * struct)

        # Alignment progress: rare dual-marker pair-gap shrinks when slots approach targets
        gap = _dual_marker_gap(grid)
        if (
            gap is not None
            and self._align_gap is not None
            and self._last_action == "ACTION6"
            and self._last_click is not None
        ):
            improved = gap < self._align_gap - 0.4
            # Only treat as worsen when gap grows AND world barely changed (true backslide/noop)
            worsened = gap > self._align_gap + 1.0 and effect < 0.08 and delta_pixels <= 4
            overshoot = gap > self._align_gap + 1.0 and effect >= 0.08
            if improved:
                effect = min(1.0, effect + 0.45)
                self.metrics["align_hits"] = int(self.metrics.get("align_hits", 0)) + 1
                xy = (int(self._last_click[0]), int(self._last_click[1]))
                if xy not in self._sticky_clicks:
                    self._sticky_clicks.append(xy)
                self._sticky_clicks = self._sticky_clicks[-4:]
                self._sticky_budget = max(self._sticky_budget, 10)
            elif overshoot and self._last_click is not None:
                # Past the sweet spot for this control — drop only this end
                xy = (int(self._last_click[0]), int(self._last_click[1]))
                self._sticky_clicks = [p for p in self._sticky_clicks if p != xy]
            elif worsened and self._last_key is not None:
                # Soft-block key only — do NOT drop other seeded chrome ends
                self._blocked_keys.add(self._last_key)
                if self._last_key in self.targets:
                    self.targets[self._last_key].effect_sum *= 0.2
        if gap is not None:
            self._align_gap = gap
        elif progressed:
            self._align_gap = None

        # Attract/pull: only on click+undo physics games (never hybrid/keyboard)
        if self._pull_eligible:
            roles = _find_pull_roles(grid) if grid is not None else None
            if roles is not None:
                agent, goal, dist = roles
                self._pull_agent, self._pull_goal = agent, goal
                if (
                    self._pull_dist is not None
                    and self._last_action == "ACTION6"
                    and dist < self._pull_dist - 1.5
                ):
                    effect = min(1.0, effect + 0.5)
                    self._pull_hits += 1
                    self._pull_mode = True
                    self.metrics["pull_hits"] = int(self.metrics.get("pull_hits", 0)) + 1
                    self._need_click_undo = False
                    self._click_fail_streak = 0
                self._pull_dist = dist
            elif progressed:
                self._pull_dist = None
                self._pull_mode = False

        # Same sticky click with no pixel change → arm at limit; rotate to next end
        if (
            self._last_action == "ACTION6"
            and self._last_click is not None
            and delta_pixels <= 2
            and not progressed
            and len(self._sticky_clicks) >= 2
        ):
            xy = (int(self._last_click[0]), int(self._last_click[1]))
            if xy in self._sticky_clicks:
                self._sticky_i += 1  # skip ahead on next choose

        if self._last_action:
            self._action_effects[self._last_action].append(effect)
            self._recent_actions.append(self._last_action)

        # Keyboard: learn controllable avatar from motion; ban decoys that never move
        kbd = self._last_action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
        if kbd and isinstance(prev_grid, np.ndarray) and isinstance(grid, np.ndarray):
            if delta_pixels <= 0 and not progressed:
                self._kb_no_move += 1
                if self._kb_prev_pos is not None and self._last_action:
                    self._kb_blocked.add(
                        (self._kb_prev_pos[0], self._kb_prev_pos[1], self._last_action)
                    )
                # Only ban unverified decoys; keep locked avatar after real motion
                if (
                    self._kb_no_move >= 3
                    and self._kb_player_colors
                    and not self._kb_verified
                ):
                    self._kb_failed_pc.add(frozenset(self._kb_player_colors))
                    self._kb_player_colors = []
                    self._kb_corridor = None
                    self._kb_goal = None
                    self._kb_blocked.clear()
                    self._kb_motion_pos = None
                    self._kb_no_move = 0
                elif self._kb_no_move >= 2 and self._kb_verified:
                    # Controllable but blocked: try alternate lattice step
                    self._kb_step = 3 if self._kb_step == 6 else 6
                    self._kb_no_move = 0
            elif 0 < delta_pixels < 500:
                self._kb_no_move = 0
                play0 = prev_grid[:58] if prev_grid.shape[0] >= 58 else prev_grid
                play1 = grid[:58] if grid.shape[0] >= 58 else grid
                if play0.shape == play1.shape:
                    diff = play0 != play1
                    ys, xs = np.where(diff)
                    if len(xs):
                        self._kb_motion_pos = (int(round(xs.mean())), int(round(ys.mean())))
                        hist1 = Counter(int(c) for c in play1.flatten())
                        bg = hist1.most_common(1)[0][0]
                        moved = [
                            int(play1[y, x])
                            for x, y in zip(xs.tolist()[:80], ys.tolist()[:80])
                        ]
                        cnt = Counter(moved)
                        learned = []
                        for c, n in cnt.most_common():
                            if c == bg or c == 0:
                                continue
                            if c == self._kb_corridor or c == self._kb_goal:
                                continue
                            if hist1.get(c, 0) > 40 or n > 36:
                                continue
                            learned.append(c)
                            if len(learned) >= 3:
                                break
                        if learned:
                            # Do not shrink a prior-level hint set down to a speck
                            if (
                                self._kb_hint_pc
                                and set(self._kb_hint_pc) <= set(self._kb_player_colors or self._kb_hint_pc)
                                and not set(self._kb_hint_pc).issubset(set(learned))
                                and len(learned) < len(self._kb_hint_pc)
                            ):
                                learned = list(dict.fromkeys(list(self._kb_hint_pc) + learned))
                            self._kb_player_colors = learned
                            self._kb_verified = True
                            # Successful motion → unban this role-set
                            self._kb_failed_pc.discard(frozenset(learned))
                            # Also unban close subsets
                            for bad in list(self._kb_failed_pc):
                                if set(learned) & set(bad) and len(bad) <= len(learned) + 1:
                                    self._kb_failed_pc.discard(bad)

        if self._last_key and self._last_key in self.targets:
            t = self.targets[self._last_key]
            t.trials += 1
            t.effect_sum += effect
            if progressed:
                t.progress_hits += 1

        if self._last_action == "ACTION6":
            self._clicks_since_submit += 1
            if effect >= 0.08:
                self._effective_clicks += 1
                self._click_fail_streak = 0
                # Hybrid: effective world-edit → try walking avatar
                self._hybrid_walk_budget = max(self._hybrid_walk_budget, 3)
            else:
                self._click_fail_streak += 1
            # Revert / illegal: near-noop click → soft-block key; undo only after streak
            if not progressed and effect < 0.05 and delta_pixels <= 2:
                if self._last_key is not None:
                    self._blocked_keys.add(self._last_key)
                    if self._last_key in self.targets:
                        self.targets[self._last_key].effect_sum *= 0.15
                # Avoid undo thrashing on pure click-explore (su15); require streak
                if self._click_fail_streak >= 2:
                    self._need_click_undo = True
            if not progressed:
                self._stale_exploits += 2 if effect < 0.05 else 1
        elif self._last_action == "ACTION5":
            self._submit_trials += 1
            self._clicks_since_submit = 0
            if progressed or effect >= 0.15:
                self._submit_success += 1
                self.hyps.support(HypothesisKind.LEVEL_PROGRESS, amount=0.25)
            else:
                self.hyps.support(HypothesisKind.NOOP, amount=0.05)
                # Failed verify → undo once then rebuild with alternate mapping
                self._arrange_plan = []
                self._arrange_i = 0
                self._arrange_variant += 1
                self._need_arrange_undo = True

        if progressed:
            self._stale_exploits = 0

        if delta_pixels <= 0 and not progressed:
            if self._last_action == "ACTION6":
                self.hyps.support(HypothesisKind.NOOP, amount=0.04)
            else:
                self.hyps.support(HypothesisKind.BLOCKED_TRANSITION, amount=0.05)
        else:
            if self._last_action == "ACTION6":
                self.hyps.support(HypothesisKind.CLICK_EFFECT, amount=0.14)
            elif self._last_action and self._last_action.startswith("ACTION"):
                self.hyps.support(HypothesisKind.MOVEMENT, amount=0.10)
            if 0 < delta_pixels < 400:
                self.hyps.support(HypothesisKind.TOGGLE_ON_CONTACT, amount=0.05)
        if progressed:
            self.hyps.support(HypothesisKind.LEVEL_PROGRESS, amount=0.2)

        if self._prev_hash and self._last_action:
            self.model.observe(self._prev_hash, self._last_action, grid_hash or "changed")
        self._prev_hash = grid_hash or self._prev_hash

    def choose(
        self,
        *,
        valid_actions: Sequence[str],
        objects: Optional[Sequence[Any]] = None,
        grid=None,
        grid_hash: str = "",
    ) -> Tuple[str, Optional[Tuple[int, int]], str]:
        self._step += 1
        valid = [_norm_action(a) for a in valid_actions if a]
        seen = set()
        valid = [a for a in valid if not (a in seen or seen.add(a))]
        if not valid:
            return "ACTION1", None, "ceax_empty"

        if self._align_gap is None and grid is not None:
            self._align_gap = _dual_marker_gap(grid)

        has_click = "ACTION6" in valid
        has_submit = "ACTION5" in valid
        has_undo = "ACTION7" in valid
        has_lr = "ACTION3" in valid or "ACTION4" in valid
        click_only = has_click and set(valid) <= {"ACTION6", "ACTION7"}
        keyboard_only = not has_click and any(
            a in valid for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
        )
        # bp35-like: left/right + click (+undo), no ACTION5
        hybrid = has_click and has_lr and not has_submit

        # Strict pull gate: only pure click+undo (su15). Never hybrid.
        self._pull_eligible = bool(
            click_only
            and has_undo
            and "ACTION6" in valid
            and "ACTION7" in valid
            and not has_lr
            and not has_submit
            and set(valid) <= {"ACTION6", "ACTION7"}
        )
        if not self._pull_eligible:
            self._pull_mode = False

        if has_click and objects:
            self._ingest_objects(objects)
            self._arrange_mode = has_submit
            roles0 = _find_pull_roles(grid) if grid is not None else None
            pull_likely = (
                self._pull_eligible
                and roles0 is not None
                and roles0[2] >= 12
                and (self._align_gap is None or self._align_gap < 8)
            )
            if pull_likely:
                self._pull_mode = True
                self._pull_agent, self._pull_goal, self._pull_dist = roles0[0], roles0[1], roles0[2]
            elif click_only or (has_click and not has_submit and not has_lr):
                self._seed_align_chrome_ends(objects)

        # --- Attract/pull leash (su15-like settle physics) ---
        if has_click and self._pull_eligible and self._pull_mode and grid is not None:
            pull_xy = self._choose_pull_click(grid)
            if pull_xy is not None:
                self.metrics["exploits"] += 1
                self._commit("ACTION6", pull_xy, ("pull", pull_xy[0], pull_xy[1]))
                return "ACTION6", pull_xy, f"ceax_pull:{pull_xy}"

        # --- Sticky align: rotate gap-reducing / chrome far-end clicks ---
        if (
            has_click
            and not self._pull_mode
            and self._sticky_budget > 0
            and self._sticky_clicks
        ):
            self._sticky_budget -= 1
            xy = self._sticky_clicks[self._sticky_i % len(self._sticky_clicks)]
            self._sticky_i += 1
            self.metrics["exploits"] += 1
            self._commit("ACTION6", xy, ("sticky", xy[0], xy[1]))
            return "ACTION6", xy, f"ceax_align_sticky:{xy}"

        # Untried high-prior chrome ends before locking onto a single exploit
        if has_click and not self._pull_mode:
            hi = [
                t for t in self.targets.values()
                if t.trials == 0
                and t.score >= 0.85
                and t.key not in self._blocked_keys
            ]
            if hi:
                hi.sort(key=lambda t: -t.score)
                t = hi[0]
                self.metrics["experiments"] += 1
                self._commit("ACTION6", t.xy, t.key)
                return "ACTION6", t.xy, f"ceax_chrome_end:{t.xy}"

        # --- Arrange: visual match palette→slots then submit ---
        if has_submit and has_click and objects is not None:
            planned = self._next_arrange_action(objects)
            if planned is not None:
                act, xy, why = planned
                self._commit(act, xy, ("arrange", xy) if xy else None)
                if act == "ACTION5":
                    self.metrics["submits"] += 1
                else:
                    self.metrics["arrange_steps"] += 1
                    self.metrics["experiments"] += 1
                return act, xy, why

        # --- Fallback submit heuristics ---
        if has_submit and self._should_try_submit(has_click):
            self.metrics["submits"] += 1
            self._commit("ACTION5", None, None)
            return "ACTION5", None, "ceax_submit"

        # --- Undo (never wipe arrange mid-plan) ---
        if has_undo and self._should_undo():
            self.metrics["undos"] += 1
            self._commit("ACTION7", None, None)
            return "ACTION7", None, "ceax_undo"

        # --- Hybrid: mutate then walk; bias walk toward rare goal / open corridor ---
        if hybrid:
            # Periodically click untried mid-board mutate candidates (platform nodes)
            if self._step % 5 == 1 and objects is not None:
                mut = self._pick_hybrid_mutate_click(objects, grid)
                if mut is not None:
                    self.metrics["experiments"] += 1
                    self._commit("ACTION6", mut.xy, mut.key)
                    self._hybrid_walk_budget = max(self._hybrid_walk_budget, 6)
                    return "ACTION6", mut.xy, f"ceax_hybrid_mut:{mut.xy}"
            if self._hybrid_walk_budget > 0:
                self._hybrid_walk_budget -= 1
                a = self._hybrid_walk_action(valid, grid)
                if a in valid:
                    self._commit(a, None, None)
                    return a, None, f"ceax_hybrid_walk:{a}"
            elif self._step % 4 == 0:
                a = self._hybrid_walk_action(valid, grid)
                if a in valid:
                    self._commit(a, None, None)
                    return a, None, f"ceax_hybrid:{a}"

        # --- Click path ---
        if has_click:
            if self._post_level_budget > 0:
                self._post_level_budget -= 1

            force_explore = self._stale_exploits >= 8 or self._post_level_budget > 24
            tops = self._top_targets(3, min_trials=1, min_score=0.02)
            # Dual-button: alternate leftmost/rightmost among top scorers
            if tops and not force_explore and (click_only or hybrid or self._exploit_streak < 20):
                if len(tops) >= 2 and self._exploit_streak >= 1:
                    by_x = sorted(tops[:3], key=lambda t: t.xy[0])
                    pick = by_x[self._alt_i % 2] if len(by_x) >= 2 else tops[0]
                    self._alt_i += 1
                else:
                    pick = tops[0]
                self._exploit_streak += 1
                self.metrics["exploits"] += 1
                self._commit("ACTION6", pick.xy, pick.key)
                return "ACTION6", pick.xy, f"ceax_exploit:c{pick.color}@{pick.xy}"

            if force_explore:
                for t in tops:
                    t.effect_sum *= 0.25
                self._stale_exploits = 0
                self._exploit_streak = 0

            # Post-level: bias preferred colors first
            if self._post_level_budget > 0 and self._preferred_colors:
                pref = [
                    t for t in self.targets.values()
                    if t.color in self._preferred_colors and t.trials < 2
                ]
                pref.sort(key=lambda t: (t.trials, t.area))
                if pref:
                    t = pref[0]
                    self.metrics["experiments"] += 1
                    self._commit("ACTION6", t.xy, t.key)
                    return "ACTION6", t.xy, f"ceax_postL:{t.xy}"

            nxt = self._next_experiment_click()
            if nxt is not None:
                xy, key = nxt
                self._exploit_streak = 0
                self.metrics["experiments"] += 1
                self._commit("ACTION6", xy, key)
                return "ACTION6", xy, f"ceax_experiment:{xy}"

            # Suppress blind grid while rediscovering after level-up
            if self._post_level_budget <= 0:
                xy = self._next_grid_click()
                if xy is not None:
                    self.metrics["experiments"] += 1
                    self._commit("ACTION6", xy, ("grid", xy[0] // 8, xy[1] // 8))
                    return "ACTION6", xy, f"ceax_grid:{xy}"

            alts = sorted(self.targets.values(), key=lambda t: t.score, reverse=True)[:4]
            if alts:
                pick = alts[self._step % len(alts)]
                self.metrics["exploits"] += 1
                self._commit("ACTION6", pick.xy, pick.key)
                return "ACTION6", pick.xy, f"ceax_alt:{pick.xy}"

            self._commit("ACTION6", (32, 32), ("center",))
            return "ACTION6", (32, 32), "ceax_center"

        # --- Keyboard-only lattice nav ---
        if keyboard_only:
            action = self._choose_keyboard(valid, grid_hash, grid)
            self._commit(action, None, None)
            self.metrics["nav_steps"] += 1
            return action, None, f"ceax_kbd:{action}"

        simple = [a for a in valid if a not in ("ACTION6", "RESET")]
        if simple:
            ranked = self._rank_simple(simple, grid_hash)
            action = ranked[0]
            self._commit(action, None, None)
            return action, None, f"ceax_epistemic:{action}"

        return valid[0], None, "ceax_fallback"

    # ---------- arrange (visual match) ----------
    def _next_arrange_action(
        self, objects: Sequence[Any]
    ) -> Optional[Tuple[str, Optional[Tuple[int, int]], str]]:
        if not self._arrange_plan or self._arrange_i >= len(self._arrange_plan):
            self._arrange_plan = self._build_arrange_plan(objects)
            self._arrange_i = 0
        if not self._arrange_plan:
            return None
        kind, xy = self._arrange_plan[self._arrange_i]
        self._arrange_i += 1
        if kind == "submit":
            return "ACTION5", None, "ceax_arrange_submit"
        return "ACTION6", xy, f"ceax_arrange:{xy}"

    def _build_arrange_plan(
        self, objects: Sequence[Any]
    ) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
        items: List[Tuple[int, int, int, int]] = []
        for obj in objects:
            xy = _obj_centroid(obj)
            if xy is None:
                continue
            x, y = xy
            if not (0 <= x < 64 and 0 <= y < 64):
                continue
            area = _obj_area(obj)
            if area < 4 or area > 100:
                continue
            items.append((x, y, int(getattr(obj, "color", -1)), area))
        if len(items) < 4:
            return []

        pal = sorted(
            [i for i in items if i[1] >= 52 and 8 <= i[3] <= 40],
            key=lambda z: z[0],
        )
        slots = sorted(
            [i for i in items if 22 <= i[1] <= 40 and i[3] <= 20],
            key=lambda z: z[0],
        )
        top_raw = sorted(
            [i for i in items if i[1] <= 12 and 8 <= i[3] <= 40],
            key=lambda z: z[0],
        )
        pal_colors = {p[2] for p in pal}
        # Prefer reference chips whose color exists in the palette
        refs: List[Tuple[int, int, int, int]] = []
        seen_x = set()
        for t in sorted(top_raw, key=lambda z: (-z[3], z[0])):
            if t[2] not in pal_colors:
                continue
            xb = t[0] // 4
            if xb in seen_x:
                continue
            seen_x.add(xb)
            refs.append(t)
        refs.sort(key=lambda z: z[0])
        if self._arrange_variant % 2 == 1:
            refs = list(reversed(refs))
        elif self._arrange_variant % 3 == 2:
            # pair by sorted palette order instead of color match order
            refs = sorted(refs, key=lambda z: z[2])

        if not (pal and slots and refs):
            # weaker fallback: bottom→mid pairs without color match
            if pal and slots:
                plan: List[Tuple[str, Optional[Tuple[int, int]]]] = []
                for p, s in zip(pal, slots):
                    plan.append(("click", (p[0], p[1])))
                    plan.append(("click", (s[0], s[1])))
                plan.append(("submit", None))
                return plan
            return []

        plan = []
        remaining = list(pal)
        for i, ref in enumerate(refs):
            if i >= len(slots):
                break
            match = next((p for p in remaining if p[2] == ref[2]), None)
            if match is None:
                continue
            remaining = [p for p in remaining if p is not match]
            s = slots[i]
            plan.append(("click", (match[0], match[1])))
            plan.append(("click", (s[0], s[1])))
        if plan:
            plan.append(("submit", None))
            self.hyps.support(HypothesisKind.CLICK_EFFECT, amount=0.08)
        return plan

    # ---------- internals ----------
    def _commit(self, action: str, xy: Optional[Tuple[int, int]], key: Optional[Tuple]):
        self._last_action = action
        self._last_click = xy
        self._last_key = key

    def _choose_pull_click(self, grid) -> Optional[Tuple[int, int]]:
        """Click a short leash from agent toward goal (within attract radius ~8)."""
        roles = _find_pull_roles(grid)
        if roles is None and self._pull_agent and self._pull_goal:
            agent, goal = self._pull_agent, self._pull_goal
            dist = float(abs(agent[0] - goal[0]) + abs(agent[1] - goal[1]))
        elif roles is not None:
            agent, goal, dist = roles
            self._pull_agent, self._pull_goal, self._pull_dist = agent, goal, dist
        else:
            return None
        if dist < 3:
            return (
                max(1, min(62, int(goal[0]))),
                max(11, min(57, int(goal[1]))),
            )
        ax, ay = float(agent[0]), float(agent[1])
        gx, gy = float(goal[0]), float(goal[1])
        dx, dy = gx - ax, gy - ay
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        step = min(6.5, max(4.0, min(7.0, norm * 0.15)))
        cx = int(round(ax + dx / norm * step))
        cy = int(round(ay + dy / norm * step))
        cx = max(1, min(62, cx))
        cy = max(11, min(57, cy))
        return (cx, cy)

    def _pick_hybrid_mutate_click(
        self, objects: Sequence[Any], grid=None
    ) -> Optional[ClickTarget]:
        """Prefer compact on-screen mid-board blobs (toggleable platform nodes)."""
        cands = [
            t for t in self.targets.values()
            if t.trials < 2
            and t.key not in self._blocked_keys
            and 6 <= t.area <= 100
            and 4 <= t.xy[1] <= 55
            and 4 <= t.xy[0] <= 60
        ]
        if not cands and isinstance(grid, np.ndarray):
            # Fallback: rare mid colors as click points (visible Q-like nodes)
            play = grid[:58] if grid.shape[0] >= 58 else grid
            hist = Counter(int(c) for c in play.flatten().tolist())
            bg = hist.most_common(1)[0][0]
            for color, n in hist.items():
                if color == bg or color == 0 or not (8 <= n <= 120):
                    continue
                for cx, cy, sz in _cc_centroids(play, color, 4, 40):
                    if 4 <= cy <= 55:
                        key = ("hyb", color, cx // 3, cy // 3)
                        if key in self._blocked_keys:
                            continue
                        if key not in self.targets:
                            self.targets[key] = ClickTarget(
                                xy=(cx, cy), key=key, area=sz, color=color
                            )
                            self.targets[key].effect_sum = 0.4
                        cands.append(self.targets[key])
        if not cands:
            return None
        # Prefer higher on screen (smaller y) — climb targets
        cands.sort(key=lambda t: (t.trials, t.xy[1], -t.score))
        return cands[0]

    def _hybrid_walk_action(self, valid: Sequence[str], grid=None) -> str:
        """L/R choice: motion EMA, with optional bias toward rare upper goal speck."""
        h3 = self._action_effects.get("ACTION3") or []
        h4 = self._action_effects.get("ACTION4") or []
        e3 = sum(h3[-4:]) / max(1, len(h3[-4:])) if h3 else 0.05
        e4 = sum(h4[-4:]) / max(1, len(h4[-4:])) if h4 else 0.05
        if len(h3) < 2 and "ACTION3" in valid:
            return "ACTION3"
        if len(h4) < 2 and "ACTION4" in valid:
            return "ACTION4"
        # Bias toward rare goal-like speck (often above avatar)
        if isinstance(grid, np.ndarray):
            play = grid[:58] if grid.shape[0] >= 58 else grid
            hist = Counter(int(c) for c in play.flatten().tolist())
            bg = hist.most_common(1)[0][0]
            rare = [
                (c, n) for c, n in hist.items()
                if c != bg and c != 0 and 2 <= n <= 30
            ]
            rare.sort(key=lambda t: t[1])
            agent = None
            if self._kb_motion_pos:
                agent = self._kb_motion_pos
            if rare and agent is not None:
                c = rare[0][0]
                ys, xs = np.where(play == c)
                if len(xs):
                    gx, gy = int(xs.mean()), int(ys.mean())
                    if gx < agent[0] - 2 and "ACTION3" in valid:
                        return "ACTION3"
                    if gx > agent[0] + 2 and "ACTION4" in valid:
                        return "ACTION4"
        a = "ACTION4" if e4 >= e3 else "ACTION3"
        if a not in valid:
            a = "ACTION4" if "ACTION4" in valid else "ACTION3"
        return a

    def _seed_align_chrome_ends(self, objects: Sequence[Any]) -> None:
        """Preload far-end clicks on hollow chrome rings when marker gap is open."""
        if len(self._sticky_clicks) >= 2 or self._align_gap is None:
            return
        if self._align_gap < 6.0:
            return
        cands: List[Tuple[int, int]] = []
        for obj in objects:
            area = _obj_area(obj)
            if area < 16 or area > 100:
                continue
            pixels = getattr(obj, "pixels", None) or []
            bbox = getattr(obj, "bbox", None)
            if not bbox or len(bbox) != 4 or not pixels:
                continue
            bw = max(1, abs(bbox[2] - bbox[0]) + 1)
            bh = max(1, abs(bbox[3] - bbox[1]) + 1)
            fill = len(pixels) / float(bw * bh)
            if fill > 0.55:
                continue
            samples = _obj_sample_points(obj)
            # samples[0]=centroid (mid/noop); [1]=far end after frac order
            if len(samples) >= 2:
                xy = (int(samples[1][0]), int(samples[1][1]))
                if xy not in cands and 0 <= xy[0] < 64 and 0 <= xy[1] < 64:
                    cands.append(xy)
        if len(cands) >= 2:
            # Merge with any surviving single sticky end
            merged = list(self._sticky_clicks)
            for xy in cands:
                if xy not in merged:
                    merged.append(xy)
            self._sticky_clicks = merged[:4]
            self._sticky_budget = max(self._sticky_budget, 24)
            self._sticky_i = 0

    def _ingest_objects(self, objects: Sequence[Any]):
        for obj in objects:
            area = _obj_area(obj)
            if area <= 0 or area > 900:
                continue
            color = int(getattr(obj, "color", -1))
            base_key = _obj_key(obj)
            samples = _obj_sample_points(obj)
            pixels = getattr(obj, "pixels", None) or []
            bbox = getattr(obj, "bbox", None)
            fill = 1.0
            if bbox and len(bbox) == 4:
                bw = max(1, abs(bbox[2] - bbox[0]) + 1)
                bh = max(1, abs(bbox[3] - bbox[1]) + 1)
                fill = len(pixels) / float(bw * bh) if pixels else 1.0
            # Hollow chrome rings (slider frames) vs solid rods
            hollow = fill <= 0.55 and 16 <= area <= 100
            for si, xy in enumerate(samples):
                x, y = xy
                if not (0 <= x < 64 and 0 <= y < 64):
                    continue
                key = base_key if si == 0 else (color, base_key[1], "sub", si)
                if key in self._blocked_keys:
                    continue
                if key not in self.targets:
                    prior = 0.75 if color in self._preferred_colors else 0.55
                    if si > 0:
                        prior = max(prior, 0.72)
                    if hollow and si > 0:
                        prior = max(prior, 1.05)  # past-mid chrome ends first
                    elif hollow and si == 0:
                        prior = min(prior, 0.35)  # mid of chrome often exact noop
                    elif fill >= 0.75 and area >= 20:
                        prior = min(prior, 0.40)  # solid rods rarely clickable
                    self.targets[key] = ClickTarget(
                        xy=(x, y),
                        key=key,
                        area=area,
                        color=color,
                    )
                    if prior > 0.55:
                        self.targets[key].effect_sum = prior * 0.5
                    elif prior < 0.5:
                        self.targets[key].effect_sum = prior * 0.3
                else:
                    self.targets[key].xy = (x, y)

    def _top_targets(
        self, n: int, min_trials: int, min_score: float
    ) -> List[ClickTarget]:
        cands = [
            t for t in self.targets.values()
            if t.trials >= min_trials
            and t.score >= min_score
            and t.key not in self._blocked_keys
        ]
        cands.sort(key=lambda t: (t.score, -t.area), reverse=True)
        return cands[:n]

    def _next_experiment_click(self) -> Optional[Tuple[Tuple[int, int], Tuple]]:
        untried = [
            t for t in self.targets.values()
            if t.trials == 0 and t.key not in self._blocked_keys
        ]

        def sort_key(t: ClickTarget):
            # Highest prior / score first (chrome ends ingest at ~1.05)
            return (-t.score, t.trials, t.area)

        untried.sort(key=sort_key)
        if untried:
            t = untried[0]
            return t.xy, t.key
        low = [
            t for t in self.targets.values()
            if t.trials < 3 and t.key not in self._blocked_keys
        ]
        low.sort(key=lambda t: (t.trials, t.area))
        if low:
            t = low[0]
            jx = (self._step % 3) - 1
            jy = ((self._step // 3) % 3) - 1
            xy = (
                max(0, min(63, t.xy[0] + jx)),
                max(0, min(63, t.xy[1] + jy)),
            )
            return xy, t.key
        return None

    def _next_grid_click(self) -> Optional[Tuple[int, int]]:
        pts = []
        for step in (8, 6, 10, 4):
            for y in range(4, 60, step):
                for x in range(4, 60, step):
                    pts.append((x, y))
        pts.sort(key=lambda p: (p[0] - 32) ** 2 + (p[1] - 32) ** 2)
        if self._grid_i >= len(pts):
            self._grid_i = 0
            return None
        xy = pts[self._grid_i]
        self._grid_i += 1
        return xy

    def _rank_simple(self, actions: Sequence[str], grid_hash: str) -> List[str]:
        rows = []
        for a in actions:
            pred = self.model.predict(grid_hash, a) if grid_hash else None
            epistemic = 1.0 if pred is None else 0.0
            hist = self._action_effects.get(a) or []
            ema = sum(hist[-5:]) / len(hist[-5:]) if hist else 0.1
            recent = list(self._recent_actions).count(a)
            loop_pen = 0.35 * recent
            score = epistemic * 1.2 + ema - loop_pen
            rows.append((score, a))
        rows.sort(reverse=True)
        return [a for _, a in rows]

    def _choose_keyboard(
        self, valid: Sequence[str], grid_hash: str, grid=None
    ) -> str:
        if grid is not None:
            nav = self._lattice_nav(grid, valid)
            if nav:
                # Break sticky repeats when pose did not change
                if (
                    self._kb_prev_pos is not None
                    and self._kb_last == nav
                    and len(self._kb_blocked) > 0
                    and any(
                        b[0] == self._kb_prev_pos[0]
                        and b[1] == self._kb_prev_pos[1]
                        and b[2] == nav
                        for b in self._kb_blocked
                    )
                ):
                    alts = [a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if a in valid and a != nav]
                    for a in alts:
                        if (self._kb_prev_pos[0], self._kb_prev_pos[1], a) not in self._kb_blocked:
                            self._kb_last = a
                            return a
                return nav
        dirs = [a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if a in valid]
        if not dirs:
            return valid[0]
        combos = [
            ("ACTION1", "ACTION4"),
            ("ACTION2", "ACTION3"),
            ("ACTION1", "ACTION3"),
            ("ACTION2", "ACTION4"),
        ]
        if self._step % 11 < 4:
            pair = combos[self._combo_phase % len(combos)]
            self._combo_phase += 1
            for a in pair:
                if a in dirs:
                    return a
        ranked = self._rank_simple(dirs, grid_hash)
        last = self._last_action
        rev = {
            "ACTION1": "ACTION2", "ACTION2": "ACTION1",
            "ACTION3": "ACTION4", "ACTION4": "ACTION3",
        }.get(last or "", "")
        for a in ranked:
            if a != rev:
                return a
        return ranked[0]

    def _lattice_nav(
        self, grid: np.ndarray, valid: Sequence[str], _depth: int = 0
    ) -> Optional[str]:
        if not isinstance(grid, np.ndarray) or grid.size == 0:
            return None
        if _depth > 1:
            return None
        play = grid[:58, :] if grid.shape[0] >= 58 else grid
        H, W = play.shape
        hist = Counter(int(c) for c in play.flatten())
        bg = hist.most_common(1)[0][0]

        # Replan roles quickly when decoy/NPC selected (no movement / many blocked edges)
        # Verified avatars: only flush blocked edges periodically (game step ~6px)
        if self._kb_verified and len(self._kb_blocked) >= 4 and self._step % 12 == 0:
            # Keep edges that failed at the current pose
            keep = {b for b in self._kb_blocked if b[0] == self._kb_prev_pos[0] and b[1] == self._kb_prev_pos[1]} if self._kb_prev_pos else set()
            self._kb_blocked.clear()
            self._kb_blocked |= keep
        stuck = (
            self._kb_prev_pos is not None
            and self._kb_last is not None
            and len(self._kb_blocked) >= 3
            and not self._kb_verified
        )
        if stuck and (self._step % 4 == 0 or len(self._kb_blocked) >= 5):
            if self._kb_player_colors:
                self._kb_failed_pc.add(frozenset(self._kb_player_colors))
            self._kb_blocked.clear()
            self._kb_player_colors = []
            self._kb_corridor = None
            self._kb_goal = None
            self._kb_step = 3 if self._kb_step == 6 else 6
            self._kb_verified = False
            self._kb_motion_pos = None
            self._kb_no_move = 0

        if not self._kb_player_colors or self._kb_corridor is None:
            rares = []
            for c, n in hist.items():
                if c == bg or not (1 <= n <= 24):
                    continue
                ys, xs = np.where(play == c)
                if len(xs) == 0:
                    continue
                cy = float(ys.mean())
                if cy >= 55:
                    continue
                rares.append((c, float(xs.mean()), cy, n))
            seeds = [r for r in rares if 3 <= r[3] <= 16] or rares
            corr_colors = {
                c for c, n in hist.items()
                if c != bg and 40 <= n <= 800
            }
            def _seed_key(r):
                # Prefer agent blobs with tiny companion, corridor contact, and high corridor degree
                tiny = sum(
                    1 for t in rares
                    if t[3] <= 2
                    and (t[1] - r[1]) ** 2 + (t[2] - r[2]) ** 2 <= 36
                )
                ys, xs = np.where(play == r[0])
                degree = 0
                on_corr = 0
                for x, y in zip(xs.tolist()[:32], ys.tolist()[:32]):
                    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny < H and int(play[ny, nx]) in corr_colors:
                            degree += 1
                            on_corr = 1
                outer_tiny = 0
                for t in rares:
                    if t[3] > 2:
                        continue
                    if (t[1] - r[1]) ** 2 + (t[2] - r[2]) ** 2 > 36:
                        continue
                    if xs.size and (abs(t[1] - xs.mean()) + abs(t[2] - ys.mean())) >= 0.8:
                        outer_tiny = 1
                hint_hit = 1 if r[0] in self._kb_hint_pc else 0
                return (-hint_hit, -tiny, -outer_tiny, -on_corr, -degree, abs(r[3] - 8), r[3])
            seeds.sort(key=_seed_key)
            # If prior-level player colors still exist, lock onto them first
            hint_present = [c for c in self._kb_hint_pc if any(r[0] == c for r in rares)]
            if hint_present and frozenset(hint_present) not in self._kb_failed_pc:
                self._kb_player_colors = hint_present
                seed = next((r for r in rares if r[0] == hint_present[0]), None)
                if seed is None and seeds:
                    seed = seeds[0]
                far = [r for r in rares if r[0] not in self._kb_player_colors]
                if far and seed is not None:
                    far.sort(
                        key=lambda r: -(
                            (r[1] - seed[1]) ** 2 + (r[2] - seed[2]) ** 2
                        )
                    )
                    sized = [
                        r for r in far
                        if 4 <= r[3] <= 20 and r[0] not in self._kb_hint_pc
                    ]
                    prefer = sized or [r for r in far if r[0] not in self._kb_hint_pc] or far
                    self._kb_goal = prefer[0][0]
            elif seeds:
                seed = seeds[0]
                near = [
                    r for r in rares
                    if (r[1] - seed[1]) ** 2 + (r[2] - seed[2]) ** 2 <= 64
                ]
                # Keep tiny companions even if slightly farther
                tinies = [
                    r for r in rares
                    if r[3] <= 2
                    and (r[1] - seed[1]) ** 2 + (r[2] - seed[2]) ** 2 <= 100
                ]
                cand = list({r[0] for r in (near + tinies) if r[0] != bg and r[0] != 0}) or (
                    [seed[0]] if seed[0] != bg else []
                )
                # try alternate seeds if this role-set already failed (stuck)
                if not cand or frozenset(cand) in self._kb_failed_pc:
                    picked = False
                    for alt in seeds[1:]:
                        near2 = [
                            r for r in rares
                            if (r[1] - alt[1]) ** 2 + (r[2] - alt[2]) ** 2 <= 64
                        ]
                        tinies2 = [
                            r for r in rares
                            if r[3] <= 2
                            and (r[1] - alt[1]) ** 2 + (r[2] - alt[2]) ** 2 <= 100
                        ]
                        cand2 = list(
                            {r[0] for r in (near2 + tinies2) if r[0] != bg and r[0] != 0}
                        ) or ([alt[0]] if alt[0] != bg else [])
                        if cand2 and frozenset(cand2) not in self._kb_failed_pc:
                            seed = alt
                            near, tinies, cand = near2, tinies2, cand2
                            picked = True
                            break
                    if not picked and self._kb_failed_pc:
                        # Escape hatch: all role-sets banned → clear and retry best seed
                        self._kb_failed_pc.clear()
                        cand = list(
                            {r[0] for r in (near + tinies) if r[0] != bg and r[0] != 0}
                        ) or ([seed[0]] if seed[0] != bg else [])
                if not cand:
                    return None
                self._kb_player_colors = cand
                far = [r for r in rares if r[0] not in self._kb_player_colors]
                if far:
                    far.sort(
                        key=lambda r: -(
                            (r[1] - seed[1]) ** 2 + (r[2] - seed[2]) ** 2
                        )
                    )
                    sized = [r for r in far if 4 <= r[3] <= 20]
                    self._kb_goal = (sized[0] if sized else far[0])[0]
            if self._kb_corridor is None:
                mids = [
                    c for c, n in hist.items()
                    if c != bg
                    and c != 0
                    and c not in self._kb_player_colors
                    and c != self._kb_goal
                    and 40 <= n <= 800
                ]
                self._kb_corridor = mids[0] if mids else None
            if self._kb_goal is None:
                goals = [
                    c for c, n in hist.items()
                    if c != bg
                    and c not in self._kb_player_colors
                    and c != self._kb_corridor
                    and 4 <= n <= 40
                ]
                self._kb_goal = goals[0] if goals else None

        pc = self._kb_player_colors
        if not pc:
            return None
        mask = np.zeros_like(play, dtype=bool)
        for c in pc:
            mask |= play == c
        ys, xs = np.where(mask)
        if len(xs) == 0:
            self._kb_player_colors = []
            return None
        px, py = int(round(xs.mean())), int(round(ys.mean()))
        # Motion hint only if near a real player-colored pixel (never ghost-nav on empty cells)
        if self._kb_motion_pos is not None:
            mx, my = self._kb_motion_pos
            best = None
            best_d = 10**9
            for x, y in zip(xs.tolist(), ys.tolist()):
                d = (x - mx) ** 2 + (y - my) ** 2
                if d < best_d:
                    best_d = d
                    best = (x, y)
            if best is not None and best_d <= 64:
                px, py = best
            else:
                self._kb_motion_pos = None

        if self._kb_prev_pos == (px, py) and self._kb_last:
            self._kb_blocked.add((self._kb_prev_pos[0], self._kb_prev_pos[1], self._kb_last))
        # Death/warp teleport — drop committed plan
        if self._kb_prev_pos is not None:
            if abs(px - self._kb_prev_pos[0]) + abs(py - self._kb_prev_pos[1]) >= 18:
                self._kb_plan = []
                self._kb_blocked.clear()
                self._kb_dist_hist.clear()
        revisiting = (px, py) in self._kb_recent_pos
        self._kb_recent_pos.append((px, py))
        self._kb_prev_pos = (px, py)

        corridor = self._kb_corridor
        goal_c = self._kb_goal
        if corridor is None:
            return None

        if goal_c is not None:
            gy, gx = np.where(play == goal_c)
            if len(gx):
                tx, ty = int(gx.mean()), int(gy.mean())
            else:
                tx, ty = px, py
        else:
            tx, ty = px, py

        dist_now = abs(px - tx) + abs(py - ty)
        best_dist = min(self._kb_dist_hist) if self._kb_dist_hist else dist_now
        self._kb_dist_hist.append(dist_now)
        made_progress = dist_now < best_dist
        if made_progress:
            # Drop stale blocks — a better route may reuse cleared cells
            self._kb_blocked = {
                b for b in self._kb_blocked
                if abs(b[0] - px) + abs(b[1] - py) > 12
            }
        no_progress = (
            len(self._kb_dist_hist) >= 6
            and self._kb_dist_hist[-1] >= min(self._kb_dist_hist)
            and self._kb_dist_hist[-1] >= self._kb_dist_hist[0] - 2
        )
        if revisiting and no_progress and self._kb_last and self._kb_prev_pos is not None:
            # prev_pos already overwritten — use recent history's prior entry
            if len(self._kb_recent_pos) >= 2:
                ox, oy = self._kb_recent_pos[-2]
                self._kb_blocked.add((ox, oy, self._kb_last))
        step = max(1, int(self._kb_step or 6))
        if no_progress and _depth == 0 and self._step % 5 == 0:
            step = 3 if step == 6 else 6
            self._kb_step = step
            self._kb_plan = []

        # Commit to an existing plan to avoid per-frame thrashing
        if self._kb_plan and _depth == 0:
            # Only abort plan when clearly stuck on same pose
            if not (self._kb_prev_pos == (px, py) and len(self._kb_blocked) >= 2):
                name = self._kb_plan.pop(0)
                self._kb_last = name
                return name
            self._kb_plan = []

        moves = [
            (0, -step, "ACTION1"),
            (0, step, "ACTION2"),
            (-step, 0, "ACTION3"),
            (step, 0, "ACTION4"),
        ]
        moves = [m for m in moves if m[2] in valid]
        if not moves:
            return None

        # Walkable: corridor + open (0) cells + player + goal halo
        walk = set()
        ys_c, xs_c = np.where((play == corridor) | (play == 0))
        for x, y in zip(xs_c.tolist(), ys_c.tolist()):
            walk.add((x, y))
        for x, y in zip(xs.tolist(), ys.tolist()):
            walk.add((x, y))
        walk.add((px, py))
        if goal_c is not None:
            gy2, gx2 = np.where(play == goal_c)
            for x, y in zip(gx2.tolist(), gy2.tolist()):
                for dx, dy in (
                    (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                    (2, 0), (-2, 0), (0, 2), (0, -2),
                ):
                    walk.add((x + dx, y + dy))

        # Near-goal force only when mid-hop is not a solid wall
        man = abs(px - tx) + abs(py - ty)
        if 0 < man <= step + 1:
            if abs(tx - px) >= abs(ty - py):
                name = "ACTION4" if tx > px else "ACTION3"
                dx, dy = (step if name == "ACTION4" else -step), 0
            else:
                name = "ACTION2" if ty > py else "ACTION1"
                dx, dy = 0, (step if name == "ACTION2" else -step)
            hx, hy = px + dx // 2, py + dy // 2
            nx, ny = px + dx, py + dy
            wall = bg
            mid_ok = (
                0 <= hx < W and 0 <= hy < H
                and int(play[hy, hx]) != wall
            )
            if (
                name in valid
                and mid_ok
                and (px, py, name) not in self._kb_blocked
                and 0 <= nx < W and 0 <= ny < H
            ):
                self._kb_last = name
                return name

        # Hazards: other agent-sized rare blobs (decoys). Side-on contact kills;
        # approaching on the orthogonal axis (from below/above) clears them.
        hazards: set = set()
        hazard_colors = set()
        for c, n in hist.items():
            if c in (bg, corridor, 0) or c == goal_c or c in set(pc):
                continue
            if not (3 <= n <= 16):
                continue
            ys_h, xs_h = np.where(play == c)
            if len(xs_h) == 0:
                continue
            # Skip if far from corridor ribbon
            cy = float(ys_h.mean())
            if cy < 10 or cy > 50:
                continue
            hazard_colors.add(c)
            for x, y in zip(xs_h.tolist(), ys_h.tolist()):
                # Wide horizontal aura: same-lane approach within one lattice step kills
                for ax in range(-6, 7):
                    for ay in range(-2, 3):
                        hazards.add((x + ax, y + ay))

        def can(x: int, y: int, dx: int, dy: int, name: str, cleared: bool) -> bool:
            if (x, y, name) in self._kb_blocked:
                return False
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                return False
            samples = []
            sample_xy = []
            for t in (0.25, 0.5, 0.75, 1.0):
                sx = int(round(x + dx * t))
                sy = int(round(y + dy * t))
                if not (0 <= sx < W and 0 <= sy < H):
                    return False
                samples.append(int(play[sy, sx]))
                sample_xy.append((sx, sy))
            dst = samples[-1]
            wall_hits = sum(
                1 for p in samples if p == bg and p not in set(pc) and p != goal_c
            )
            if wall_hits >= 3:
                return False
            if dst == bg and dst not in set(pc) and dst != goal_c and dst not in (corridor, 0):
                return False

            hit_core = any(p in hazard_colors for p in samples)
            hit_aura = any(p in sample_xy for p in hazards) or hit_core
            clearing = abs(dy) > abs(dx)
            if not cleared:
                if hit_aura and not hit_core:
                    # Death-zone gap beside a decoy — never enter sideways or vertically
                    return False
                if hit_core and not clearing:
                    return False

            if goal_c is not None and (dst == goal_c or goal_c in samples):
                return True
            if dst in (corridor, 0) or any(p in (corridor, 0) for p in samples[:3]):
                return True
            if dst in set(pc) or (nx, ny) in walk:
                return True
            if hit_core and clearing:
                return True
            return False

        # BFS carries a 'cleared' flag once a vertical hazard-clear hop is taken
        q = deque([(px, py, False)])
        parent: Dict[Tuple[int, int, bool], Optional[Tuple[Tuple[int, int, bool], str]]] = {
            (px, py, False): None
        }
        found = None
        while q:
            cx, cy, cleared = q.popleft()
            if abs(cx - tx) + abs(cy - ty) <= max(2, step // 2):
                found = (cx, cy, cleared)
                break
            for dx, dy, name in moves:
                if not can(cx, cy, dx, dy, name, cleared):
                    continue
                nx, ny = cx + dx, cy + dy
                # Core clear only when the hop actually touches hazard colors
                samples_c = [
                    int(play[int(round(cy + dy * t)), int(round(cx + dx * t))])
                    for t in (0.5, 1.0)
                    if 0 <= int(round(cy + dy * t)) < H
                    and 0 <= int(round(cx + dx * t)) < W
                ]
                hit_core = (not cleared) and any(p in hazard_colors for p in samples_c)
                ncleared = bool(cleared or (hit_core and abs(dy) > abs(dx)))
                key = (nx, ny, ncleared)
                if key in parent:
                    continue
                parent[key] = ((cx, cy, cleared), name)
                q.append(key)

        name = None
        if found and parent.get(found) is not None:
            cur = found
            plan: List[str] = []
            while parent[cur] is not None:
                prevc, actn = parent[cur]  # type: ignore
                plan.append(actn)
                cur = prevc
            plan.reverse()
            if plan:
                self._kb_plan = plan[1:]
                name = plan[0]
        elif self._kb_plan:
            name = self._kb_plan.pop(0)
        if name is None:
            for dx, dy, n in sorted(
                moves, key=lambda m: abs(px + m[0] - tx) + abs(py + m[1] - ty)
            ):
                if can(px, py, dx, dy, n, False):
                    name = n
                    break
        if name is None:
            if _depth == 0:
                alt = 3 if self._kb_step == 6 else 6
                prev = self._kb_step
                self._kb_step = alt
                out = self._lattice_nav(grid, valid, _depth=1)
                if out is None:
                    self._kb_step = prev
                return out
            return None
        if no_progress:
            better = []
            for dx, dy, n in moves:
                if (px, py, n) in self._kb_blocked:
                    continue
                nd = abs(px + dx - tx) + abs(py + dy - ty)
                if nd < dist_now and can(px, py, dx, dy, n, False):
                    better.append((nd, n))
            if better:
                better.sort()
                name = better[0][1]
                self._kb_plan = []
            else:
                for dx, dy, n in moves:
                    if n != self._kb_last and can(px, py, dx, dy, n, False):
                        name = n
                        self._kb_plan = []
                        break
        self._kb_last = name
        return name

    def _should_try_submit(self, has_click: bool) -> bool:
        if self._arrange_mode and self._arrange_plan:
            return False  # arrange plan owns submits
        if "submit_after_arrange" in self.skills.skills and self._effective_clicks >= 2:
            if self._clicks_since_submit >= 3:
                return True
        colored = {
            t.color for t in self.targets.values()
            if t.trials > 0 and t.score >= 0.08
        }
        if len(colored) >= 3 and self._clicks_since_submit >= 2:
            return True
        if self._effective_clicks >= 3 and self._clicks_since_submit >= 3:
            return True
        if self._effective_clicks >= 1 and self._clicks_since_submit >= 5:
            return True
        if has_click and self._clicks_since_submit >= 8 and self._step > 5:
            return True
        return False

    def _should_undo(self) -> bool:
        # Pull leash working — never undo mid-attract (su15 thrash)
        if self._pull_mode and self._pull_hits > 0:
            return False
        # Arrange: undo after failed submit (once), then rebuild
        if self._arrange_mode:
            if self._need_arrange_undo:
                self._need_arrange_undo = False
                return True
            if self._last_action == "ACTION5":
                hist = self._action_effects.get("ACTION5") or []
                if hist and hist[-1] < 0.1:
                    return True
            return False
        # Click+undo: reverse noop/revert then try sibling target
        if self._need_click_undo:
            self._need_click_undo = False
            self._click_fail_streak = 0
            return True
        if self._last_action == "ACTION5":
            hist = self._action_effects.get("ACTION5") or []
            if hist and hist[-1] < 0.1:
                return True
        if self._last_action == "ACTION6":
            hist = self._action_effects.get("ACTION6") or []
            if len(hist) >= 4 and all(h < 0.04 for h in hist[-4:]):
                return self._step % 8 == 0
        return False

    def snapshot(self) -> dict:
        return {
            "metrics": dict(self.metrics),
            "targets": len(self.targets),
            "skills": len(self.skills.skills),
            "effective_clicks": self._effective_clicks,
            "submit_trials": self._submit_trials,
            "arrange_plan": len(self._arrange_plan),
            "kb_step": self._kb_step,
            "kb_pc": list(self._kb_player_colors),
            "kb_goal": self._kb_goal,
            "kb_corr": self._kb_corridor,
            "kb_failed": [list(x) for x in self._kb_failed_pc],
            "kb_blocked": len(self._kb_blocked),
            "kb_pos": self._kb_prev_pos,
            "hypotheses": [(h.name, round(h.confidence, 2)) for h in self.hyps.ranked()],
            "best_targets": [
                {"xy": t.xy, "color": t.color, "score": round(t.score, 3), "trials": t.trials}
                for t in sorted(self.targets.values(), key=lambda z: z.score, reverse=True)[:5]
            ],
        }
