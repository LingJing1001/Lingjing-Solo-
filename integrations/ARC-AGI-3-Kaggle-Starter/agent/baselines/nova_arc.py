"""Nova-ARC Agent for ARC-AGI-3 (Kaggle-safe, offline-first).

Architecture (mapped from Nova industrial stack → ARC black-box games):
  L1 Explore   — InfoGainRouter (entropy / mutual-information action scoring)
  L2 Cause     — CausalEngine SCM-lite (do-style intervention vs mere correlation)
  L3 Memory    — HypergraphStore (multi-way state constraints, not only pairwise)
  L4 Plan      — InternalWorldModel + MCTS-UCT over learned transitions
  L5 Reflect   — prediction vs observation → targeted re-probe
  L6 Meta      — reusable explore→abstract→verify workflow priors
  Optional     — LLM advisor only when keys/network exist (disabled on Kaggle)

Class name MUST remain `MyAgent` for the Kaggle notebook contract.
Deps: stdlib + arcengine + agents.agent only (numpy optional, not required).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import ssl
import time
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path
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
    return (
        sum(t[0] for t in ch) / len(ch),
        sum(t[1] for t in ch) / len(ch),
        len(ch),
    )


def color_entropy(grid: list[list[int]]) -> float:
    if not grid:
        return 0.0
    cnt = Counter(c for row in grid for c in row)
    n = sum(cnt.values()) or 1
    ent = 0.0
    for v in cnt.values():
        p = v / n
        ent -= p * math.log(p + 1e-12)
    return ent


def compact_signature(grid: list[list[int]], bins: int = 8) -> tuple[int, ...]:
    """Low-dim manifold-ish signature: coarse color histogram + entropy bucket."""
    if not grid:
        return (0,)
    hist = [0] * 16
    for row in grid:
        for c in row:
            hist[int(c) & 15] += 1
    total = sum(hist) or 1
    step = max(1, 16 // bins)
    sig = []
    for i in range(0, 16, step):
        sig.append(sum(hist[i : i + step]) * 10 // total)
    sig.append(int(color_entropy(grid) * 10))
    return tuple(sig)


def compact_grid_text(frame: FrameData, max_side: int = 16) -> str:
    """Downsample grid to short hex text for LLM context."""
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


def connected_blobs(
    grid: list[list[int]], bg: int, max_blobs: int = 36
) -> list[dict[str, Any]]:
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
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            blobs.append(
                {
                    "color": color,
                    "size": len(cells),
                    "cx": sum(xs) / len(xs),
                    "cy": sum(ys) / len(ys),
                }
            )
            if len(blobs) >= max_blobs:
                return blobs
    blobs.sort(key=lambda b: b["size"])
    return blobs


def salient_clicks(frame: FrameData, limit: int = 40) -> list[tuple[int, int]]:
    grid = top_grid(frame)
    if not grid:
        return [(32, 32)]
    h, w = len(grid), len(grid[0])
    bg = guess_bg(grid)
    hist = Counter(c for row in grid for c in row)
    rare = [c for c, _ in sorted(hist.items(), key=lambda kv: kv[1]) if c != bg]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(x: float, y: float) -> None:
        xi = max(0, min(w - 1, min(63, int(x))))
        yi = max(0, min(h - 1, min(63, int(y))))
        if (xi, yi) not in seen:
            seen.add((xi, yi))
            out.append((xi, yi))

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
    for p in ((x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) // 2, (y0 + y1) // 2)):
        add(*p)
    return out[:limit] or [(w // 2, h // 2)]


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
# Action key + Hypergraph + Causal + InfoGain + MCTS
# ═══════════════════════════════════════════════════════════════════════════

class ActionKey:
    __slots__ = ("name", "x", "y")

    def __init__(self, name: str, x: Optional[int] = None, y: Optional[int] = None):
        self.name = name
        self.x = x
        self.y = y

    def fp(self) -> str:
        return self.name if self.x is None else f"{self.name}:{self.x},{self.y}"


class SymbolicRule:
    """Executable causal axiom: ordered action sequence → predicted effect."""

    __slots__ = ("kind", "seq", "support", "success", "effect", "confidence")

    def __init__(
        self,
        kind: str,
        seq: tuple[str, ...],
        support: int,
        success: int,
        effect: str,
    ) -> None:
        self.kind = kind
        self.seq = seq
        self.support = support
        self.success = success
        self.effect = effect  # progress | change | unlock
        self.confidence = success / max(support, 1)

    def score(self) -> float:
        # Prefer short high-confidence progress rules
        length_pen = 0.15 * max(0, len(self.seq) - 2)
        bonus = 2.5 if self.effect == "progress" else 1.0
        return bonus * self.confidence * math.log(1.0 + self.success) - length_pen

    def to_axiom(self) -> str:
        body = ">>".join(self.seq)
        return f"{self.effect.upper()}<-do([{body}]) conf={self.confidence:.2f} n={self.success}"


class HypergraphStore:
    """State nodes + ordered combo hyperedges for multi-way constraints."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.hyperedges: dict[str, frozenset[str]] = {}
        self.edge_weight: Counter = Counter()
        self.co_occur: Counter = Counter()
        # ordered combo: action-tuple -> counts (preserves button order)
        self.combo_edges: Counter = Counter()
        self.combo_effect: dict[tuple[str, ...], Counter] = defaultdict(Counter)

    def upsert_state(self, sid: str, meta: dict[str, Any]) -> None:
        self.nodes[sid] = meta

    def add_hyperedge(self, members: list[str], tag: str) -> None:
        key = frozenset(members)
        if len(key) < 2:
            return
        eid = hashlib.blake2b(repr(sorted(key)).encode(), digest_size=8).hexdigest()
        self.hyperedges[eid] = key
        self.edge_weight[eid] += 1
        mem = list(key)
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                a, b = sorted((mem[i], mem[j]))
                self.co_occur[(a, b, tag)] += 1

    def add_ordered_combo(self, seq: tuple[str, ...], effect: str) -> None:
        if len(seq) < 2:
            return
        self.combo_edges[seq] += 1
        self.combo_effect[seq][effect] += 1
        # also unordered hyperedge for co-occurrence graph
        self.add_hyperedge(list(seq) + [f"eff:{effect}"], "combo")

    def relatedness(self, a: str, b: str) -> float:
        x, y = sorted((a, b))
        return float(self.co_occur[(x, y, "seq")] + self.co_occur[(x, y, "effect")])


class CausalEngine:
    """SCM-lite + combo mining → executable SymbolicRule axioms.

    Single-action interventional stats (do(A)) plus n-gram combos that
    distinguish spurious co-occurrence from necessary ordered preconditions.
    """

    def __init__(self) -> None:
        self.trials: dict[str, float] = defaultdict(float)
        self.change: dict[str, float] = defaultdict(float)
        self.progress: dict[str, float] = defaultdict(float)
        self.noop: dict[str, float] = defaultdict(float)
        self.self_loop: dict[str, float] = defaultdict(float)
        self.next_sig: dict[str, Counter] = defaultdict(Counter)
        self.axioms: list[str] = []
        self.rules: list[SymbolicRule] = []
        self._act_window: deque = deque(maxlen=8)
        self.combo_try: Counter = Counter()
        self.combo_hit: Counter = Counter()

    def observe(
        self,
        action: str,
        before_h: str,
        after_h: str,
        n_changed: int,
        progressed: bool,
        before_sig: tuple[int, ...],
        after_sig: tuple[int, ...],
        hyper: Optional[HypergraphStore] = None,
    ) -> None:
        self.trials[action] += 1.0
        effect = "noop"
        if progressed:
            effect = "progress"
            self.progress[action] += 1.0
            if n_changed > 0:
                self.change[action] += 1.0
            self.axioms.append(f"PROGRESS<-do({action})")
            self.axioms = self.axioms[-48:]
        elif n_changed > 0:
            effect = "change"
            self.change[action] += 1.0
        if before_h == after_h and not progressed:
            self.noop[action] += 1.0
            self.self_loop[action] += 1.0

        self.next_sig[action][(before_sig, after_sig)] += 1
        self._act_window.append(action)
        # Mine ordered combos of length 2..4 ending at this action
        win = list(self._act_window)
        for L in (2, 3, 4):
            if len(win) < L:
                continue
            seq = tuple(win[-L:])
            self.combo_try[seq] += 1
            if effect in ("progress", "change"):
                self.combo_hit[seq] += 1
                if hyper is not None:
                    hyper.add_ordered_combo(seq, effect)
        if effect in ("progress", "change"):
            self._remine_rules()

    def _remine_rules(self) -> None:
        """Promote high-support combos / single actions into executable rules."""
        candidates: list[SymbolicRule] = []
        # Combo rules: require support≥2 and success rate above baseline
        for seq, trials in self.combo_try.items():
            hits = self.combo_hit[seq]
            if trials < 2 or hits < 1:
                continue
            conf = hits / trials
            # Filter spurious: last action alone already explains most hits
            last = seq[-1]
            alone = self.p_change(last) + self.p_progress(last)
            # Keep if combo beats single-action baseline OR is a progress combo
            is_progressish = hits >= 1 and conf >= 0.35
            if conf + 0.08 < alone and not (
                self.progress[last] > 0 and len(seq) >= 2 and conf >= 0.5
            ):
                if not is_progressish or trials < 3:
                    continue
            effect = "progress" if self.progress[last] > 0 and conf >= 0.4 else "change"
            # Prefer progress if any progress observed on suffix
            if any(self.progress[a] > 0 for a in seq) and conf >= 0.3:
                effect = "progress"
            candidates.append(SymbolicRule("combo", seq, trials, hits, effect))

        # Single-action progress/change axioms as length-1 rules
        for a, t in self.trials.items():
            if t < 2:
                continue
            if self.progress[a] > 0:
                candidates.append(
                    SymbolicRule(
                        "single",
                        (a,),
                        int(t),
                        int(self.progress[a]),
                        "progress",
                    )
                )
            elif self.change[a] / t >= 0.45 and self.noop[a] / t < 0.4:
                candidates.append(
                    SymbolicRule(
                        "single",
                        (a,),
                        int(t),
                        int(self.change[a]),
                        "change",
                    )
                )

        candidates.sort(key=lambda r: r.score(), reverse=True)
        # Dedup by sequence
        seen: set[tuple[str, ...]] = set()
        rules: list[SymbolicRule] = []
        for r in candidates:
            if r.seq in seen:
                continue
            seen.add(r.seq)
            rules.append(r)
            if len(rules) >= 16:
                break
        self.rules = rules
        # Mirror top rules into axiom log
        for r in rules[:6]:
            ax = r.to_axiom()
            if ax not in self.axioms:
                self.axioms.append(ax)
        self.axioms = self.axioms[-48:]

    def best_rule(
        self, legal_names: set[str], min_conf: float = 0.35
    ) -> Optional[SymbolicRule]:
        for r in self.rules:
            if r.confidence < min_conf:
                continue
            if all(a in legal_names for a in r.seq):
                return r
        return None

    def p_change(self, action: str) -> float:
        t = max(self.trials[action], 1.0)
        return self.change[action] / t

    def p_progress(self, action: str) -> float:
        t = max(self.trials[action], 1.0)
        return self.progress[action] / t

    def p_noop(self, action: str) -> float:
        t = max(self.trials[action], 1.0)
        return self.noop[action] / t

    def causal_score(self, action: str) -> float:
        base = (
            4.0 * self.p_progress(action)
            + 1.5 * self.p_change(action)
            - 2.0 * self.p_noop(action)
            - 1.0 * (self.self_loop[action] / max(self.trials[action], 1.0))
        )
        # Boost actions that appear in high-scoring combo rules
        for r in self.rules[:8]:
            if action in r.seq:
                base += 0.35 * r.score()
                break
        return base

    def transfer_surprise(self, action: str, before_sig: tuple[int, ...]) -> float:
        c = self.next_sig[action]
        if not c:
            return 1.5
        sub = Counter()
        total = 0
        for (bs, as_), n in c.items():
            if bs == before_sig:
                sub[as_] += n
                total += n
        if total == 0:
            return 1.2
        ent = 0.0
        for n in sub.values():
            p = n / total
            ent -= p * math.log(p + 1e-12)
        return ent

    def rule_summaries(self, n: int = 8) -> list[str]:
        return [r.to_axiom() for r in self.rules[:n]]


class InfoGainRouter:
    """Score candidate actions by expected information gain + causal prior."""

    def __init__(self, causal: CausalEngine) -> None:
        self.causal = causal
        self.pair_visits: Counter = Counter()  # (state, action)
        self.state_visits: Counter = Counter()
        self.gain_history: deque = deque(maxlen=40)
        self.explore_stop = False

    def note_visit(self, state: str, action: str) -> None:
        self.pair_visits[(state, action)] += 1
        self.state_visits[state] += 1

    def expected_gain(
        self,
        state: str,
        action: str,
        before_sig: tuple[int, ...],
        known_next: Optional[str],
        next_visits: int,
    ) -> float:
        # Kakeya-inspired coverage prior: prefer under-touched (state,action) rays
        coverage = 2.8 / (1.0 + self.pair_visits[(state, action)])
        # Mutual-information proxy: causal transfer surprise
        mi = self.causal.transfer_surprise(action, before_sig)
        # Novelty of predicted next state
        novelty = 1.8 / (1.0 + next_visits) if known_next else 2.0
        causal = self.causal.causal_score(action)
        # Penalize chronic noops
        score = coverage + 1.1 * mi + novelty + 0.9 * causal
        if known_next == state:
            score -= 3.0
        return score

    def record_realized_gain(self, before_ent: float, after_ent: float, new_state: bool) -> None:
        g = abs(after_ent - before_ent) + (0.5 if new_state else 0.0)
        self.gain_history.append(g)
        if len(self.gain_history) >= 12:
            recent = list(self.gain_history)[-8:]
            if sum(recent) / len(recent) < 0.05:
                self.explore_stop = True

    def reset_level(self) -> None:
        self.explore_stop = False
        self.pair_visits = Counter()


class InternalWorldModel:
    """Digital twin: learned deterministic-ish transitions for planning."""

    def __init__(self) -> None:
        self.graph: dict[str, dict[str, str]] = defaultdict(dict)
        self.reward: dict[str, float] = defaultdict(float)  # state value proxy
        self.move_vec: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        self._move_samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.player_xy: Optional[tuple[float, float]] = None
        self.goals: list[tuple[float, float]] = []
        self.seek_target: Optional[tuple[float, float]] = None
        self.seek_ttl: int = 0

    def learn_transition(self, s: str, a: str, sp: str, progressed: bool) -> None:
        self.graph[s][a] = sp
        if progressed:
            self.reward[sp] += 5.0
            self.reward[s] += 1.0
        elif s != sp:
            self.reward[sp] += 0.05

    def bfs_unexplored(self, start: str, depth: int = 8) -> Optional[str]:
        """First action toward a less-visited / frontier state (Aether heritage)."""
        q: deque = deque([(start, [])])
        seen = {start}
        while q:
            node, path = q.popleft()
            if len(path) >= depth:
                continue
            edges = self.graph.get(node, {})
            for aname, nxt in edges.items():
                if nxt not in seen:
                    if self.reward[nxt] > 0.5 or len(path) == 0:
                        return path[0] if path else aname
                    seen.add(nxt)
                    q.append((nxt, path + [aname]))
            if path and len(edges) < 3:
                return path[0]
        return None

    def update_motion(
        self, action: str, before_g: list[list[int]], after_g: list[list[int]]
    ) -> None:
        cen = change_centroid(before_g, after_g)
        if not cen:
            return
        cx, cy, _ = cen
        if self.player_xy is not None:
            dx, dy = cx - self.player_xy[0], cy - self.player_xy[1]
            if 0.5 < (dx * dx + dy * dy) ** 0.5 < 24:
                self._move_samples[action].append((dx, dy))
                samples = self._move_samples[action][-24:]
                self._move_samples[action] = samples
                ax = sum(s[0] for s in samples) / len(samples)
                ay = sum(s[1] for s in samples) / len(samples)
                self.move_vec[action] = [ax, ay, float(len(samples))]
        self.player_xy = (cx, cy)

    def axis_actions(self) -> dict[str, Optional[str]]:
        best = {
            "up": (None, 0.0),
            "down": (None, 0.0),
            "left": (None, 0.0),
            "right": (None, 0.0),
        }
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

    def refresh_goals(self, grid: list[list[int]]) -> None:
        if not grid:
            return
        bg = guess_bg(grid)
        hist = Counter(c for row in grid for c in row)
        goals: list[tuple[float, float]] = []
        for bl in connected_blobs(grid, bg):
            if bl["size"] <= 80 and hist[bl["color"]] < hist[bg]:
                goals.append((bl["cx"], bl["cy"]))
        cleaned: list[tuple[float, float]] = []
        for g in goals:
            if all((g[0] - c[0]) ** 2 + (g[1] - c[1]) ** 2 > 9 for c in cleaned):
                cleaned.append(g)
        self.goals = cleaned[:20]
        if self.seek_target is None and self.goals and self.player_xy:
            self.seek_target = max(
                self.goals,
                key=lambda g: (g[0] - self.player_xy[0]) ** 2  # type: ignore[index]
                + (g[1] - self.player_xy[1]) ** 2,
            )
            self.seek_ttl = 36


class MCTSPlanner:
    """UCT over the internal transition graph (not the real env)."""

    def __init__(self, world: InternalWorldModel, rng: random.Random) -> None:
        self.world = world
        self.rng = rng
        self.N: Counter = Counter()
        self.W: dict[tuple[str, str], float] = defaultdict(float)

    def uct_action(self, state: str, legal_names: list[str], sims: int = 48) -> Optional[str]:
        edges = self.world.graph.get(state, {})
        # only actions with known transitions participate in sim; unknowns probed outside
        known = [a for a in legal_names if a in edges]
        if not known:
            return None
        for _ in range(sims):
            self._simulate(state, known, depth=6)
        best, best_v = None, -1e18
        for a in known:
            n = self.N[(state, a)]
            if n <= 0:
                continue
            v = self.W[(state, a)] / n
            if v > best_v:
                best_v = v
                best = a
        return best

    def _simulate(self, state: str, legal: list[str], depth: int) -> float:
        if depth <= 0:
            return self.world.reward[state]
        # UCT pick
        total = sum(self.N[(state, a)] for a in legal) + 1
        best_a, best_u = legal[0], -1e18
        for a in legal:
            n = self.N[(state, a)]
            if n == 0:
                u = 1e6 + self.rng.random()
            else:
                q = self.W[(state, a)] / n
                u = q + 1.25 * math.sqrt(math.log(total) / n)
            if u > best_u:
                best_u = u
                best_a = a
        nxt = self.world.graph.get(state, {}).get(best_a, state)
        # bootstrap reward
        r = self.world.reward[nxt]
        if nxt != state:
            r += 0.02
        cont_legal = list(self.world.graph.get(nxt, {}).keys()) or legal
        v = r + 0.9 * self._simulate(nxt, cont_legal, depth - 1)
        self.N[(state, best_a)] += 1
        self.W[(state, best_a)] += v
        return v


class Reflector:
    """Self-critique: compare predicted next hash vs reality."""

    def __init__(self) -> None:
        self.mismatch = 0
        self.ok = 0
        self.force_probe = False
        self.notes: list[str] = []

    def predict(self, world: InternalWorldModel, state: str, action: str) -> Optional[str]:
        return world.graph.get(state, {}).get(action)

    def verify(self, predicted: Optional[str], actual: str) -> None:
        if predicted is None:
            return
        if predicted == actual:
            self.ok += 1
            self.force_probe = False
        else:
            self.mismatch += 1
            self.force_probe = True
            self.notes.append(f"MISMATCH pred={predicted[:8]} act={actual[:8]}")
            self.notes = self.notes[-20:]


class Brain:
    """Nova-ARC cognitive stack for one game episode."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.hyper = HypergraphStore()
        self.causal = CausalEngine()
        self.router = InfoGainRouter(self.causal)
        self.world = InternalWorldModel()
        self.mcts = MCTSPlanner(self.world, rng)
        self.reflect = Reflector()
        self.recent: deque = deque(maxlen=64)
        self.recipes: list[list[ActionKey]] = []
        self.best_levels = 0
        self.stall = 0
        self.last_hash: Optional[str] = None
        self.last_action: Optional[str] = None
        self.streak = 0
        self.steps_level = 0
        self.deaths = 0
        self.phase = "explore"  # explore | exploit | reflect
        self.tried_clicks: set[tuple[int, int]] = set()
        self.click_q: deque = deque()
        self.replaying: Optional[list[ActionKey]] = None
        self.recipe_i = 0
        self.meta_cycles = 0
        self.act_hist: deque = deque(maxlen=12)
        self.rule_queue: deque = deque()  # pending actions from SymbolicRule
        self.rule_active: Optional[str] = None
        self.rule_fails = 0
        # Discrete navigation memory (ls20-like: 4-way move + step budget)
        self.wall_hits: Counter = Counter()  # (state, action) -> count
        self.death_trails: list[tuple[str, ...]] = []
        self.death_actions: Counter = Counter()
        self.unique_states: set[str] = set()
        self.discrete_nav = False

    def note_action(self, name: str) -> None:
        self.act_hist.append(name)

    def note_wall(self, state: str, action: str) -> None:
        self.wall_hits[(state, action)] += 1

    def note_death(self) -> None:
        trail = tuple(self.act_hist)
        if trail:
            self.death_trails.append(trail)
            self.death_trails = self.death_trails[-8:]
            for a in trail[-6:]:
                self.death_actions[a] += 1

    def is_wall(self, state: str, action: str) -> bool:
        return self.wall_hits[(state, action)] >= 1

    def start_rule(self, rule: SymbolicRule) -> None:
        self.rule_queue = deque(rule.seq)
        self.rule_active = rule.to_axiom()
        self.rule_fails = 0

    def clear_rule(self) -> None:
        self.rule_queue.clear()
        self.rule_active = None

    def oscillating(self) -> bool:
        h = list(self.act_hist)
        if len(h) < 6:
            return False
        a, b = h[-2], h[-1]
        if a == b:
            return False
        # A,B,A,B,A,B pattern
        return all(h[-2 * i] == b and h[-2 * i - 1] == a for i in range(1, 4))

    def on_level_up(self, levels: int) -> None:
        window = [t[0] for t in list(self.recent)[-20:]]
        if window:
            self.recipes.append(window)
            self.recipes = self.recipes[-24:]
        self.best_levels = levels
        self.router.reset_level()
        self.tried_clicks.clear()
        self.click_q.clear()
        self.stall = 0
        self.replaying = None
        self.steps_level = 0
        self.world.seek_target = None
        self.phase = "explore"
        self.meta_cycles += 1
        self.causal.axioms.append(f"LEVEL->{levels}")
        self.clear_rule()
        # Promote level-up window as a progress combo rule if long enough
        names = [a.name for a in window if a.x is None]
        if len(names) >= 2:
            seq = tuple(names[-min(6, len(names)) :])
            self.causal.combo_try[seq] += 2
            self.causal.combo_hit[seq] += 2
            self.hyper.add_ordered_combo(seq, "progress")
            self.causal._remine_rules()


# ═══════════════════════════════════════════════════════════════════════════
# Optional LLM (Zova secondary path) — disabled on Kaggle
# ═══════════════════════════════════════════════════════════════════════════

def _load_dotenv() -> None:
    try:
        here = Path(__file__).resolve()
    except NameError:
        return
    root = here.parents[1] if here.parent.name == "agent" else here.parent
    for env_path in (root / ".env", Path.cwd() / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            existing = os.environ.get(k, "")
            weak_key = (
                k.endswith("_KEY")
                or k.endswith("_TOKEN")
                or k == "OPENAI_API_KEY"
            )
            weak = weak_key and (
                len(existing) < 16
                or existing.lower()
                in {"sk-xxxx", "sk-xxx", "your_api_key", "replace_me"}
            )
            if k and (k not in os.environ or weak):
                os.environ[k] = v
        break


class LLMAdvisor:
    """Zova secondary path: local can enable via AETHER_LLM=1; Kaggle forces 0."""

    def __init__(self) -> None:
        _load_dotenv()
        self.key = os.environ.get("OPENAI_API_KEY", "").strip()
        if len(self.key) < 16 or self.key.lower() in {"sk-xxxx", "sk-xxx"}:
            self.key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            self.base = os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
            ).rstrip("/")
            self.model = os.environ.get("AETHER_LLM_MODEL", "deepseek-v4-pro")
            self.provider = "deepseek"
        else:
            self.base = os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/")
            self.model = os.environ.get("AETHER_LLM_MODEL", "deepseek-v4-pro")
            self.provider = "openai-compat"
            if "deepseek" in self.base:
                self.provider = "deepseek"
        # Default OFF for Kaggle/offline safety; local .env sets AETHER_LLM=1
        switch = os.environ.get("AETHER_LLM", "0").strip()
        self.enabled = bool(self.key) and len(self.key) >= 16 and switch == "1"
        self.calls = 0
        self.fail_streak = 0
        self.max_calls = int(os.environ.get("AETHER_LLM_MAX_CALLS", "80"))
        self.last_error = ""

    def suggest(
        self, frame: FrameData, legal: list[GameAction], brain: Brain, game_id: str
    ) -> Optional[tuple[str, Optional[int], Optional[int], str]]:
        if not self.enabled or self.fail_streak >= 4 or self.calls >= self.max_calls:
            return None
        # Ask cadence: checkpoints + every-other stall tick (avoid spam)
        ask = False
        if brain.stall >= 2 and brain.stall % 2 == 0:
            ask = True
        elif brain.steps_level in (2, 5, 8, 12, 18, 25, 40, 60, 90, 130, 180, 250, 350, 500):
            ask = True
        elif brain.steps_level > 15 and brain.steps_level % 15 == 0:
            ask = True
        elif brain.best_levels > 0 and brain.steps_level < 40:
            ask = True
        if not ask:
            return None
        print(
            f"[LLM] ask game={game_id} step={brain.steps_level} stall={brain.stall} "
            f"calls={self.calls}/{self.max_calls} last={brain.last_action}",
            flush=True,
        )
        legal_names = [a.name for a in legal]
        effects = {
            name: {
                "trials": int(brain.causal.trials[name]),
                "change": round(brain.causal.p_change(name), 2),
                "noop": round(brain.causal.p_noop(name), 2),
                "progress": int(brain.causal.progress[name]),
                "causal": round(brain.causal.causal_score(name), 2),
            }
            for name in legal_names
            if brain.causal.trials[name] > 0
        }
        recent = []
        for item in list(brain.recent)[-8:]:
            act, _bh, _ah, nch, prog = item
            recent.append({"action": act.fp(), "changed": nch, "progressed": bool(prog)})
        hist = list(brain.act_hist)
        hist_cnt = Counter(hist)
        overused = [
            name
            for name, c in hist_cnt.items()
            if len(hist) >= 6 and c >= max(4, len(hist) // 2)
        ]
        ban = []
        if brain.stall >= 2 and brain.last_action:
            ban.append(brain.last_action)
        if brain.streak >= 3 and brain.last_action:
            ban.append(brain.last_action)
        ban.extend(overused)
        # Prefer under-tried / high-causal alternatives in the prompt
        prefer = sorted(
            [
                n
                for n in legal_names
                if n not in ban and n != "RESET"
            ],
            key=lambda n: (
                -brain.causal.causal_score(n) if brain.causal.trials[n] else 0.0,
                brain.causal.p_noop(n) if brain.causal.trials[n] else 0.5,
                hist_cnt.get(n, 0),
            ),
        )[:4]
        # Ban known wall directions at current visual state if we can match
        # (approximate: heavily noop actions)
        wallish = [
            n
            for n in legal_names
            if brain.causal.trials[n] >= 3 and brain.causal.p_noop(n) >= 0.7
        ]
        ban.extend(wallish)
        prompt = {
            "task": "ARC-AGI-3 interactive black-box. No instructions given.",
            "game_id": game_id,
            "levels": frame.levels_completed,
            "deaths": brain.deaths,
            "discrete_nav": brain.discrete_nav,
            "unique_states": len(brain.unique_states),
            "legal": legal_names,
            "do_not_repeat": list(dict.fromkeys(ban)),
            "prefer_try": prefer,
            "phase": brain.phase,
            "symbolic_rules": brain.causal.rule_summaries(8),
            "axioms": brain.causal.axioms[-10:],
            "axes": brain.world.axis_actions(),
            "player_xy": brain.world.player_xy,
            "seek_target": brain.world.seek_target,
            "goals": brain.world.goals[:8],
            "effects": effects,
            "recent": recent,
            "action_hist_tail": hist[-10:],
            "stall": brain.stall,
            "streak": brain.streak,
            "last_action": brain.last_action,
            "steps_level": brain.steps_level,
            "grid_downsample": compact_grid_text(frame, max_side=18),
            "reply": 'JSON {"action":"ACTION2","x":null,"y":null,"why":"..."}',
            "hints": [
                "MUST NOT pick any action in do_not_repeat.",
                "Bias toward prefer_try and axes toward seek_target/goals.",
                "If discrete_nav=true: actions are movement; step budget is limited — reach goals fast, avoid walls (high noop).",
                "If deaths>0, diversify path; do not spam the same corridor.",
                "Never RESET mid-game.",
            ],
        }
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.25,
            "max_tokens": int(os.environ.get("AETHER_LLM_MAX_TOKENS", "1200")),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Nova-ARC for ARC-AGI-3. Use evidence + grid downsample. "
                        "If stalled, change strategy. Return one JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        if self.provider == "deepseek" and os.environ.get("AETHER_LLM_THINKING", "1") != "0":
            body["thinking"] = {"type": "enabled"}
            body["reasoning_effort"] = os.environ.get("AETHER_LLM_REASONING_EFFORT", "high")
        try:
            req = urllib.request.Request(
                f"{self.base}/chat/completions",
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180, context=ssl.create_default_context()) as resp:
                raw = json.loads(resp.read().decode())
            msg = raw["choices"][0]["message"]
            text = msg.get("content") or ""
            if not text and msg.get("reasoning_content"):
                text = str(msg.get("reasoning_content"))
            # DeepSeek thinking sometimes puts final answer after reasoning
            if msg.get("reasoning_content") and "{" not in text:
                text = f"{text}\n{msg.get('reasoning_content')}"
            self.calls += 1
            self.fail_streak = 0
            parsed = self._parse(text, legal_names)
            if parsed is None:
                snippet = text.replace("\n", " ")[:220]
                print(f"[LLM] ok call={self.calls} parsed=None raw={snippet!r}", flush=True)
            else:
                print(
                    f"[LLM] ok call={self.calls} parsed={parsed[0]}",
                    flush=True,
                )
            return parsed
        except Exception as e:  # noqa: BLE001
            self.fail_streak += 1
            self.last_error = f"{type(e).__name__}:{e}"
            print(
                f"[LLM] FAIL streak={self.fail_streak} err={self.last_error[:180]}",
                flush=True,
            )
            return None

    @staticmethod
    def _parse(
        text: str, legal_names: list[str]
    ) -> Optional[tuple[str, Optional[int], Optional[int], str]]:
        import re

        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                obj = json.loads(text[s : e + 1])
                name = str(obj.get("action", "")).upper().strip()
                if name in legal_names:
                    try:
                        xi = int(obj["x"]) if obj.get("x") is not None else None
                        yi = int(obj["y"]) if obj.get("y") is not None else None
                    except (TypeError, ValueError):
                        xi = yi = None
                    return name, xi, yi, str(obj.get("why", "llm"))[:160]
            except json.JSONDecodeError:
                pass
        # Fallback: first legal ACTION* token in free text
        for m in re.finditer(r"ACTION[0-7]|RESET", text.upper()):
            name = m.group(0)
            if name in legal_names:
                return name, None, None, "llm_fallback"
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════

class MyAgent(Agent):
    """Nova-ARC hybrid causal agent (offline-capable)."""

    MAX_ACTIONS = 1200
    STALL_UNDO = 5
    INFO_GAIN_FLOOR = 0.04

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        self.rng = random.Random(seed)
        self.brain = Brain(self.rng)
        self.llm = LLMAdvisor()
        print(
            f"[Nova] game={self.game_id} llm_enabled={self.llm.enabled} "
            f"model={self.llm.model} provider={self.llm.provider}",
            flush=True,
        )
        self._pending: Optional[tuple[str, ActionKey, int, list[list[int]], float, tuple[int, ...]]] = None
        self._sweep: list[tuple[int, int]] = []
        self._sweep_i = 0

    @property
    def name(self) -> str:
        mode = "llm" if self.llm.enabled else "causal"
        return f"{super().name}.nova-arc.{mode}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ── learning ───────────────────────────────────────────────────────────

    def _observe(self, latest: FrameData) -> None:
        h = grid_hash(latest)
        grid = top_grid(latest)
        ent = color_entropy(grid)
        sig = compact_signature(grid)
        self.brain.hyper.upsert_state(
            h, {"ent": ent, "sig": sig, "levels": latest.levels_completed}
        )
        if self.brain.last_hash == h:
            self.brain.stall += 1
        else:
            self.brain.stall = 0
        self.brain.last_hash = h

        if self._pending is not None:
            bh, action, lb, bg, bent, bsig = self._pending
            self._pending = None
            ah = h
            n_ch = len(changed_cells(bg, grid)) if bg and grid else 0
            progressed = int(latest.levels_completed or 0) > lb
            pred = self.brain.reflect.predict(self.brain.world, bh, action.name)
            self.brain.reflect.verify(pred, ah)

            self.brain.causal.observe(
                action.name,
                bh,
                ah,
                n_ch,
                progressed,
                bsig,
                sig,
                hyper=self.brain.hyper,
            )
            if self.brain.rule_active and n_ch == 0 and not progressed:
                self.brain.rule_fails += 1
                if self.brain.rule_fails >= 2:
                    self.brain.clear_rule()
            elif self.brain.rule_active and (n_ch > 0 or progressed):
                self.brain.rule_fails = 0
            if bh == ah and not progressed:
                self.brain.note_wall(bh, action.name)
            self.brain.unique_states.add(ah)
            self.brain.world.learn_transition(bh, action.name, ah, progressed)
            self.brain.world.update_motion(action.name, bg, grid)
            self.brain.router.record_realized_gain(bent, ent, bh != ah)
            self.brain.recent.append((action, bh, ah, n_ch, progressed))
            self.brain.hyper.add_hyperedge([bh, action.name, ah], "seq")
            if n_ch:
                self.brain.hyper.add_hyperedge(
                    [action.name, f"chg:{n_ch // 10}", f"lv:{lb}"], "effect"
                )
                for x, y, _, _ in changed_cells(bg, grid)[:16]:
                    if (x, y) not in self.brain.tried_clicks:
                        self.brain.click_q.appendleft((x, y))

            if progressed:
                self.brain.on_level_up(int(latest.levels_completed or 0))

            if self.brain.reflect.force_probe:
                self.brain.phase = "reflect"
            elif self.brain.router.explore_stop and self.brain.causal.rules:
                self.brain.phase = "exploit"
            elif self.brain.steps_level < 18:
                self.brain.phase = "explore"
            else:
                self.brain.phase = "exploit" if self.brain.world.graph else "explore"

        if self.brain.steps_level % 6 == 0:
            self.brain.world.refresh_goals(grid)

    def _commit(
        self,
        action: GameAction,
        latest: FrameData,
        reason: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> GameAction:
        key = ActionKey(action.name, x, y)
        grid = top_grid(latest)
        if action.is_complex():
            xx = 32 if x is None else int(x)
            yy = 32 if y is None else int(y)
            action.set_data({"x": xx, "y": yy})
            key = ActionKey(action.name, xx, yy)
            action.reasoning = {
                "policy": "nova-arc",
                "phase": self.brain.phase,
                "why": reason,
                "xy": [xx, yy],
                "rules": self.brain.causal.rule_summaries(3),
            }
        else:
            action.reasoning = (
                f"nova:{self.brain.phase}:{reason}|L{latest.levels_completed}"
                f"|S{self.brain.stall}|R{len(self.brain.causal.rules)}"
            )
        sh = grid_hash(latest)
        self._pending = (
            sh,
            key,
            int(latest.levels_completed or 0),
            [row[:] for row in grid],
            color_entropy(grid),
            compact_signature(grid),
        )
        self.brain.router.note_visit(sh, key.name)
        self.brain.note_action(key.name)
        if key.name == self.brain.last_action:
            self.brain.streak += 1
        else:
            self.brain.streak = 1
            self.brain.last_action = key.name
        self.brain.steps_level += 1
        if self.brain.world.seek_ttl > 0:
            self.brain.world.seek_ttl -= 1
            if self.brain.world.seek_ttl == 0:
                self.brain.world.seek_target = None
        return action

    def _by_name(self, legal: list[GameAction], name: str) -> Optional[GameAction]:
        return next((a for a in legal if a.name == name), None)

    def _ensure_sweep(self, latest: FrameData) -> None:
        if self._sweep and self._sweep_i < len(self._sweep):
            return
        pts = salient_clicks(latest, limit=64)
        grid = top_grid(latest)
        if grid:
            bg = guess_bg(grid)
            for y, row in enumerate(grid):
                for x, c in enumerate(row):
                    if c != bg and (x + y) % 2 == 0:
                        pts.append((x, y))
        for y in range(0, 64, 4):
            for x in range(0, 64, 4):
                pts.append((x, y))
        seen: set[tuple[int, int]] = set()
        uniq: list[tuple[int, int]] = []
        for p in pts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        self._sweep = uniq
        self._sweep_i = 0

    def _pick_click(self, latest: FrameData) -> tuple[int, int]:
        while self.brain.click_q:
            pt = self.brain.click_q.popleft()
            if pt not in self.brain.tried_clicks:
                self.brain.tried_clicks.add(pt)
                return pt
        self._ensure_sweep(latest)
        while self._sweep_i < len(self._sweep):
            pt = self._sweep[self._sweep_i]
            self._sweep_i += 1
            if pt not in self.brain.tried_clicks:
                self.brain.tried_clicks.add(pt)
                return pt
        return self.rng.randint(0, 63), self.rng.randint(0, 63)

    def _spatial(self, legal: list[GameAction], latest: FrameData) -> Optional[GameAction]:
        w = self.brain.world
        if not w.player_xy or not w.seek_target:
            return None
        axis = w.axis_actions()
        if not any(axis.values()):
            return None
        px, py = w.player_xy
        tx, ty = w.seek_target
        dx, dy = tx - px, ty - py
        order = (
            (["right", "left"] if dx > 0 else ["left", "right"])
            + (["down", "up"] if dy > 0 else ["up", "down"])
            if abs(dx) >= abs(dy)
            else (["down", "up"] if dy > 0 else ["up", "down"])
            + (["right", "left"] if dx > 0 else ["left", "right"])
        )
        if (dx * dx + dy * dy) ** 0.5 < 3.5:
            for name in ("ACTION5",):
                a = self._by_name(legal, name)
                if a and a.is_simple():
                    return self._commit(a, latest, "seek_interact")
            for a in legal:
                if a.is_complex():
                    return self._commit(a, latest, "seek_click", int(tx), int(ty))
            w.seek_target = None
            return None
        sh = grid_hash(latest)
        for card in order:
            aname = axis.get(card)
            if not aname:
                continue
            a = self._by_name(legal, aname)
            if a is None:
                continue
            if self.brain.is_wall(sh, aname):
                continue
            if self.brain.world.graph.get(sh, {}).get(aname) == sh:
                continue
            if aname == self.brain.last_action and self.brain.streak >= 5:
                continue
            return self._commit(a, latest, f"seek_{card}")
        return None

    def _recipe(self, legal: list[GameAction], latest: FrameData) -> Optional[GameAction]:
        if self.brain.replaying is None:
            if not self.brain.recipes:
                return None
            if self.brain.best_levels == 0 and self.brain.stall < 2:
                return None
            idx = -1 if self.rng.random() < 0.75 else self.rng.randrange(
                len(self.brain.recipes)
            )
            self.brain.replaying = list(self.brain.recipes[idx])
            self.brain.recipe_i = 0
        assert self.brain.replaying is not None
        if self.brain.recipe_i >= len(self.brain.replaying):
            self.brain.replaying = None
            return None
        step = self.brain.replaying[self.brain.recipe_i]
        self.brain.recipe_i += 1
        a = self._by_name(legal, step.name)
        if a is None:
            self.brain.replaying = None
            return None
        if a.is_complex():
            x = step.x if step.x is not None else self._pick_click(latest)[0]
            y = step.y if step.y is not None else self._pick_click(latest)[1]
            return self._commit(a, latest, "recipe", x, y)
        return self._commit(a, latest, "recipe")

    def _follow_rule(
        self,
        legal: list[GameAction],
        latest: FrameData,
        *,
        allow_start: bool = False,
    ) -> Optional[GameAction]:
        """Continue an active symbolic combo; optionally start a high-conf progress rule."""
        if self.brain.rule_queue:
            name = self.brain.rule_queue.popleft()
            a = self._by_name(legal, name)
            if a is None:
                self.brain.clear_rule()
                return None
            if a.is_complex():
                x, y = self._pick_click(latest)
                return self._commit(a, latest, "sym_rule", x, y)
            return self._commit(a, latest, "sym_rule")

        if not allow_start:
            return None
        # Only after real progress evidence / exploit: avoid spurious early combos
        if self.brain.best_levels <= 0 and self.brain.steps_level < 40:
            return None
        if self.brain.oscillating():
            return None
        legal_names = {a.name for a in legal}
        rule = self.brain.causal.best_rule(legal_names, min_conf=0.55)
        if rule is None or rule.effect != "progress":
            return None
        if len(rule.seq) == 1 and rule.seq[0] == self.brain.last_action and self.brain.streak >= 3:
            return None
        if self.rng.random() > 0.45:
            return None
        self.brain.start_rule(rule)
        return self._follow_rule(legal, latest, allow_start=False)

    def _aether_score(
        self, action: GameAction, state_h: str, simples: list[GameAction]
    ) -> float:
        """Aether-Prime heritage scoring + causal boost."""
        name = action.name
        pair_n = self.brain.router.pair_visits[(state_h, name)]
        trials = max(self.brain.causal.trials[name], 1.0)
        score = 3.2 / (1.0 + pair_n)
        nxt = self.brain.world.graph.get(state_h, {}).get(name)
        if nxt is not None:
            score += 2.2 / (1.0 + self.brain.router.state_visits[nxt])
            if nxt == state_h:
                score -= 3.0
        else:
            score += 1.4
        score += 5.0 * (self.brain.causal.progress[name] / trials)
        score += 0.9 * (self.brain.causal.change[name] / trials)
        score -= 1.4 * (self.brain.causal.noop[name] / trials)
        score += 0.6 * self.brain.causal.causal_score(name)
        if name == self.brain.last_action:
            score -= 0.45 * self.brain.streak
        if action.is_complex():
            productive = any(
                self.brain.causal.change[a.name] > 0 for a in simples
            )
            if productive:
                score -= 2.2
            else:
                score += 0.15 * max(0, 24 - len(self.brain.tried_clicks))
            if self.brain.steps_level < 10:
                score -= 1.8
        score += 0.04 * self.rng.random()
        return score

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self._observe(latest_frame)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.GAME_OVER:
                self.brain.deaths += 1
                self.brain.note_death()
            self.brain.replaying = None
            self.brain.clear_rule()
            self.brain.stall = 0
            self.brain.streak = 0
            self.brain.steps_level = 0
            self.brain.world.seek_target = None
            self._pending = None
            self._sweep = []
            self._sweep_i = 0
            self.brain.phase = "explore"
            self.brain.router.explore_stop = False
            action = GameAction.RESET
            action.reasoning = "nova:reset"
            return action

        legal = resolve_actions(latest_frame.available_actions)
        state = grid_hash(latest_frame)
        simples = [a for a in legal if a.is_simple()]
        complex_acts = [a for a in legal if a.is_complex()]
        self.brain.discrete_nav = bool(simples) and not complex_acts and len(simples) <= 5
        self.brain.unique_states.add(state)

        # Click-only / click+undo
        if complex_acts and not simples:
            x, y = self._pick_click(latest_frame)
            return self._commit(complex_acts[0], latest_frame, "click_cover", x, y)
        if complex_acts and simples and all(a.name == "ACTION7" for a in simples):
            if self.brain.stall >= 3:
                return self._commit(simples[0], latest_frame, "undo")
            x, y = self._pick_click(latest_frame)
            return self._commit(complex_acts[0], latest_frame, "click_cover", x, y)

        # 0) Finish an already-running symbolic rule (do not start early)
        ruled = self._follow_rule(legal, latest_frame, allow_start=False)
        if ruled is not None:
            return ruled

        # 1) Optional LLM (only when AETHER_LLM=1 survives dotenv)
        advice = self.llm.suggest(latest_frame, legal, self.brain, self.game_id)
        if advice:
            name, xi, yi, why = advice
            hist = list(self.brain.act_hist)
            monopoly = (
                len(hist) >= 6
                and hist.count(name) >= max(4, len(hist) // 2)
            )
            reject = False
            if name == "RESET":
                reject = True
                print("[LLM] reject RESET", flush=True)
            elif self.brain.stall >= 2 and name == self.brain.last_action:
                reject = True
                print(
                    f"[LLM] reject repeat {name} under stall={self.brain.stall}",
                    flush=True,
                )
            elif monopoly and self.brain.best_levels == 0:
                reject = True
                print(f"[LLM] reject monopoly {name} hist={hist.count(name)}/{len(hist)}", flush=True)
            if not reject:
                a = self._by_name(legal, name)
                if a is not None:
                    if a.is_complex():
                        if xi is None or yi is None:
                            xi, yi = self._pick_click(latest_frame)
                        return self._commit(a, latest_frame, f"llm:{why}", xi, yi)
                    return self._commit(a, latest_frame, f"llm:{why}")

        # 1b) Locksmith-ish: after many no-progress steps, force click exploration
        if (
            complex_acts
            and self.brain.best_levels == 0
            and self.brain.steps_level >= 40
            and self.brain.steps_level % 7 == 0
        ):
            x, y = self._pick_click(latest_frame)
            return self._commit(complex_acts[0], latest_frame, "force_click", x, y)

        undo = self._by_name(legal, "ACTION7")
        if self.brain.stall >= self.STALL_UNDO and undo is not None:
            if self.rng.random() < 0.7:
                return self._commit(undo, latest_frame, "undo")

        # 2) Recipe after prior progress
        rec = self._recipe(legal, latest_frame)
        if rec is not None:
            return rec

        # 3) Break A↔B oscillation
        if self.brain.oscillating():
            self.brain.world.seek_target = None
            self.brain.clear_rule()
            if undo is not None and self.rng.random() < 0.45:
                return self._commit(undo, latest_frame, "break_osc_undo")
            alt = [
                a
                for a in simples
                if a.name not in set(list(self.brain.act_hist)[-2:])
            ]
            if alt:
                pick = max(alt, key=lambda a: self._aether_score(a, state, simples))
                return self._commit(pick, latest_frame, "break_osc")

        # 4) Spatial seek FIRST — especially discrete 4-way nav (ls20)
        seek_p = 0.9 if self.brain.discrete_nav else 0.75
        seek_after = 4 if self.brain.discrete_nav else 6
        if self.brain.steps_level >= seek_after and not self.brain.oscillating():
            # Refresh goals more often on discrete nav (step budget games)
            if self.brain.discrete_nav and self.brain.steps_level % 3 == 0:
                self.brain.world.refresh_goals(top_grid(latest_frame))
            seek = self._spatial(legal, latest_frame)
            if seek is not None and self.rng.random() < seek_p:
                return seek

        # 5) Untried edges (skip known walls)
        untried = [
            a
            for a in simples
            if self.brain.router.pair_visits[(state, a.name)] == 0
            and not self.brain.is_wall(state, a.name)
        ]
        if untried:
            if self.brain.streak >= 2 and self.brain.last_action:
                alt = [a for a in untried if a.name != self.brain.last_action]
                if alt:
                    untried = alt
            return self._commit(self.rng.choice(untried), latest_frame, "edge")

        # 5b) Discrete nav: never re-ram known walls; pick freshest direction
        if self.brain.discrete_nav and simples:
            open_acts = [a for a in simples if not self.brain.is_wall(state, a.name)]
            if open_acts:
                # Prefer underused actions and avoid death-trail-heavy ones
                def nav_score(a: GameAction) -> float:
                    s = self._aether_score(a, state, simples)
                    s -= 0.8 * self.brain.death_actions[a.name]
                    if a.name == self.brain.last_action:
                        s -= 1.2 * self.brain.streak
                    return s

                pick = max(open_acts, key=nav_score)
                if not (
                    pick.name == self.brain.last_action and self.brain.streak >= 4
                ):
                    return self._commit(pick, latest_frame, "nav_open")

        # 6) BFS frontier
        hint = self.brain.world.bfs_unexplored(state)
        if hint:
            a = self._by_name(legal, hint)
            if a is not None and not (
                a.name == self.brain.last_action and self.brain.streak >= 5
            ):
                return self._commit(a, latest_frame, "bfs")

        # 7) Start high-confidence progress symbolic rules (late / post-progress)
        ruled = self._follow_rule(legal, latest_frame, allow_start=True)
        if ruled is not None:
            return ruled

        # 8) MCTS when exploit phase has graph
        if self.brain.phase == "exploit" and simples and len(self.brain.world.graph) >= 4:
            names = [a.name for a in simples]
            choice = self.brain.mcts.uct_action(state, names, sims=64)
            if choice:
                a = self._by_name(legal, choice)
                if a is not None and not (
                    a.name == self.brain.last_action and self.brain.streak >= 6
                ):
                    return self._commit(a, latest_frame, "mcts")

        # 9) Softmax over Aether+causal scores
        scored: list[tuple[float, GameAction]] = []
        for a in simples:
            scored.append((self._aether_score(a, state, simples), a))
        for a in complex_acts:
            scored.append((self._aether_score(a, state, simples), a))
        if not scored:
            fb = self.rng.choice(legal)
            if fb.is_complex():
                x, y = self._pick_click(latest_frame)
                return self._commit(fb, latest_frame, "fallback", x, y)
            return self._commit(fb, latest_frame, "fallback")

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: min(5, len(scored))]
        temp = 0.5 + 0.05 * min(self.brain.stall, 10)
        weights = [math.exp(s / max(temp, 0.15)) for s, _ in top]
        pick = self.rng.choices(top, weights=weights, k=1)[0][1]
        if pick.is_complex():
            x, y = self._pick_click(latest_frame)
            return self._commit(pick, latest_frame, "explore_click", x, y)
        return self._commit(pick, latest_frame, "explore")
