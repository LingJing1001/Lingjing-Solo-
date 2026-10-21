"""Explore what was actually observed, not what the rules predict.

`worldsim` plans through a *learned* forward model: it extrapolates rules to boards
the agent has never stood on. That is powerful and it is also how you end up walking
into a wall for two hundred steps — the model is wrong exactly where the rules are
thin, which is exactly where the agent needs help.

This module takes the opposite stance. Every edge is a fact: "from this board, this
key led to that board". Nothing is extrapolated. The cost is that the graph only
knows where the agent has been; the compensation is a deliberate frontier policy:

    plan()  ->  BFS to the nearest board that still has an untried key,
                return the keys that walk there

The interesting subtlety is *why* the frontier is "a board with an untried key"
rather than "a board never visited". A visited board can still be unexplored: you
arrived, you tried one direction, you left. Treating it as finished strands you in
a corridor you never actually finished testing. Counting untried keys makes the
agent systematically drain each board's action set, which is also what the rule
learner needs — it only learns from repeated (board, action) pairs.
"""
from __future__ import annotations

import collections


class StateGraph:
    """Boards as nodes, (board, action) as edges, all of them observed."""

    def __init__(self, actions, max_nodes: int = 3000):
        self.actions = tuple(actions)
        self.edges: dict = {}        # (fp, action) -> fp'
        self.visits: dict = {}       # fp -> how often the engine actually showed it
        self.bad: set = set()        # (fp, action) known useless (no-op or fatal)
        self.max_nodes = max_nodes
        self.last_path_len = 0

    # ------------------------------------------------------------------ observation

    def note_visit(self, fp):
        if fp is None:
            return
        self.visits[fp] = self.visits.get(fp, 0) + 1

    def observe(self, src, action, dst):
        """Record one real transition. Returns True if it contradicts an earlier one."""
        if src is None or dst is None or not action:
            return False
        a = action.split("@")[0]
        key = (src, a)
        old = self.edges.get(key)
        self.edges[key] = dst
        return old is not None and old != dst

    def mark_bad(self, fp, action):
        if fp is None or not action:
            return
        self.bad.add((fp, action.split("@")[0]))

    # ------------------------------------------------------------------ query

    def untried(self, fp, skip_bad: bool = True):
        """A key never pressed on this board (or one we know is useless)."""
        for a in self.actions:
            if skip_bad and (fp, a) in self.bad:
                continue
            if (fp, a) not in self.edges:
                return a
        return None

    def exhausted(self, fp):
        return all((fp, a) in self.edges or (fp, a) in self.bad for a in self.actions)

    def plan(self, start, deadline_nodes: int = 800):
        """Keys that walk to the nearest board with something left to try.

        Returns [] when every reachable board is drained — the caller should then
        fall back to the rule model, which can at least guess about unobserved edges.
        """
        if start is None:
            return []
        here = self.untried(start)
        if here is not None:
            return [here]

        prev = {start: None}
        q = collections.deque([start])
        found = None
        while q and len(prev) < deadline_nodes:
            cur = q.popleft()
            for a in self.actions:
                if (cur, a) in self.bad:
                    continue
                nxt = self.edges.get((cur, a))
                if nxt is None or nxt in prev:
                    continue
                prev[nxt] = (cur, a)
                if self.untried(nxt) is not None:
                    found = nxt
                    break
                q.append(nxt)
            if found is not None:
                break
        if found is None:
            return []

        path = []
        node = found
        while prev[node] is not None:
            node_prev, a = prev[node]
            path.append(a)
            node = node_prev
        path.reverse()
        nxt_a = self.untried(found)
        if nxt_a is not None:
            path.append(nxt_a)
        self.last_path_len = len(path)
        return path

    # ------------------------------------------------------------------ stats

    def stats(self):
        boards = len({k[0] for k in self.edges} | set(self.visits))
        return f"graph: {boards} boards, {len(self.edges)} edges, {len(self.bad)} bad"

    def __len__(self):
        return len(self.edges)
