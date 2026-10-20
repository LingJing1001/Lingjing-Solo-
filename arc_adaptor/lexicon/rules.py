"""Rule grammar: one rule shape, learned by explaining and compressing.

    when <trigger> [if | unless <relation>] then <verb>(<who>, <how>)

trigger   an action (ACTION1 ... ACTION7; ACTION6 carries its click) or `any` action
who       picked by relation on the screen before the step:
              clicked, all, same_as_clicked, touching_clicked, colour:c, touching:S, pic:S
          pic:S names one picture of this game. It is allowed but costs more, so on a tie a
          relation wins and the rule is worded without names.
relation  touch:c  the object touches colour c        path:c  colour c lies in the path of its move
          click:c  the clicked object carries colour c (click:none: the click hit no object)
          (c may be "edge"; contact, solidity and the action's own target, nothing game-specific)
verb/how  the engine's sprite operations as the screen shows them:
              move(d)  recolor(map)  turn(transform)  resize(delta)  reshape(to S)  vanish  appear(S, where)

There are no rule kinds. What the first prototype (../2026-10-07__lexicon-code/effect_code.py) called
steer, toggle, solid, remote and tick are sentences here, e.g.
    when ACTION1 unless path:c8 then move(pic:G, (-1,+0))
    when ACTION6 then recolor(clicked, {8:9, 9:8})
    when any then move(colour:c3, (+1,+0))
and so are mechanics it had no kind for (gravity, conveyors, chasers, lights-out ...).

A rule may also range over several actions: one selector and one verb, with a parameter for each
action (when ACTION1|ACTION2 then move(pic:G, A1 (-1,+0) | A2 (+1,+0))). It pools the evidence of its
actions, so "these actions move this object" is confirmed sooner than four separate rules would be.

Learning (`Learner`). Every observed change proposes the rules that would have produced it, under
every selector that picks its object. A candidate seen twice is scored on every step of the level:
its hits, its false predictions, and the events it explains. A greedy minimum-description choice
keeps rules whose explained events outweigh their false predictions plus their own length. A rule
with false predictions may buy a condition (if / unless a relation) that removes them. Events that
no kept rule explains are the residual: where to look next, and what a language model can be asked
to explain. Prediction is closed-world: if no rule fires, nothing is predicted to change.

Nothing here is taken from a particular game. Stdlib only, on top of kaggle/lexicon-dfranzen/lexicon.py.
"""
from __future__ import annotations

import collections
try:
    LX                                   # a host that already holds the lexicon passes it in
except NameError:
    from . import lexicon as LX           # ported: normal package import instead of the sandbox's file loader

_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7")

# description length of each part, in rough units of "one event explained"; a rule is kept only if
# it explains more events than its false predictions plus this length
COST = {"any": 0.25, "action": 0.5,
        "clicked": 0.5, "all": 0.5, "same_as_clicked": 0.75, "touching_clicked": 0.75,
        "colour": 1.0, "pic": 1.25, "touching": 1.5,
        "move": 0.5, "recolor": 0.5, "turn": 0.5, "resize": 0.5, "vanish": 0.5, "reshape": 1.0, "appear": 1.25,
        "condition": 0.75}


# --------------------------------------------------------------------------- the screen as a state

class State:
    """One screen read by the lexicon, with the indexes that selectors and relations need."""

    def __init__(self, code):
        self.code = code
        self.objs = code["placed"]
        self.grid = code["native"]
        self.bg = code["background"]
        self.h, self.w = code["shape"]
        self.cellset = [frozenset(p["cells"]) for p in self.objs]
        self.colours = [frozenset(self.grid[r][c] for r, c in p["cells"]) for p in self.objs]
        self.owner = {}
        for i, p in enumerate(self.objs):
            if not p["container"]:
                for rc in p["cells"]:
                    self.owner[rc] = i
        self.by_sym = collections.defaultdict(list)
        for i, p in enumerate(self.objs):
            self.by_sym[p["sym"]].append(i)
        self.free = [i for i, p in enumerate(self.objs) if not p["container"]]
        self._adj = None
        self._touch = {}
        self._path = {}

    def adjacent(self, i):
        if self._adj is None:
            adj = collections.defaultdict(set)
            for (r, c), a in self.owner.items():
                for dr, dc in _N4:
                    b = self.owner.get((r + dr, c + dc))
                    if b is not None and b != a:
                        adj[a].add(b)
            self._adj = adj
        return self._adj.get(i, set())

    def object_at(self, rc):
        hits = [i for i, cs in enumerate(self.cellset) if rc in cs]
        return min(hits, key=lambda i: len(self.cellset[i])) if hits else None

    def cell_of(self, x, y):
        """Native cell under screen pixel (x, y)."""
        k = self.code["scale"]
        oy, ox = self.code["offset"]
        return ((y - oy) // k, (x - ox) // k)

    def _colour(self, rc):
        r, c = rc
        if not (0 <= r < self.h and 0 <= c < self.w):
            return "edge"
        v = self.grid[r][c]
        return None if v == self.bg else f"c{v}"

    def touch(self, i):
        """Colours (and "edge") on the cells next to object i."""
        if i not in self._touch:
            own, out = self.cellset[i], set()
            for r, c in own:
                for dr, dc in _N4:
                    t = (r + dr, c + dc)
                    if t not in own:
                        v = self._colour(t)
                        if v:
                            out.add(v)
            self._touch[i] = frozenset(out)
        return self._touch[i]

    def path(self, i, d):
        """Colours (and "edge") on the cells object i would sweep moving by d."""
        key = (i, d)
        if key not in self._path:
            own, out = self.cellset[i], set()
            n = max(abs(d[0]), abs(d[1]))
            unit = ((d[0] > 0) - (d[0] < 0), (d[1] > 0) - (d[1] < 0))
            for r, c in own:
                for s in range(1, n + 1):
                    t = (r + unit[0] * s, c + unit[1] * s)
                    if t not in own:
                        v = self._colour(t)
                        if v:
                            out.add(v)
            self._path[key] = frozenset(out)
        return self._path[key]


# --------------------------------------------------------------------------- events: what changed

def _mask(p):
    return tuple(tuple(v >= 0 for v in row) for row in p)


def _colours(p):
    return frozenset(v for row in p for v in row if v >= 0)


def observe(before, after):
    """Changes between two States of one level, as the engine's verbs.

    Each event: verb, obj (index in `before`, None for appear), sym, at (anchor after), and
    d (move), map (recolor: colour pairs), transform (turn), delta (resize), to (new picture).
    """
    lex = before.code["lexicon"]
    d = LX.diff(before.code, after.code)
    i0 = {(p["sym"], p["at"]): i for i, p in enumerate(before.objs)}
    cells0 = {(p["sym"], p["at"]): before.cellset[i] for i, p in enumerate(before.objs)}
    cells1 = {(p["sym"], p["at"]): after.cellset[i] for i, p in enumerate(after.objs)}
    ev = []

    def change(s0, a0, s1, a1):
        p0, p1 = lex.entries[s0]["patch"], lex.entries[s1]["patch"]
        base = {"obj": i0[(s0, a0)], "sym": s0, "to": s1, "at": a1}
        for name, var in LX._variants(p0):
            if var == p1 and name != "rot0":
                return dict(base, verb="turn", transform=name)
        if a0 == a1 and _mask(p0) == _mask(p1):
            pairs = sorted({(a, b) for r0, r1 in zip(p0, p1) for a, b in zip(r0, r1) if a >= 0 and a != b})
            if len({a for a, _ in pairs}) == len(pairs):          # a function: one new colour per old one
                return dict(base, verb="recolor", map=tuple(pairs))
        if _colours(p0) == _colours(p1):
            return dict(base, verb="resize", delta=sum(map(sum, _mask(p1))) - sum(map(sum, _mask(p0))))
        return dict(base, verb="reshape")

    for sym, l, r in d["moved"]:
        ev.append({"verb": "move", "obj": i0[(sym, l)], "sym": sym, "d": (r[0] - l[0], r[1] - l[1]), "at": r})
    for s0, s1, at in d["became"]:
        ev.append(change(s0, at, s1, at))
    gone, new = list(d["disappeared"]), list(d["appeared"])
    for s0, a0 in list(gone):
        for s1, a1 in new:
            if cells0[(s0, a0)] & cells1[(s1, a1)]:
                ev.append(change(s0, a0, s1, a1))
                gone.remove((s0, a0))
                new.remove((s1, a1))
                break
    for s, a in gone:
        ev.append({"verb": "vanish", "obj": i0[(s, a)], "sym": s, "at": a})
    for s, a in new:
        ev.append({"verb": "appear", "obj": None, "sym": s, "at": a})
    return ev


# --------------------------------------------------------------------------- one step, with cached selections

class Step:
    def __init__(self, state, action, click, events):
        self.state, self.action, self.events = state, action, events
        self.click = state.cell_of(*click) if click else None
        self.clicked = state.object_at(self.click) if self.click else None
        self._sel = {}

    def click_features(self):
        if self.click is None:
            return ()
        if self.clicked is None:
            return ("click:none",)
        return tuple(f"click:c{v}" for v in sorted(self.state.colours[self.clicked]))

    def select(self, sel):
        if sel not in self._sel:
            self._sel[sel] = select(sel, self.state, self.clicked)
        return self._sel[sel]


def select(sel, st, clicked):
    """Objects picked by a selector on state `st` (a list of indices)."""
    if sel == "clicked":
        return [clicked] if clicked is not None else []
    if sel == "all":
        return st.free
    if sel == "same_as_clicked":
        return list(st.by_sym[st.objs[clicked]["sym"]]) if clicked is not None else []
    if sel == "touching_clicked":
        return sorted(st.adjacent(clicked)) if clicked is not None else []
    kind, _, arg = sel.partition(":")
    if kind == "pic":
        return st.by_sym.get(arg, [])
    if kind == "colour":
        c = int(arg)
        return [i for i in st.free if c in st.colours[i]]
    if kind == "touching":
        near = set()
        for j in st.by_sym.get(arg, []):
            near |= st.adjacent(j)
        return sorted(i for i in near if st.objs[i]["sym"] != arg)
    return []


def selectors_for(step, i):
    """Every selector that picks object i on this step (the ways to say "this object")."""
    st = step.state
    out = ["all", f"pic:{st.objs[i]['sym']}"]
    out += [f"colour:{c}" for c in sorted(st.colours[i])[:3]]
    if step.clicked is not None:
        if i == step.clicked:
            out.append("clicked")
        if st.objs[i]["sym"] == st.objs[step.clicked]["sym"]:
            out.append("same_as_clicked")
        if i in st.adjacent(step.clicked):
            out.append("touching_clicked")
    out += [f"touching:{s}" for s in sorted({st.objs[j]["sym"] for j in st.adjacent(i)})[:4]]
    return out


def effects_for(step, e):
    """The ways to state what happened to the object."""
    v = e["verb"]
    if v == "move":
        return [("move", e["d"])]
    if v == "recolor":
        return [("recolor", e["map"])]
    if v == "turn":
        return [("turn", e["transform"])]
    if v == "resize":
        return [("resize", e["delta"])]
    if v == "reshape":
        return [("reshape", e["to"])]
    if v == "vanish":
        return [("vanish",)]
    out = [("appear", e["sym"], ("somewhere",)), ("appear", e["sym"], ("at", e["at"]))]
    if step.click:
        out.append(("appear", e["sym"], ("click", (e["at"][0] - step.click[0], e["at"][1] - step.click[1]))))
    return out


def applicable(eff, st, i):
    """recolor only fires on objects that carry a colour it maps."""
    if eff[0] == "recolor":
        return any(a in st.colours[i] for a, _ in eff[1])
    return True


def matches(eff, i, e, click):
    kind = eff[0]
    if kind == "appear":
        if e["verb"] != "appear" or e["sym"] != eff[1]:
            return False
        where = eff[2]
        if where[0] == "at":
            return e["at"] == where[1]
        if where[0] == "click":
            return click is not None and (e["at"][0] - click[0], e["at"][1] - click[1]) == where[1]
        return True
    if e.get("obj") != i:
        return False
    if kind == "move":
        return e["verb"] == "move" and e["d"] == eff[1]
    if kind == "recolor":
        return e["verb"] == "recolor" and set(e["map"]) <= set(eff[1])
    if kind == "turn":
        return e["verb"] == "turn" and e["transform"] == eff[1]
    if kind == "resize":
        return e["verb"] == "resize" and e["delta"] == eff[1]
    if kind == "reshape":
        return e["verb"] != "move" and e.get("to") == eff[1]
    if kind == "vanish":
        return e["verb"] == "vanish"
    return False


def features(step, i, eff):
    """Relations that a condition may test: of object i, and of the step's click target."""
    st = step.state
    f = set(step.click_features())
    if i is not None:
        f |= {f"touch:{c}" for c in st.touch(i)}
        if eff[0] == "move":
            f |= {f"path:{c}" for c in st.path(i, eff[1])}
    return frozenset(f)


def trigger_matches(trig, action):
    return trig == "any" or trig == action


def cost(trig, sel, eff, cond=None):
    c = COST["any"] if trig == "any" else COST["action"]
    c += COST.get(sel.split(":")[0], 1.5)
    c += COST[eff[0]]
    if eff[0] == "recolor":
        c += 0.25 * max(0, len(eff[1]) - 1)
    return c + (COST["condition"] if cond else 0.0)


# --------------------------------------------------------------------------- rules and learning

def _label(trig):
    """Short name of an action inside a family: ACTION3 -> A3; model-facing names (UP, MOUSE) stay."""
    return f"A{trig[-1]}" if trig.startswith("ACTION") and len(trig) == 7 else trig


class Rule:
    """One rule, or (parts) one selector and verb with a parameter per action."""

    def __init__(self, trig, sel, eff, cond, hits, false, explained, parts=None):
        self.trig, self.sel, self.eff, self.cond = trig, sel, eff, cond
        self.hits, self.false, self.explained = hits, false, explained
        self.parts = parts

    def for_action(self, action):
        """The rule (or the family's part) that applies to this action, or None."""
        if self.parts:
            return next((p for p in self.parts if p.trig == action), None)
        return self if trigger_matches(self.trig, action) else None

    def fires_on(self, step, i):
        if i is not None and not applicable(self.eff, step.state, i):
            return False
        if self.cond:
            pol, f = self.cond
            return (f in features(step, i, self.eff)) == (pol == "if")
        return True

    def _how(self, sel_shown=True):
        e = self.eff
        sel = f"{self.sel}, " if sel_shown else ""
        return {"move": lambda: f"move({sel}({e[1][0]:+d},{e[1][1]:+d}))",
                "recolor": lambda: f"recolor({sel}{{{', '.join(f'{a}:{b}' for a, b in e[1])}}})",
                "turn": lambda: f"turn({sel}{e[1]})",
                "resize": lambda: f"resize({sel}{e[1]:+d})",
                "reshape": lambda: f"reshape({sel}to {e[1]})",
                "vanish": lambda: f"vanish({self.sel})",
                "appear": lambda: f"appear({e[1]}, {e[2][0]}{'' if len(e[2]) == 1 else ' ' + str(e[2][1])})"}[e[0]]()

    def text(self):
        if self.parts:
            per = " | ".join(f"{_label(p.trig)}{' ' + p.cond[0] + ' ' + p.cond[1] if p.cond else ''} "
                             + (p._how(False)[len(p.eff[0]) + 1:-1] if p.eff[0] != "vanish" else "")
                             for p in self.parts)
            return f"when {'|'.join(p.trig for p in self.parts)} then {self.eff[0]}({self.sel}, by action: {per})"
        e = self.eff
        how = {"move": lambda: f"move({self.sel}, ({e[1][0]:+d},{e[1][1]:+d}))",
               "recolor": lambda: f"recolor({self.sel}, {{{', '.join(f'{a}:{b}' for a, b in e[1])}}})",
               "turn": lambda: f"turn({self.sel}, {e[1]})",
               "resize": lambda: f"resize({self.sel}, {e[1]:+d})",
               "reshape": lambda: f"reshape({self.sel}, to {e[1]})",
               "vanish": lambda: f"vanish({self.sel})",
               "appear": lambda: f"appear({e[1]}, {e[2][0]}{'' if len(e[2]) == 1 else ' ' + str(e[2][1])})"}[e[0]]()
        cond = f" {self.cond[0]} {self.cond[1]}" if self.cond else ""
        return f"when {self.trig}{cond} then {how}"

    def line(self):
        return f"{self.text():<62s} # {self.hits} right, {self.false} wrong"

    def uses_names(self):
        if self.parts:
            return any(p.uses_names() for p in self.parts)
        return self.sel.startswith(("pic:", "touching:")) or self.eff[0] in ("reshape", "appear")


class _Cand:
    __slots__ = ("key", "inst", "hits", "n", "dead")

    def __init__(self, key):
        self.key = key            # (trigger, selector, effect)
        self.inst = []            # (step, obj, explained event ids, features)
        self.hits = self.n = 0
        self.dead = False


class Learner:
    """Rules for one level, re-chosen after every step from everything seen on the level so far."""

    def __init__(self, min_count=3):
        self.min_count = min_count
        self.steps = []
        self.cands = {}
        self.proposed = collections.Counter()
        self._rules = None

    def observe(self, state, action, click, events):
        step = Step(state, action, click, events)
        t = len(self.steps)
        self.steps.append(step)
        for c in self.cands.values():
            if not c.dead:
                self._score(c, t)
        for e in events:
            sels = ["none"] if e["verb"] == "appear" else selectors_for(step, e["obj"])
            for eff in effects_for(step, e):
                for trig in (action, "any"):
                    for sel in sels:
                        key = (trig, sel, eff)
                        if key in self.cands:
                            continue
                        self.proposed[key] += 1
                        if self.proposed[key] >= 2:        # seen twice: worth scoring on the whole level
                            c = self.cands[key] = _Cand(key)
                            for s in range(t + 1):
                                self._score(c, s)
            if e["verb"] == "recolor":                   # also propose the union of this selector's maps
                self._propose_union(step, e, t)
        self._rules = None

    def _propose_union(self, step, e, t):
        for trig in (step.action, "any"):
            for sel in selectors_for(step, e["obj"]):
                union = dict(e["map"])
                for (tg, sl, ef) in self.cands:
                    if tg == trig and sl == sel and ef[0] == "recolor":
                        for a, b in ef[1]:
                            if union.setdefault(a, b) != b:
                                break
                key = (trig, sel, ("recolor", tuple(sorted(union.items()))))
                if key not in self.cands and len(union) > len(e["map"]):
                    c = self.cands[key] = _Cand(key)
                    for s in range(t + 1):
                        self._score(c, s)

    def _score(self, c, s):
        trig, sel, eff = c.key
        step = self.steps[s]
        if not trigger_matches(trig, step.action):
            return
        if eff[0] == "appear":
            ev = tuple(k for k, e in enumerate(step.events) if matches(eff, None, e, step.click))
            c.inst.append((s, None, ev, features(step, None, eff)))
            c.n += 1
            c.hits += bool(ev)
        else:
            st = step.state
            for i in step.select(sel):
                if not applicable(eff, st, i):
                    continue
                ev = tuple(k for k, e in enumerate(step.events) if matches(eff, i, e, step.click))
                c.inst.append((s, i, ev, features(step, i, eff)))
                c.n += 1
                c.hits += bool(ev)
        if c.n >= 30 and c.hits < 0.1 * c.n:              # hopeless: stop scoring it
            c.dead = True
            c.inst = []

    def _options(self, c):
        """The candidate as it stands, and with its best condition."""
        trig, sel, eff = c.key
        base_ex = {(s, k) for s, _, ev, _ in c.inst for k in ev}
        out = [(trig, sel, eff, None, c.hits, c.n - c.hits, base_ex)]
        miss = c.n - c.hits
        if miss >= self.min_count:
            hit_f, miss_f = collections.Counter(), collections.Counter()
            for _, _, ev, f in c.inst:
                (hit_f if ev else miss_f).update(f)
            best, best_gain = None, 0
            for f in set(hit_f) | set(miss_f):
                for pol, h, m in (("unless", c.hits - hit_f[f], miss - miss_f[f]), ("if", hit_f[f], miss_f[f])):
                    gain = (miss - m) - (c.hits - h)
                    if h >= self.min_count and gain > best_gain:
                        best, best_gain = (pol, f), gain
            if best:
                pol, f = best
                keep = [(s, ev) for s, _, ev, fs in c.inst if (f in fs) == (pol == "if")]
                hits = sum(bool(ev) for _, ev in keep)
                out.append((trig, sel, eff, best, hits, len(keep) - hits, {(s, k) for s, ev in keep for k in ev}))
        return out

    def rules(self):
        """Greedy minimum-description choice among the scored candidates (and families of them)."""
        if self._rules is not None:
            return self._rules
        options, per_action = [], collections.defaultdict(dict)
        for c in self.cands.values():
            if c.dead or not c.hits:
                continue
            opts = [Rule(*o) for o in self._options(c)]
            if c.hits >= self.min_count:
                options += opts
            trig, sel, eff = c.key
            if trig != "any" and eff[0] != "appear":
                best = per_action[(sel, eff[0])]
                for o in opts:
                    cur = best.get(trig)
                    if cur is None or (o.hits - o.false, -_cost(o)) > (cur.hits - cur.false, -_cost(cur)):
                        best[trig] = o
        for (sel, verb), best in per_action.items():          # one selector and verb, a parameter per action
            if len(best) >= 2:
                parts = [best[a] for a in sorted(best)]
                options.append(Rule("by_action", sel, parts[0].eff, None, sum(p.hits for p in parts),
                                    sum(p.false for p in parts), set().union(*(p.explained for p in parts)), parts))
        covered, chosen = set(), []
        while True:
            best = None
            for o in options:
                new = len(o.explained - covered)
                if new < self.min_count:
                    continue
                gain = new - o.false - _cost(o)
                key = (gain, -_cost(o), o.text())
                if gain > 0 and (best is None or key > best[0]):
                    best = (key, o)
            if best is None:
                break
            chosen.append(best[1])
            covered |= best[1].explained
            options.remove(best[1])
        self._rules = chosen
        return chosen

    def residual(self):
        """Events of the level that no chosen rule explains: (step, event)."""
        covered = set().union(*(r.explained for r in self.rules())) if self.rules() else set()
        return [(s, e) for s, st in enumerate(self.steps) for k, e in enumerate(st.events) if (s, k) not in covered]


def _cost(r):
    """Description length of a rule; a family pays for its selector and verb once."""
    if r.parts:
        return _cost(r.parts[0]) + sum(0.25 + (COST["condition"] if p.cond else 0) for p in r.parts[1:])
    return cost(r.trig, r.sel, r.eff, r.cond)


def predict(rules, state, action, click=None):
    """Closed-world prediction for the next step: [(rule, object index or None, effect)]."""
    step = Step(state, action, click, [])
    out = []
    for rule in rules:
        r = rule.for_action(action)
        if r is None:
            continue
        if r.eff[0] == "appear":
            if r.fires_on(step, None):
                out.append((r, None, r.eff))
            continue
        for i in step.select(r.sel):
            if r.fires_on(step, i):
                out.append((r, i, r.eff))
    return out, step.click


def score(predictions, click, events):
    """(predictions right, predictions, events explained, events, exact) for one step."""
    right, explained = 0, set()
    for _, i, eff in predictions:
        ks = [k for k, e in enumerate(events) if matches(eff, i, e, click)]
        right += bool(ks)
        explained |= set(ks)
    exact = right == len(predictions) and len(explained) == len(events)
    return right, len(predictions), len(explained), len(events), exact
