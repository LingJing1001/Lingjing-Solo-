"""Sandbox side of `current_frame.rules` (notebook version 3). Stdlib only.

Given one level's history as (action text, frame) pairs, it reads every frame with the lexicon,
learns rules with the rule grammar (design/2026-10-07__rule-grammar/rule_grammar.py) from the level's
recent steps, and reports them with the changes of the last step that no rule explained.

The harness writes actions as model-facing strings: UP, DOWN, LEFT, RIGHT, SPACE, UNDO, RESET, and
MOUSE(row=R, col=C) with R, C on the 64x64 screen. The level's first entry has an empty action.
"""
import re
import time

_MOUSE = re.compile(r"^\s*MOUSE\s*\(\s*(?:row\s*=\s*)?(-?\d+)\s*,\s*(?:col\s*=\s*)?(-?\d+)\s*\)\s*$", re.I)
_COLOUR = re.compile(r"\b(colour:|touch:c|path:c|click:c)(\d+)\b")
_MAP = re.compile(r"\{(\d+:\d+(?:, \d+:\d+)*)\}")


def parse_action(text):
    """'MOUSE(row=4, col=7)' -> ('MOUSE', (7, 4)) as screen (x, y); 'UP' -> ('UP', None)."""
    m = _MOUSE.match(text or "")
    if m:
        return "MOUSE", (int(m.group(2)), int(m.group(1)))
    return (text or "").strip().upper(), None


def with_chars(text, chars):
    """Write colours as the board's characters (colour:12 -> colour:O), as `.code` and `.ascii` do."""
    if not chars:
        return text

    def ch(n):
        n = int(n)
        return chars[n] if 0 <= n < len(chars) else str(n)

    text = _COLOUR.sub(lambda m: m.group(1).rstrip("c") + ch(m.group(2)) if m.group(1).endswith(":c")
                       else m.group(1) + ch(m.group(2)), text)
    return _MAP.sub(lambda m: "{" + ", ".join(f"{ch(a)}:{ch(b)}" for a, b in (p.split(":") for p in m.group(1).split(", "))) + "}", text)


def describe(e):
    v, s, at = e["verb"], e["sym"], e.get("at")
    if v == "move":
        return f"{s} moved by ({e['d'][0]:+d},{e['d'][1]:+d}) to {at}"
    if v == "recolor":
        return f"{s} at {at} recoloured {{{', '.join(f'{a}:{b}' for a, b in e['map'])}}}"
    if v == "turn":
        return f"{s} at {at} turned ({e['transform']})"
    if v == "resize":
        return f"{s} at {at} changed size by {e['delta']:+d} cells"
    if v == "reshape":
        return f"{s} at {at} became {e['to']}"
    if v == "vanish":
        return f"{s} vanished from {at}"
    return f"{s} appeared at {at}"


def report(g, seq, encode, level=None, chars=None, window=60, budget=8.0, max_rules=10, max_lines=6):
    """Text for `current_frame.rules`.

    g       the rule grammar's namespace (State, observe, Learner, predict, matches)
    seq     [(action text, frame)] for one level, oldest first; frame i is the result of action i
    encode  (frame, previous encoding or None) -> lexicon encoding (shared Lexicon, per-level scale
            lock; the previous encoding lets the lexicon cut merged objects apart)
    """
    t0 = time.time()
    start_frame = seq[0][1] if seq else None
    seq = seq[-(window + 1):]
    states = {}
    start_state = g["State"](encode(start_frame, None)) if start_frame is not None else None
    if seq and seq[0][1] is start_frame:
        states[0] = start_state

    def state(i):
        if i not in states:
            prev = states.get(i - 1)
            states[i] = g["State"](encode(seq[i][1], prev.code if prev is not None else None))
        return states[i]

    def is_reset_snap(a, b, evs):
        if start_state is None or b.code["native"] != start_state.code["native"] or a.code["native"] == start_state.code["native"]:
            return False
        return len(evs) >= 2 or any(e["verb"] == "move" and abs(e["d"][0]) + abs(e["d"][1]) > 1 for e in evs)

    steps, skipped, last = [], 0, len(seq) - 1
    i = 1
    while i <= last:
        if i < last and time.time() - t0 > budget:
            # out of time: skip the older steps still unread and read the newest one, so that
            # LAST STEP is always the action just taken
            skipped, i = last - i, last
        name, click = parse_action(seq[i][0])
        if name and name != "RESET":
            a, b = state(i - 1), state(i)
            if (a.code["scale"], a.code["offset"]) == (b.code["scale"], b.code["offset"]):
                evs = g["observe"](a, b)
                if not (i < last and is_reset_snap(a, b, evs)):
                    steps.append((a, name, click, evs, seq[i][0]))
        i += 1
    where = f"level {level}" if level is not None else "this level"
    if not steps:
        return f"RULES for {where}: no steps yet. Act, then look again."
    learner = g["Learner"]()
    for a, name, click, evs, _ in steps[:-1]:
        learner.observe(a, name, click, evs)
    preds, crc = g["predict"](learner.rules(), steps[-1][0], steps[-1][1], steps[-1][2])
    a, name, click, evs, text = steps[-1]
    right, explained = 0, set()
    for _, i, eff in preds:
        ks = {k for k, e in enumerate(evs) if g["matches"](eff, i, e, crc)}
        right += bool(ks)
        explained |= ks
    if not is_reset_snap(a, state(last), evs):
        learner.observe(a, name, click, evs)
    rules = learner.rules()
    lines = [f"RULES for {where}, learned from its last {len(steps)} steps "
             + (f"({skipped} older ones skipped for time) " if skipped else "")
             + "(when <action> [if/unless <relation>] then <change>; counts: right / wrong on this level):"]
    lines += ["  " + r.line() for r in rules[:max_rules]]
    if len(rules) > max_rules:
        lines.append(f"  ... and {len(rules) - max_rules} more")
    if not rules:
        lines.append("  (none yet: no change has repeated often enough)")
    unexplained = [e for k, e in enumerate(evs) if k not in explained]
    lines.append(f"LAST STEP {text}: rules predicted {len(preds)} change(s), {right} came true; "
                 f"{len(evs)} change(s) seen, {len(unexplained)} that no rule explains" + (":" if unexplained else "."))
    lines += ["  " + describe(e) for e in unexplained[:max_lines]]
    if len(unexplained) > max_lines:
        lines.append(f"  ... and {len(unexplained) - max_lines} more")
    return with_chars("\n".join(lines), chars)
