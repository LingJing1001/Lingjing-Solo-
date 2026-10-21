"""Champion prompt text, ported verbatim where it survives the port.

The wording here is not mine. Every block marked PORTER carries the champion
notebook's own sentences (`inference/agent/prompts.py`, cells 3 and 21). Wording
matters more than usual in this harness: the champion's notes record that their
ARCAID3 lexicon view was added, wired correctly, and then used in 0 of 1130
calls -- because the system prompt still described the old set of fields. A
capability the prompt does not name does not exist.

What I did NOT port: their tool schema, their turn loop, and their retry
machinery all live in files the patch only modifies, so the originals are not in
the notebook. Those parts are re-derived in `agent.py` and obviously thinner.
"""
from __future__ import annotations

# --------------------------------------------------------------------- preamble

# PORTER: the definitions the champion's system prompt relies on (flag semantics,
# what an action costs). Their exact full system prompt is in a file the patch
# only patches, so this preamble is a reconstruction around the sentences I do
# have -- it says the same things in fewer words.
SYSTEM_PREAMBLE = (
    "You are playing a game you have never seen. You see only the current board, "
    "as a grid of colours, plus a history of what you have already done. The rules "
    "are not documented anywhere and nobody will tell you them.\n\n"
    "You act by running Python in a sandbox. Reading and computing cost nothing; "
    "only `action(...)` spends the level budget. The sandbox exposes:\n"
    "- `current_frame`: `.ascii` (rows of colour characters), `.shape` (rows, cols), "
    "`.segmentation['nodes']` (4-connected same-colour objects, each with `id`, "
    "`color`, `pixels`, `bbox`, `hash`, `children`).\n"
    "- `previous_frame`: the frame before the last action.\n"
    "- `history`: list of entries with `.action`, `.frame`, `.result`.\n"
    "- `transitions`: list of actual action transitions with `.action`, "
    "`.before_frame`, `.after_frame`, `.result`.\n"
    "- `frame_diff(before=None, after=None)`: object-level diff; defaults to "
    "previous_frame -> current_frame.\n"
    "- `action(actions)`: execute one action or a short batch. Accepts "
    "'UP'/'DOWN'/'LEFT'/'RIGHT'/'SPACE'/'UNDO'/'RESET' (or ACTION1..6), or "
    "`{'action': 'MOUSE', 'row': r, 'col': c}`. Returns a dict describing what "
    "happened.\n"
    "- `last_action_call_result`: the result of the most recent `action(...)`.\n"
    "- `valid_actions`: names you may use right now.\n\n"
    "Flag semantics: `game_over` = this attempt FAILED (death or a budget ran out); "
    "the level resets to its start and completed levels are kept -- it never means "
    "the run is won. `level_completed` = advanced one level. `run_complete`/`done` = "
    "the whole game is won. Deaths come from a hazard cell or from a step/time "
    "budget depleting -- watch for a bar at the grid border that shrinks each "
    "action, and count remaining budget into your plans.\n"
)

# PORTER: WORLD_MODEL_ADDENDUM, verbatim. This is the whole of the champion's
# "reasoning strategy": maintain a model, and when unsure, run an experiment that
# separates two hypotheses. There is no goal detector anywhere in their harness.
WORLD_MODEL_ADDENDUM = (
    "- Maintain a compact working world model: what entities or regions exist, what actions "
    "seem to do, what the goal likely is, what remains uncertain, and what plan best fits the "
    "evidence so far.\n"
    "- Optimize for the shortest reliable sequence that advances the current goal as described "
    "by your world model. If confidence is low, program a discriminating probe and revise the "
    "world model from the result.\n"
)

# PORTER: STEP_VERIFICATION_ADDENDUM. The second paragraph matters as much as the
# first: it tells the model that a failed search is evidence about the model, not
# about solvability.
STEP_VERIFICATION_ADDENDUM = (
    "- During a multi-action plan, verify important effects by comparing the actual game "
    "state with your predictions—for example, changed cells, object properties, positions, "
    "or counters. Make these comparisons in code so the batch can stop when a mismatch "
    "undermines the remaining plan. Test uncertain interactions before committing to long "
    "batches.\n"
    "- If your search finds no solution under your current model of the game, remember "
    "that the game is solvable. Reconsider your mechanics, goal, search implementation, "
    "or search limits, including interactions with new elements. Take a targeted action "
    "to test an uncertain rule or overlooked interaction, then update your model from "
    "the result.\n"
)

# PORTER: NOOP_GUARD_ADDENDUM. Note the escape hatch -- the guard is overridable by
# immediately re-issuing the action, because a hard block makes the refutation
# unreachable. We hit exactly that failure with a hard-coded blacklist.
NOOP_GUARD_ADDENDUM = (
    "- An action already proven to change nothing in this EXACT board state is refused before "
    "execution: it raises `KnownNoOpActionError`, which is NOT catchable and ends the snippet - "
    "nothing runs, and no action budget is spent. `stop_reason` is `known_noop`. If you have a "
    "specific reason to believe the action will behave differently now, issue that same action "
    "again as your very next action and it will execute.\n"
)

# PORTER: DEATH_LEDGER_ADDENDUM. The no-op guard's cheaper twin: a no-op burns one
# action, a fatal action burns the attempt. Same escape hatch, for the same reason.
DEATH_LEDGER_ADDENDUM = (
    "- An action that ended the run from this EXACT board last time is refused the same way: "
    "it raises `KnownDeadlyActionError`, nothing runs, no budget is spent, `stop_reason` is "
    "`known_deadly`. Issue it again if you think the outcome differs -- but if you have no "
    "reason to think so, that action is the one thing you already know does not work here.\n"
)

# PORTER: PREFER_TOOL_CALLS_LINE. Their measurement: 2000 tokens of reasoning costs
# far more wall-clock than running the code that settles the question.
PREFER_TOOL_CALLS_LINE = (
    "- Reading and computing cost nothing; only `action(...)` spends the level budget. When you "
    "are unsure, prefer another tool call over more reasoning: the code answers what the "
    "reasoning would only guess at, and running it is faster than thinking your way to the "
    "same answer.\n"
)

# PORTER: PERSISTENT_FUNCTIONS_LINE, trimmed to what our sandbox actually supports.
PERSISTENT_FUNCTIONS_LINE = (
    "- Python variables reset between tool calls. Eligible functions you define are "
    "retained automatically for the rest of this level. Call your retained functions "
    "directly without repeating their definitions. Redefine a function when you need to "
    "change it.\n"
    "- For a function to be retained, its dependencies must also be available next call. "
    "Pass snippet-specific data as arguments. Use plain top-level definitions without "
    "decorators or annotations, and literal defaults.\n"
)

# PORTER: LEVEL_START_USER_PROMPT, verbatim. "Do not rediscover them without
# reason" is the single line that makes cross-level transfer cheap.
LEVEL_START_USER_PROMPT = (
    "You have completed the previous level. `current_frame` now contains the starting "
    "board of the next level. Build a new plan for this layout rather than continuing "
    "the previous level's action sequence.\n\n"
    "Start from the mechanics you established on the previous level; do not rediscover "
    "them without reason. Inspect the new board for unfamiliar elements, changed "
    "arrangements, or interactions your previous understanding does not explain. New "
    "elements often introduce mechanics needed to solve this level, so prioritize small, "
    "informative tests when their behavior is unclear.\n\n"
    "Reassess the goal: does the previous objective still apply, now requiring the new "
    "mechanics, or does the evidence suggest a different objective? Combine retained "
    "knowledge with new findings to plan for this board."
)

# PORTER: SUMMARY_REQUEST_PROMPT, verbatim. "Length is not a problem here; leaving
# something out is" is doing real work -- earlier summaries get dropped too.
SUMMARY_REQUEST_PROMPT = (
    "Summarize everything above: what you have established about this game, what you have "
    "ruled out, what you were in the middle of, and anything else a later turn would need. "
    "Keep uncertain ideas tentative, and preserve unresolved alternatives. "
    "Do not act and do not call any tool - write the summary only.\n"
    "Write it to stand completely on its own. Everything above this point will eventually be "
    "dropped and replaced by this summary, and earlier summaries go with it, so anything you "
    "leave out is gone.\n"
    "Length is not a problem here; leaving something out is."
)

# --------------------------------------------------------------------- assembly


def build_system(*, level_transfer: bool = True) -> str:
    """System prompt. Order follows the champion: definitions, then strategy, then
    guards, then the sandbox conventions."""
    parts = [SYSTEM_PREAMBLE, "World model:\n", WORLD_MODEL_ADDENDUM]
    parts.append("Verification:\n")
    parts.append(STEP_VERIFICATION_ADDENDUM)
    parts.append("Sandbox conventions:\n")
    parts.append(PERSISTENT_FUNCTIONS_LINE)
    parts.append(PREFER_TOOL_CALLS_LINE)
    parts.append(NOOP_GUARD_ADDENDUM)
    parts.append(DEATH_LEDGER_ADDENDUM)
    if level_transfer:
        parts.append(
            "- Levels usually build on mechanics learned in earlier levels. Carry forward "
            "supported knowledge as a starting hypothesis, while re-checking anything "
            "contradicted by new evidence.\n")
    return "".join(parts)


def build_opener(*, ascii_board: str, shape: tuple[int, int], valid: list[str],
                 level: int, step: int, need_action: bool = True) -> str:
    """The per-turn user message.

    The board is inlined rather than left for the model to fetch: it is the one
    thing every turn needs, and a turn that begins by printing the board has
    already spent the turn.
    """
    head = f"Level {level}, step {step}. `current_frame` is:\n\n{ascii_board}\n\n"
    head += f"shape={shape[0]}x{shape[1]}  valid_actions={valid}\n"
    if need_action:
        head += ("Decide what to do next. Inspect or compute first if that is what you "
                 "need, then call `action(...)`. One short batch is usually better than "
                 "one action at a time, but do not batch past something you are unsure of.")
    return head
