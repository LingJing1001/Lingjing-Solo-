# Lingjing-Solo ARC-AGI-3 Adapter Architecture

## Status

- Branch: `feature/arc-strategy-registry`
- Scope: restructure the ARC boundary adapter without changing the Lingjing core.
- Current validated game: `LS20`
- Next game: intentionally out of scope until LS20 is re-verified on this architecture.

## Decision

Do not create one independent solver for every ARC game. Use one stable agent core and resolve a game strategy at runtime:

```text
ARC FrameData/GameAction
          |
          v
  LingjingSolo ARC Adapter
          |
          v
  GameStrategyRegistry
       /           \
LS20Strategy    GenericStrategy
       |               |
validated route   Lingjing core exploration/planning
```

A game receives a specialized strategy only when generic exploration cannot provide a stable route. A route artifact is data, not a new solver class.

## Repository tree

```text
arc_adaptor/
├── agents/
│   ├── __init__.py
│   ├── templates/
│   │   └── lingjing_solo_agent.py   # thin ARC boundary adapter
│   └── strategies/
│       ├── __init__.py
│       ├── base.py                  # strategy protocol
│       ├── generic.py               # default fallback
│       ├── ls20.py                  # validated LS20 policy
│       └── registry.py              # game_id -> strategy resolution
├── tests/
│   ├── test_lingjing_solo_agent.py
│   └── test_action_recording.py
├── tools/
│   └── ls20_single_action_probe.py
├── patches/
│   └── arc-agent-recording.patch
├── MANIFEST.md
└── sync_to_arc.sh
```

The long-term route layout is:

```text
routes/
├── ls20/
│   ├── level_0.json
│   └── ...
└── <game-id>/
    └── README.md
```

The sync script intentionally does not overwrite the ARC checkout's native `agents/__init__.py`; the ARC registration import remains an ARC-side integration change. This prevents the bundle from deleting unrelated ARC exports.

The current LS20 route remains supplied by the existing Lingjing planning implementation for compatibility. Moving the route into versioned data files is a follow-up migration after the adapter structure is validated.

## Component contracts

### `LingjingSolo`

Owns only ARC lifecycle concerns:

- convert `FrameData.frame` to a numpy grid;
- normalize legal actions;
- call `observe()` on the Lingjing core;
- handle RESET and terminal states;
- resolve the strategy through the registry;
- convert a strategy action name back to `GameAction`;
- apply the safe legal-action fallback.

It must not contain game-specific route branches.

### `GameStrategy`

The strategy boundary receives the current frame, grid, legal action names, and completed-level count. It returns an abstract action name or `None`.

Required lifecycle:

```python
strategy.reset(frame)
strategy.choose_action(frames, frame, grid, legal_names, levels_completed)
```

### `GenericStrategy`

Default for unknown games. It delegates to the general Lingjing core and does not assume a game-specific action sequence. If the core proposes an illegal/unknown action, the adapter selects the first legal action as a fail-closed fallback.

### `LS20Strategy`

Owns LS20-only behavior:

- LS20 default level plans;
- explicit `LINGJING_LS20_PLAN` override;
- LS20 solver reset and level reseeding;
- action selection from the validated route.

No other game may import or depend on this strategy.

### `GameStrategyRegistry`

The only place that maps game identifiers to specialized policies:

```python
if game_id.startswith("ls20"):
    return LS20Strategy(...)
return GenericStrategy(...)
```

Future games add a profile or strategy registration here, not another branch in `LingjingSolo.choose_action()`.

## Game onboarding policy

Each new game follows this progression:

1. **Generic mode** — reset, probe legal actions, record frame/action/state changes.
2. **Game profile** — record action family, reset behavior, coordinate requirements, and level count.
3. **Route artifact** — store a replayable route when a stable route is discovered.
4. **Specialized strategy** — add only if stateful behavior or route selection cannot be represented by the generic planner plus data.
5. **Replay verification** — run the route through the same adapter and compare terminal state, action legality, and level progress.

This prevents the one-solver-per-game anti-pattern while still allowing difficult games to receive targeted policies.

## LS20 re-verification gate

The restructure is accepted only if all of the following pass:

- the registry resolves `ls20-9607627b` to `LS20Strategy`;
- LS20 reset loads the expected level-0 plan;
- explicit LS20 plan override still works;
- every emitted action is legal or safely falls back;
- terminal states remain recognized;
- the official ARC harness completes an LS20 run and returns observable output;
- recording tests continue to pass when the optional recording patch is applied.

R11L and other new games are deliberately not part of this validation gate.

## Failure and safety behavior

- Empty legal action list: return RESET.
- Unknown game id: use `GenericStrategy`.
- Unknown strategy action: use the first legal action.
- Complex action without coordinates: set the existing neutral coordinate payload.
- Already-applied recording patch: do not apply it twice; verify the target diff and run recording tests.
- No external credentials or secret values are stored in this bundle.

## Verification commands

From the ARC-AGI-3-Agents checkout:

```bash
bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh .
uv run pytest -q tests/unit/test_lingjing_solo_agent.py
uv run pytest -q tests/unit/test_lingjing_solo_agent.py tests/unit/test_action_recording.py
uv run python main.py -a lingjingsolo -g ls20-9607627b -t restructure,ls20-reverify
```

The final online run must be reported with its actual exit code and harness/Scorecard evidence. A passing unit test alone is not sufficient evidence of game completion.
