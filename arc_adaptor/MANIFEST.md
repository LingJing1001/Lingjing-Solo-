# ARC reproduction bundle

This directory contains the versioned files needed to reproduce the Lingjing-Solo ARC LS20 run without committing the complete official ARC workspace.

## Source versions

- Lingjing-Solo branch: `fix/ls20-plan-reseed`
- Lingjing-Solo commit when prepared: `5b476ca269640cc917c3399a985d6d8196c9c9e1`
- ARC upstream baseline when prepared: `4743e7d0aaae0ded0d98a89a7e282e63564cd58b`
- ARC upstream: https://github.com/arcprize/ARC-AGI-3-Agents

## Bundle contents

- `agents/`: required ARC boundary adapter, strategy registry, game strategies, and registration.
- `agents/strategies/`: `GameStrategy` protocol, generic fallback, and the isolated LS20 route strategy.
- `tests/`: adaptor tests; `test_action_recording.py` requires the optional recording patch.
- Three tests shared from the Lingjing repo root `tests/` (not from this directory), synced to
  `tests/unit/`:
  `test_plan_contract.py` (the §3.2 nine-field contract itself), `test_ar25_plan_contract.py`
  (the AR25 runner's plan exit), and `test_ar25_recording.py` (the AR25 per-tick evidence layer:
  `r2_ar25` state hash + `tick_trail` JSONL writer). Both AR25 tests **skip** inside an ARC
  checkout. For the plan-contract test the reason is that its subject,
  `agents/strategies/run_ar25_r234.py`, imports `arc_adaptor/paths.py`, which does not exist
  there; the skip reports that import error rather than passing silently. For the recording test
  the reason is that its subjects, `arc_adaptor/r2_ar25.py` and `arc_adaptor/tick_trail.py`, are
  engine-side and deliberately outside the sync whitelist — its module-level skip is evaluated
  *before* those imports, so carrying it over cannot break collection of the other `tests/unit`
  files. They are still synced because "skips, does not error" is itself the reproducible
  boundary evidence for rule ①.
- `tools/`: optional online single-action probes for LS20 and R11L.
- `patches/arc-agent-recording.patch`: optional ARC `Agent` recording enhancement. Apply only when recording requested actions is needed.
- `sync_to_arc.sh`: copies the bundle into an existing ARC checkout and can apply the optional patch.

## Sync

From an ARC checkout:

```bash
bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh
```

To also apply the optional recording patch:

```bash
bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh . --with-recording-patch
```

The script never copies `.env`, `.venv`, recordings, or API keys. It does copy `__pycache__/`
today: `sync_to_arc.sh` copies `agents/strategies/` with `cp -R`, so compiled caches from that
directory arrive in the ARC checkout as a known, declared gap (fixing the copy mode is a code
change, still gated on the maintainer's decision).

## Expected LS20 plan

- Levels: L1-L7
- Actions: 309
- Expected online acceptance: `levels_completed >= 7`, positive score, `state=WIN`
