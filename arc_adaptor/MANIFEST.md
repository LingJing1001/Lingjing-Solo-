# ARC reproduction bundle

This directory contains the versioned files needed to reproduce the Lingjing-Solo ARC LS20 run without committing the complete official ARC workspace.

## Source versions

- Lingjing-Solo branch: `fix/ls20-plan-reseed`
- Lingjing-Solo commit when prepared: `5b476ca269640cc917c3399a985d6d8196c9c9e1`
- ARC upstream baseline when prepared: `4743e7d0aaae0ded0d98a89a7e282e63564cd58b`
- ARC upstream: https://github.com/arcprize/ARC-AGI-3-Agents

## Bundle contents

- `agents/`: required ARC adaptor and registration.
- `tests/`: adaptor tests; `test_action_recording.py` requires the optional recording patch.
- `tools/`: optional online single-action probe.
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

The script never copies `.env`, `.venv`, recordings, `__pycache__`, or API keys.

## Expected LS20 plan

- Levels: L1-L7
- Actions: 309
- Expected online acceptance: `levels_completed >= 7`, positive score, `state=WIN`
