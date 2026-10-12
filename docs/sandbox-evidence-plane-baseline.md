# Sandbox Evidence Plane baseline

- Branch: `feature/sandbox-evidence-plane`
- Baseline commit: `3b184033474ba7b6fc5fea6fb6def097e6e41b5e`
- Python: `3.12.3`
- Authority mode: `shadow`
- External UGC / Workshop: `disabled`
- Production strategy write-back: `disabled`

## Input fixtures

The first local fixture set uses repository-owned, fixed recordings:

- `data/aop/episodes/ls20_success_rep0/recording.jsonl` — successful/partial ARC recording, validated by `models.aop.recording_gate`.
- `data/aop/episodes/dc22_fail_rep0/recording.jsonl` — failure recording.
- `data/aop/episodes/ar25_success_rep0/recording.jsonl` — RESET-bearing recording.

The adapter consumes these files read-only and emits `episode_events.v1.jsonl`; it does not claim the source recording is synthetic or alter it.

## Baseline verification

- [x] Branch created from updated `origin/main`.
- [x] Python version and baseline commit captured above.
- [x] Existing recording validator is reused by `RecordingAdapter`.
- [x] A successful/partial fixture and RESET-capable fixture are present in the repository.
- [ ] Existing AOP/DPL/SICA full test baseline is clean: current interpreter lacks optional `torch` (`ModuleNotFoundError` during `tests/test_aop_readiness.py` and `tests/test_aop_training.py` collection).
- [x] This phase does not claim ARC score improvement and does not enter authority control.

## Current implementation audit — 2026-10-12

The original Phase 1 commit was not fully complete: the adapter emitted a hard-coded recording gate result and had no executable DPL evaluator boundary. Its schema, deterministic conversion, metrics, and shadow registry were real and tested, but the DPL claim was incomplete. Phase 2 now provides the `SandboxAdapter` facade and an injected `dpl_evaluator` contract. The adapter records `dpl_status=not_configured` unless a real evaluator is supplied; it never fabricates a DPL pass.

## Phase 2 SandboxAdapter status

- [x] Read-only source hash before/after conversion.
- [x] Existing normalizer and recording gate reused.
- [x] Actual requested action and settled before/after ticks preserved.
- [x] `RESET` creates a new segment.
- [x] Injected DPL evaluator invoked once per event and its JSON result recorded.
- [x] Non-finite DPL results rejected.
- [x] CLI remains shadow-only and reports `dpl_status=not_configured` unless wired to a real DPL projection.
- [ ] Live DPL projection from recording frames into `lingjing_solo.dpl.types` — not yet implemented because the repository has no recording-to-DPL projection function.
- [ ] Production authority / external UGC — intentionally out of scope.


- [x] `EpisodeEvent v1` schema and fail-closed tests.
- [x] Deterministic recording adapter, provenance, replay hash, and manifest.
- [x] GateBench metric calculation and report writer.
- [x] Shadow candidate registry with cross-environment/evidence thresholds and rollback snapshot.
- [ ] Live/open Mod or remote UGC integration — intentionally out of scope.
- [ ] Production authority write-back — intentionally out of scope.

## Commands and limitations

Focused evidence tests: `pytest -q tests/evidence`.

The repository-wide suite must be run in an environment with the project optional dependencies installed. This branch does not install or enable `torch`; the baseline blocker is recorded rather than bypassed.
