# V14.2 DPL integration snapshot

Source archive: `C:\Users\wanga\Documents\xwechat_files\wxid_t979q1oefd5q11_bcbf\msg\file\2026-09\lingjing_v14.2.zip`

Imported under `vendor/lingjing_v14/` on branch `feature/dpl-v14.2-unknown-bench-400`.

## Included

- V14.2 contract-layer implementation: `core/writeback.py`, `core/scenario.py`
- Conservation field and numerical evolution: `core/field.py`, `core/laplacian.py`
- Causal engine loop and evacuation scenario: `engine.py`, `agents.py`, `scenarios/evacuation.py`
- Persistence/replay helpers, diagnostics, benchmark scripts, architecture/changelog
- Contract, end-to-end, interface, and historical regression tests

Generated caches and output images/trajectories from the archive were intentionally omitted.

## Verification

- `python regress.py`: 4/4 suites passed; C1-C9, E1-E7, and 10 historical checks passed.
- Root package: `158 passed, 18 skipped`.
- ARC Starter: `53 passed`.
- 25-game ARC local run: maximum 400 actions per game; evidence is
  `integrations/ARC-AGI-3-Kaggle-Starter/ui/static/games_benchmark_dpl_v14.2_400.json`.

The V14.2 physical-field engine is vendored and verified independently. It is not yet
wired as the action-transition engine for ARC's 2-D game environments; therefore the
25-game score is an ARC-agent integration baseline, not proof of a V14.2 score uplift.
