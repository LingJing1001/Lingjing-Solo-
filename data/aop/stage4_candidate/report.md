# AOP Stage 4 — real frozen-holdout evaluation

- Status: **training/evaluation complete; candidate rejected for deployment**
- Runtime authority: **false**
- Dataset: Stage 3 packet (`1,572` train / `1,552` holdout rows)
- Seed / epochs: `20261013` / `20`
- Trainer commit: `213640c3580a2a181eaf7e4b4cb9a87311367da1`
- Checkpoint SHA-256: `0bc98e1c0ec13d56ad66c23a61a2a7f30e61199afb081e7a8c9e395ce96b04a9`

## Holdout deltas

- val_loss: `3.134462 → 1.941367` (improvement `+1.193095`; lower is better)
- action raw accuracy: `0.003222 → 0.166237` (`+0.163015`)
- action masked accuracy: `0.296392 → 0.273840` (**-0.022552 regression**)
- action-effective accuracy: `0.120490 → 0.896907` (`+0.776418`)
- progressed accuracy: `0.878222 → 0.985825` (`+0.107603`)

## Fail-closed readiness decision

The candidate is not deploy-ready because masked action accuracy regressed, ACTION2/ACTION4/ACTION5 masked accuracy is `0`, and ACTION7 has no target support in the frozen holdout. The checkpoint remains an offline candidate only; deterministic/legal-action runtime authority is unchanged.
