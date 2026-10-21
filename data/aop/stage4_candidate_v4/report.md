# AOP Stage 4 candidate v4 — profile-prior calibrated scoring

- Status: **improved offline candidate; shadow-only**
- Runtime authority: **false**
- Frozen dataset: Stage 3 packet (`1,572` train / `1,552` holdout rows)
- Seed / epochs: `20261013` / `800`
- Source commit: `457ed94470d68b5d9d824fb4fd616eadb2e34410`
- Checkpoint SHA-256: `4ff0d3f398318d260dfc8a8eb3dd592a77751bd24afcf78605eb821cdd3162de`

## Change

The model still scores every candidate independently and takes the diagonal candidate logit. v4 adds a smoothed profile×action prior learned from the training split only, with calibration weight `alpha=0.5`. The same prior is applied by the trainer, independent evaluator, and controller. No holdout target is used to construct the prior.

## Runtime-aligned result

- raw action accuracy: `0.469716`
- masked action accuracy: `0.550258`
- masked correct: `854 / 1,552`
- val loss: `1.255827`
- unknown-legal rows: `0`
- masked legal prediction rate: `1.0`

Compared with v3 (`0.485825` masked), v4 improves masked accuracy by `+0.064433`.

## Holdout masked accuracy by requested action

- ACTION1: `0.797619` (support `252`)
- ACTION2: `0.263158` (support `266`)
- ACTION3: `0.378601` (support `243`)
- ACTION4: `0.608607` (support `488`)
- ACTION5: `0.375000` (support `16`)
- ACTION6: `0.655052` (support `287`)
- ACTION7: **unverified** (support `0`)

## Readiness decision

V4 is an improved offline/shadow candidate. It is not deploy-ready: ACTION7 has zero target support, ACTION5 has only 16 holdout rows, and runtime authority/takeover remains disabled. Stage 5 remains unchecked.
