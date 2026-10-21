# AOP Stage 4 candidate v3 — extended runtime-aligned training

- Status: **improved offline candidate; shadow-only**
- Runtime authority: **false**
- Frozen dataset: Stage 3 packet (`1,572` train / `1,552` holdout rows)
- Seed / epochs: `20261013` / `800`
- Source commit: `6d13b3d6776012e9ab717a41b0eefd9110e8c578`
- Checkpoint SHA-256: `d5359ae2704721f953411fb3ef7d4fbdee07bc80e37c3abaa029b03aa39f37e7`

## Runtime-aligned result

The trainer and independent `tools/aop/evaluate.py` both use per-candidate encoding and diagonal action scores. On the unchanged frozen holdout:

- raw action accuracy: `0.405284`
- masked action accuracy: `0.485825`
- masked correct: `754 / 1,552`
- val loss: `1.359189` versus baseline `3.137146`
- unknown-legal rows: `0`

Compared with v2 (`0.383376` masked), v3 improves masked accuracy by `+0.102448`. Compared with the original v1 runtime result (`0.272552`), the exact improvement is `+0.213273`.

## Holdout masked accuracy by requested action

- ACTION1: `0.507937` (support `252`)
- ACTION2: `0.218045` (support `266`)
- ACTION3: `0.403292` (support `243`)
- ACTION4: `0.459016` (support `488`)
- ACTION5: `0.250000` (support `16`)
- ACTION6: `0.843206` (support `287`)
- ACTION7: **unverified** (support `0`)

## Readiness decision

V3 improves every previously weak supported action relative to v2 except ACTION5, which remains statistically weak because it has only 16 holdout rows. It is not deploy-ready: ACTION7 has zero target support, ACTION5 coverage is insufficient, and no runtime authority/takeover is enabled. Stage 5 remains unchecked.
