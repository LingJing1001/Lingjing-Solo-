# AOP Stage 4 candidate v2 — runtime-aligned action scoring

- Status: **supported-action quality improved; offline/shadow-only**
- Runtime authority: **false**
- Frozen dataset: Stage 3 packet (`1,572` train / `1,552` holdout rows)
- Seed / epochs: `20261013` / `200`
- Trainer/source commit: `8ba47494c2cabc5c3d01c2009c0c4d36291c2fb8`
- Checkpoint SHA-256: `d023ad71f3203f0c5de8cd1ee4acf783146c650b3c0f05c3a4502f110371d804`

## Root cause fixed

The v1 trainer evaluated a single row whose input already contained the requested candidate action, while runtime scores every legal candidate separately and compares each candidate's diagonal logit. That mismatch rewarded candidate echo and overstated action-selection quality. V2 trains and evaluates with the exact candidate-scoring shape used by `tools/aop/evaluate.py` and the controller.

## Independent runtime-aligned result

Compared with the v1 checkpoint under the same independent evaluator:

- raw action accuracy: `0.164948 → 0.251289` (`+0.086340`)
- masked action accuracy: `0.272552 → 0.383376` (`+0.110825`)
- masked correct: `595 / 1,552`
- unknown-legal rows: `0`
- val loss: `3.137147 → 1.585523` (`1.551624` improvement from the same-seed baseline)

## Holdout masked accuracy by requested action

- ACTION1: `0.194444` (support `252`)
- ACTION2: `0.063910` (support `266`)
- ACTION3: `0.312757` (support `243`)
- ACTION4: `0.415984` (support `488`)
- ACTION5: `0.687500` (support `16`)
- ACTION6: `0.832753` (support `287`)
- ACTION7: **unverified** (support `0`)

## Reproducibility and readiness

A second run with the same seed produced all 18 state tensors and all baseline/trained/delta metrics identically. Trainer and independent evaluator produced identical correct counts (`390` raw, `595` masked). V2 is still not deployment-ready: overall masked accuracy is only `0.383376`, ACTION2 remains weak, ACTION5 has only 16 holdout rows, and ACTION7 has no target support. Deterministic/legal-action runtime remains the sole authority.
