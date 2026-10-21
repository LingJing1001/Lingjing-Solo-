# ACTION7 supplement candidate — shadow/rejected

- Candidate packet: `data/aop/stage3_packet_action7_candidate/`
- Input: 3 QA-passed ar25 ACTION7 shards, 20 counterfactual groups / 40 rows total.
- Split: train `1,576`, holdout `1,552`; group keys disjoint; packet freeze status `complete`.
- ACTION7 holdout support: `4` (ACTION7 train support is `4`).
- Checkpoint: `data/aop/stage4_candidate_action7`
- Independent evaluator: `data/aop/stage4_candidate_action7_runtime_evaluation.json`

## Metrics

- masked accuracy: `0.531572` (`825/1552`), below profile-prior v4 `0.550258`.
- raw accuracy: `0.255155`.
- val_loss: `1.282051`.
- ACTION7 masked accuracy: `0.750000`, support `4`.
- ACTION2 masked accuracy: `0.522556`.
- ACTION5 masked accuracy: `0.375000`, support `16`.
- unknown legal rows: `0`; masked legal prediction rate: `1.0`.

## Decision

**Rejected for deployment; shadow-only.** The ACTION7 signal is present and runtime-aligned, but only four holdout rows are available and overall masked accuracy regressed versus v4. Keep the packet/checkpoint as evidence; collect more independent ACTION7 groups before another training decision. `runtime_authority=false`.
