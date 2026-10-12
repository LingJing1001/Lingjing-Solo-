# AOP training run: 20261010-v4

This run is persisted outside `/tmp` under the project data directory.

## Contents

- `datasets/merged_v4_labels.jsonl`: canonical 11,058-row training corpus.
- `reports/merged_v4_qa.json`: QA report for the canonical corpus.
- `checkpoints/aop_v2_candidate_fixed_20ep.pt`: checkpoint trained from the corpus.
- `collection/`: subsequent counterfactual collection and QA artifacts.
- `manifest.json`: SHA-256 hashes and provenance.

## Persistence policy

- `/tmp` is scratch space only.
- Every collection/training run gets a unique directory under `data/aop/runs/<run-id>/`.
- A run is admissible only when its manifest, input hashes, QA report, and checkpoint metadata are present.
- Large datasets/checkpoints are not committed to Git; the manifest and run README are the durable provenance records.
- Copy to external backup/object storage before deleting a run.
