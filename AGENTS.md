# Lingjing-Solo Mission Control Context

## Project scope

- Canonical project root: `/srv/agent-platform/projects/Lingjing-Solo-`
- This project covers Lingjing-Solo and the Appendix A analysis/evidence workflow.
- The Appendix A workflow is distinct from AOP model training. Do not relabel Appendix A tasks as model-training tasks.
- Canonical Python source is `lingjing_solo/`; tests are under `tests/`.

## Mission Control contract

- Create and update tasks through the Mission Control API and its task contract.
- Use project slug `lingjing-solo` for this project.
- Independent verification must be assigned to the registered verifier agent `research`; source agents must not self-approve.
- Production authority is fail-closed: do not change `runtime_authority` or enable `external_ugc` without an independent, hash-bound verifier approval recorded in Mission Control.
- Tasks entering `in-review` must include real deliverables with absolute paths and executable evidence.
- Never move a task from `in-review` to `done`; final closure requires Jj approval.

## Evidence and safety

- Use frozen canonical snapshots for long-lived compatibility claims; live append-only data is not a substitute.
- Preserve provenance, rollback, audit, and verifier report hashes.
- Do not access or commit credentials, tokens, private keys, or generated local secrets.
