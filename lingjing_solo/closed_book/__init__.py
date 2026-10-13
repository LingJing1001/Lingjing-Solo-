"""Closed-book capability modules — frame-only, no baked answers, no engine access.

Kept in a dedicated package so importing a closed-book solver never drags in
the heavy ``lingjing_solo.planning`` / ``lingjing_solo.perception`` initialisers
(which pull in baked solution tables and the source-parsing AR25 encoder).

Contract for every module in here:

    * input is the ``FrameData`` the agent already receives;
    * no reads of ``environment_files/**``;
    * no hardcoded per-level action strings.

The measured closed-book score is the repo's headline metric
(see ``state/score_registry.json``).
"""
