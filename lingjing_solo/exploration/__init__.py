from .action_diff import (
    ActionDelta,
    ActionObservation,
    ActionSummary,
    analyze_observation,
    analyze_recording,
    summarize_actions,
)
from .click_sweep import BubbleClickPlanner
from .explorer import ExplorationEngine

from .hotspot_candidates import (
    HotspotCandidate,
    HotspotDetectionResult,
    HotspotFeatures,
    HotspotScore,
    stable_hotspot_id,
)
from .hotspot_detector import R4HotspotDetector
from .probe_dry_run import (
    DryRunProbe,
    prepare_probe_dry_run,
    write_probe_dry_run_jsonl,
)
from .probe_evidence import (
    ProbeEvidence,
    run_synthetic_probe_loop,
    write_probe_evidence_jsonl,
)
from .probe_feedback import (
    FeedbackOutcome,
    ProbeFeedback,
    TabuContext,
    TabuEntry,
    attribute_feedback,
    calculate_frame_delta,
    update_candidate_after_feedback,
)
from .probe_gate import R4ProbeGate
from .probe_planner import ProbeAction, ProbePlan, ProbePlanner
from .temporal_noise import (
    TemporalWindowSummary,
    analyze_temporal_window,
    suppress_dynamic_noise,
)

__all__ = [
    "ActionDelta",
    "ActionObservation",
    "ActionSummary",
    "BubbleClickPlanner",
    "ExplorationEngine",

    "HotspotCandidate",
    "HotspotDetectionResult",
    "HotspotFeatures",
    "HotspotScore",
    "FeedbackOutcome",
    "DryRunProbe",
    "ProbeAction",
    "ProbeEvidence",
    "ProbeFeedback",
    "ProbePlan",
    "ProbePlanner",
    "R4ProbeGate",
    "R4HotspotDetector",
    "TabuContext",
    "TabuEntry",
    "analyze_observation",
    "analyze_recording",

    "summarize_actions",
    "stable_hotspot_id",
    "attribute_feedback",
    "calculate_frame_delta",
    "prepare_probe_dry_run",
    "run_synthetic_probe_loop",
    "update_candidate_after_feedback",
    "write_probe_dry_run_jsonl",
    "write_probe_evidence_jsonl",
    "TemporalWindowSummary",
    "analyze_temporal_window",
    "suppress_dynamic_noise",
]
