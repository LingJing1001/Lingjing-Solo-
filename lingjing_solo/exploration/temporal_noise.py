"""Short-window temporal evidence and dynamic-noise suppression for R4."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ..perception.observation import NormalizedObservation
from .hotspot_candidates import HotspotCandidate, HotspotScore


@dataclass(frozen=True)
class TemporalWindowSummary:
    frame_ids: tuple[str, ...]
    changed_ratios: tuple[float, ...]
    mean_change: float
    periodicity_score: float
    dynamic_noise: bool
    explanation: tuple[str, ...] = ()


def analyze_temporal_window(
    observations: Sequence[NormalizedObservation],
    *,
    change_threshold: float = 0.01,
    periodicity_threshold: float = 0.5,
) -> TemporalWindowSummary:
    """Classify repeated visual changes without assigning game semantics."""
    if not 0 <= change_threshold <= 1:
        raise ValueError("change_threshold must be in [0, 1]")
    if not 0 <= periodicity_threshold <= 1:
        raise ValueError("periodicity_threshold must be in [0, 1]")
    if not observations:
        return TemporalWindowSummary((), (), 0.0, 0.0, False, ("empty_window",))
    frames = [np.asarray(item.frame) for item in observations]
    if any(frame.ndim != 2 or frame.size == 0 for frame in frames):
        return TemporalWindowSummary(
            tuple(item.frame_id for item in observations), (), 0.0, 0.0, False,
            ("invalid_frame_in_window",),
        )
    if any(frame.shape != frames[0].shape for frame in frames[1:]):
        return TemporalWindowSummary(
            tuple(item.frame_id for item in observations), (), 0.0, 0.0, False,
            ("mismatched_frame_dimensions",),
        )
    ratios = tuple(float(np.mean(left != right)) for left, right in zip(frames, frames[1:]))
    if len(frames) < 3:
        return TemporalWindowSummary(
            tuple(item.frame_id for item in observations), ratios,
            float(np.mean(ratios)) if ratios else 0.0, 0.0, False,
            ("window_too_short_for_periodicity",),
        )
    periodic_matches = [
        float(np.mean(frames[index] != frames[index - 1])) > change_threshold
        and float(np.mean(frames[index] != frames[index - 2])) <= change_threshold
        for index in range(2, len(frames))
    ]
    periodicity = float(np.mean(periodic_matches)) if periodic_matches else 0.0
    mean_change = float(np.mean(ratios)) if ratios else 0.0
    noisy = periodicity >= periodicity_threshold and mean_change > change_threshold
    explanation = ("periodic_frame_change",) if noisy else ("no_periodic_noise_detected",)
    return TemporalWindowSummary(
        tuple(item.frame_id for item in observations), ratios, mean_change,
        periodicity, noisy, explanation,
    )


def suppress_dynamic_noise(
    candidate: HotspotCandidate,
    summary: TemporalWindowSummary,
) -> HotspotCandidate:
    """Penalize temporal candidates only when the window supports noise."""
    if not summary.dynamic_noise or candidate.source != "delta_region":
        return candidate
    penalty = max(candidate.score.noise_penalty, min(1.0, summary.periodicity_score))
    features = replace(candidate.features, dynamic_noise=penalty)
    score = candidate.score
    score = HotspotScore(
        total=round(max(0.0, score.total - 0.30 * penalty), 6),
        information_gain=score.information_gain,
        progress_potential=score.progress_potential,
        interaction=score.interaction,
        risk_penalty=score.risk_penalty,
        noise_penalty=penalty,
        explanation=score.explanation + ("periodic_dynamic_noise",),
    )
    return replace(candidate, features=features, score=score,
                   confidence=round(max(0.0, candidate.confidence - 0.25 * penalty), 6))
