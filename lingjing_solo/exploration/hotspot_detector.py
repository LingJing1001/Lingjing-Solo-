"""R4 offline hotspot detector.

This module generates bounded, evidence-backed candidates. It intentionally does
not infer game rules, success, level IDs, or execute actions.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from ..perception.observation import NormalizedObservation
from .hotspot_candidates import (
    HotspotCandidate,
    HotspotDetectionResult,
    HotspotFeatures,
    HotspotScore,
    stable_hotspot_id,
)
from .temporal_noise import analyze_temporal_window, suppress_dynamic_noise

DETECTOR_VERSION = "r4-offline-0.1"


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components as ``(y, x)`` pixels."""
    if mask.ndim != 2:
        raise ValueError("component mask must be 2-D")
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    result: list[list[tuple[int, int]]] = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        component: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        result.append(component)
    return result


def _bbox(component: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pixels = tuple(component)
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    return min(xs), min(ys), max(xs), max(ys)


def _centroid(component: Iterable[tuple[int, int]]) -> tuple[int, int]:
    pixels = tuple(component)
    return int(round(sum(p[1] for p in pixels) / len(pixels))), int(
        round(sum(p[0] for p in pixels) / len(pixels))
    )


def _safe_float(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class R4HotspotDetector:
    """Deterministic geometry/delta detector with an explicit candidate budget."""

    def __init__(
        self,
        *,
        min_component_area: int = 4,
        max_component_area_ratio: float = 0.80,
        temporal_delta_threshold: float = 0.08,
        deduplicate_radius: int = 3,
    ) -> None:
        if min_component_area < 1:
            raise ValueError("min_component_area must be positive")
        if not 0 < max_component_area_ratio <= 1:
            raise ValueError("max_component_area_ratio must be in (0, 1]")
        if not 0 <= temporal_delta_threshold <= 1:
            raise ValueError("temporal_delta_threshold must be in [0, 1]")
        if deduplicate_radius < 0:
            raise ValueError("deduplicate_radius must be non-negative")
        self.min_component_area = min_component_area
        self.max_component_area_ratio = max_component_area_ratio
        self.temporal_delta_threshold = temporal_delta_threshold
        self.deduplicate_radius = deduplicate_radius

    def detect(
        self, observation: NormalizedObservation, *, max_candidates: int = 20
    ) -> HotspotDetectionResult:
        if not isinstance(max_candidates, int) or isinstance(max_candidates, bool):
            raise ValueError("max_candidates must be an integer")
        if not 0 <= max_candidates <= 20:
            raise ValueError("max_candidates must be in [0, 20]")
        if observation.game_family == "keyboard":
            return HotspotDetectionResult(
                candidates=(), raw_candidate_count=0, retained_count=0,
                detector_version=DETECTOR_VERSION,
                warnings=("keyboard game family skips click hotspots",),
                evidence_refs=observation.evidence_refs + (observation.frame_id,),
            )
        frame = np.asarray(observation.frame)
        if frame.ndim != 2 or frame.size == 0:
            return HotspotDetectionResult(
                candidates=(), raw_candidate_count=0, retained_count=0,
                detector_version=DETECTOR_VERSION,
                warnings=("empty or invalid frame",),
            )
        candidates = self._component_candidates(observation)
        candidates.extend(self._delta_candidates(observation))
        candidates = self._deduplicate(candidates)
        ranked = sorted(
            candidates,
            key=lambda c: (-c.score.total, c.hotspot_id),
        )[:max_candidates]
        return HotspotDetectionResult(
            candidates=tuple(ranked),
            raw_candidate_count=len(candidates),
            retained_count=len(ranked),
            detector_version=DETECTOR_VERSION,
            evidence_refs=observation.evidence_refs + (observation.frame_id,),
        )

    def detect_window(
        self,
        observations: Iterable[NormalizedObservation],
        *,
        max_candidates: int = 20,
    ) -> HotspotDetectionResult:
        """Detect from the newest frame and penalize periodic delta noise."""
        window = tuple(observations)
        if not window:
            return HotspotDetectionResult(
                candidates=(), raw_candidate_count=0, retained_count=0,
                detector_version=DETECTOR_VERSION,
                warnings=("empty observation window",),
            )
        summary = analyze_temporal_window(window)
        result = self.detect(window[-1], max_candidates=max_candidates)
        candidates = tuple(
            suppress_dynamic_noise(candidate, summary) for candidate in result.candidates
        )
        ranked = tuple(sorted(candidates, key=lambda c: (-c.score.total, c.hotspot_id)))
        return HotspotDetectionResult(
            candidates=ranked,
            raw_candidate_count=result.raw_candidate_count,
            retained_count=len(ranked),
            detector_version=DETECTOR_VERSION,
            warnings=result.warnings + summary.explanation,
            evidence_refs=result.evidence_refs + summary.frame_ids,
        )

    def _component_candidates(self, observation: NormalizedObservation) -> list[HotspotCandidate]:
        frame = np.asarray(observation.frame)
        values, counts = np.unique(frame, return_counts=True)
        background = values[int(np.argmax(counts))]
        max_area = frame.size * self.max_component_area_ratio
        output: list[HotspotCandidate] = []
        for value in values:
            if value == background:
                continue
            for component in _components(frame == value):
                area = len(component)
                if area < self.min_component_area or area > max_area:
                    continue
                x, y = _centroid(component)
                bbox = _bbox(component)
                contrast = _safe_float(abs(float(value) - float(background)) / 255.0)
                interaction = _safe_float(0.55 + 0.35 * min(1.0, area / max(1, frame.size * 0.05)))
                features = HotspotFeatures(
                    component_area=float(area),
                    color_contrast=contrast,
                    interaction_likelihood=interaction,
                    novelty=1.0,
                    target_relevance=0.0,
                    execution_risk=0.0,
                )
                score = HotspotScore(
                    total=round(0.45 * interaction + 0.25 * contrast + 0.20, 6),
                    information_gain=0.20,
                    progress_potential=0.0,
                    interaction=interaction,
                    risk_penalty=0.0,
                    noise_penalty=0.0,
                    explanation=("component_center", "non_background_color"),
                )
                output.append(
                    HotspotCandidate(
                        hotspot_id=stable_hotspot_id(
                            observation.frame_id, "component_center", x, y, bbox
                        ),
                        x=int(np.clip(x, 0, frame.shape[1] - 1)),
                        y=int(np.clip(y, 0, frame.shape[0] - 1)),
                        bbox=bbox,
                        source="component_center",
                        object_ref=f"color:{value}",
                        features=features,
                        score=score,
                        evidence_refs=observation.evidence_refs + (observation.frame_id,),
                        confidence=round(interaction * (0.5 + 0.5 * contrast), 6),
                        risk=0.0,
                    )
                )
        return output

    def _delta_candidates(self, observation: NormalizedObservation) -> list[HotspotCandidate]:
        previous = observation.previous_frame
        if previous is None or previous.shape != observation.frame.shape:
            return []
        delta = np.asarray(previous) != np.asarray(observation.frame)
        if not np.any(delta):
            return []
        output: list[HotspotCandidate] = []
        for component in _components(delta):
            if len(component) < self.min_component_area:
                continue
            x, y = _centroid(component)
            bbox = _bbox(component)
            temporal = _safe_float(len(component) / delta.size)
            features = HotspotFeatures(
                component_area=float(len(component)),
                temporal_change=temporal,
                interaction_likelihood=0.35,
                novelty=0.7,
                dynamic_noise=0.15,
            )
            score = HotspotScore(
                total=round(0.30 + 0.25 * temporal, 6),
                information_gain=0.25,
                progress_potential=0.0,
                interaction=0.35,
                risk_penalty=0.0,
                noise_penalty=0.15,
                explanation=("delta_region", "temporal_change"),
            )
            output.append(
                HotspotCandidate(
                    hotspot_id=stable_hotspot_id(observation.frame_id, "delta_region", x, y, bbox),
                    x=int(np.clip(x, 0, observation.width - 1)),
                    y=int(np.clip(y, 0, observation.height - 1)),
                    bbox=bbox,
                    source="delta_region",
                    object_ref=None,
                    features=features,
                    score=score,
                    evidence_refs=observation.evidence_refs + (observation.frame_id,),
                    confidence=0.35,
                    risk=0.15,
                )
            )
        return output

    def _deduplicate(self, candidates: list[HotspotCandidate]) -> list[HotspotCandidate]:
        kept: list[HotspotCandidate] = []
        for candidate in sorted(candidates, key=lambda c: (-c.score.total, c.hotspot_id)):
            duplicate = any(
                candidate.source == other.source
                and abs(candidate.x - other.x) <= self.deduplicate_radius
                and abs(candidate.y - other.y) <= self.deduplicate_radius
                for other in kept
            )
            if not duplicate:
                kept.append(candidate)
        return kept
