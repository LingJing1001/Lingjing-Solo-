"""Auditable R4 hotspot candidate schemas and deterministic IDs."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HotspotFeatures:
    component_area: float = 0.0
    boundary_strength: float = 0.0
    temporal_change: float = 0.0
    color_contrast: float = 0.0
    edge_density: float = 0.0
    interaction_likelihood: float = 0.0
    target_relevance: float = 0.0
    novelty: float = 0.0
    dynamic_noise: float = 0.0
    repeated_failure: float = 0.0
    execution_risk: float = 0.0


@dataclass(frozen=True)
class HotspotScore:
    total: float
    information_gain: float
    progress_potential: float
    interaction: float
    risk_penalty: float
    noise_penalty: float
    explanation: tuple[str, ...] = ()


@dataclass(frozen=True)
class HotspotCandidate:
    hotspot_id: str
    x: int
    y: int
    bbox: Optional[tuple[int, int, int, int]]
    source: str
    object_ref: Optional[str]
    features: HotspotFeatures
    score: HotspotScore
    evidence_refs: tuple[str, ...]
    confidence: float
    risk: float
    status: str = "unknown"


@dataclass(frozen=True)
class HotspotDetectionResult:
    candidates: tuple[HotspotCandidate, ...]
    raw_candidate_count: int
    retained_count: int
    detector_version: str
    warnings: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


def stable_hotspot_id(
    frame_id: str,
    source: str,
    x: int,
    y: int,
    bbox: Optional[tuple[int, int, int, int]],
) -> str:
    """Return an observation-local ID; it does not encode level/game constants."""
    payload = f"{frame_id}|{source}|{x // 2},{y // 2}|{bbox}".encode("utf-8")
    return "hs-" + hashlib.sha256(payload).hexdigest()[:16]
