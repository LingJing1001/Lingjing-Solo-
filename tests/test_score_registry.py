"""Metric-alignment gate — the headline score must be the closed-book score.

The repo used to advertise whichever number was biggest.  After the M1
remediation the honest, comparable number is the *closed-book* score
(``LINGJING_CLOSED_BOOK=2``), and ``state/score_registry.json`` is the single
committed source of truth for it.

This gate keeps the registry honest:

* it must exist and declare ``headline_metric == "closed_book_strict"``;
* the three run modes (canned / tables / strict) must be recorded with their
  provenance (source bench JSON path);
* the recorded scores must still match the committed bench artifacts, so a
  fresh 25-game run cannot silently change the headline without an explicit
  registry update;
* the decomposition identities must hold:
  ``canned - tables = hardcoded``, ``tables - strict = r3``, ``strict = closed``.

The heavy lifting lives in ``scripts/closed_book_score.py``; this file only
asserts invariants so the two can never diverge.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "closed_book_score.py"
REGISTRY = REPO / "state" / "score_registry.json"

HEADLINE = "closed_book_strict"
MODES = ("canned", "tables", "strict")


def _load_scorer():
    spec = importlib.util.spec_from_file_location("closed_book_score", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scorer():
    if not SCRIPT.exists():
        pytest.fail(f"scorer script missing: {SCRIPT}")
    return _load_scorer()


@pytest.fixture(scope="module")
def registry():
    if not REGISTRY.exists():
        pytest.fail(
            f"score registry missing: {REGISTRY}\n"
            "Generate it with: python scripts/closed_book_score.py --write state/score_registry.json"
        )
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_registry_shape_and_headline(registry):
    assert registry.get("schema") == "lingjing.score_registry/v1"
    assert registry.get("headline_metric") == HEADLINE, (
        "the headline metric must be the closed-book score, not the canned one"
    )
    for mode in MODES:
        assert mode in registry.get("modes", {}), f"missing mode: {mode}"
        entry = registry["modes"][mode]
        for field in ("score", "levels", "wins", "source"):
            assert field in entry, f"modes.{mode} missing {field}"
        assert Path(REPO / entry["source"]).exists(), (
            f"modes.{mode}.source does not exist: {entry['source']}"
        )


def test_registry_scores_match_bench_artifacts(scorer, registry):
    """A fresh run must not change the headline without a registry update."""
    fresh = scorer.build_registry(scorer.DEFAULT_SOURCES)
    for mode in MODES:
        recorded = registry["modes"][mode]
        actual = fresh["modes"][mode]
        assert recorded["score"] == actual["score"], (
            f"modes.{mode}.score drifted: registry={recorded['score']} bench={actual['score']}\n"
            "Re-run the bench and regenerate the registry if this is intentional."
        )
        assert recorded["source"] == actual["source"]
    assert registry["headline_metric"] == fresh["headline_metric"]


def test_decomposition_identities(registry):
    m = registry["modes"]
    d = registry["decomposition"]

    canned = m["canned"]["score_exact"]
    tables = m["tables"]["score_exact"]
    strict = m["strict"]["score_exact"]

    assert tables <= canned + 1e-9, "tables mode scored above canned mode"
    assert strict <= tables + 1e-9, "strict mode scored above tables mode"

    assert d["hardcoded_tables"]["score"] == pytest.approx(canned - tables, abs=5e-5)
    assert d["r3_shadow"]["score"] == pytest.approx(tables - strict, abs=5e-5)
    assert d["closed_book"]["score"] == pytest.approx(strict, abs=5e-5)

    # The whole point of the exercise: the canned headline is mostly baked answer.
    assert d["hardcoded_tables"]["pct_of_canned"] > 90.0, (
        "hardcoded tables unexpectedly account for <90% of the canned score; "
        "re-verify the closed-book switch did not regress"
    )


def test_closed_book_is_published_delta(scorer, registry):
    """The closed-book score must be the one comparable to official Kaggle."""
    assert registry["official_kaggle"]["public_score"] > 0
    ratio_closed = registry["official_kaggle"]["ratio_closed_book"]
    ratio_canned = registry["official_kaggle"]["ratio_canned"]
    assert ratio_closed is not None and ratio_canned is not None
    assert ratio_closed < ratio_canned, (
        "closed-book score should be a much closer predictor of the official score "
        "than the canned score"
    )
