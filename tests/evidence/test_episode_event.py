import json
import math

import pytest

from lingjing_solo.evidence.schema import EpisodeEvent, EventValidationError, validate_event_rows


def event(**overrides):
    value = {
        "schema_version": "episode_event.v1",
        "episode_id": "ep-1",
        "environment_id": "arc_ls20",
        "environment_version": "local-fixed-v1",
        "seed": 0,
        "tick": 1,
        "observation_ref": "sha256:obs",
        "action": {"name": "ACTION1", "id": 1, "requested": True},
        "transition": {"before_tick": 0, "after_tick": 1, "settled": True},
        "geometric_event": {"features": {"progressed": 0.0}},
        "gate_results": {"dpl": {"passed": True}},
        "hypothesis": None,
        "replay_hash": "sha256:replay",
        "provenance": {"source": "recording", "source_commit": "abc123", "input_hash": "sha256:input"},
        "authority_status": "shadow",
    }
    value.update(overrides)
    return value


def test_valid_event_round_trips_and_is_finite():
    parsed = EpisodeEvent.from_dict(event())
    assert parsed.to_dict()["schema_version"] == "episode_event.v1"
    assert parsed.authority_status == "shadow"


@pytest.mark.parametrize("bad", [
    {"episode_id": ""},
    {"provenance": None},
    {"authority_status": "active"},
    {"tick": -1},
    {"geometric_event": {"features": {"bad": math.inf}}},
])
def test_event_fail_closed(bad):
    with pytest.raises(EventValidationError):
        EpisodeEvent.from_dict(event(**bad))


def test_tick_monotonic_and_reset_segment():
    rows = [event(tick=0), event(tick=1), event(tick=0, transition={"before_tick": 0, "after_tick": 0, "settled": True}, action={"name": "RESET", "id": "RESET", "requested": True})]
    assert validate_event_rows(rows) == 3
    with pytest.raises(EventValidationError, match="monotonic"):
        validate_event_rows([event(tick=2), event(tick=1)])


def test_unknown_schema_and_non_requested_action_rejected():
    with pytest.raises(EventValidationError):
        EpisodeEvent.from_dict(event(schema_version="episode_event.v2"))
    with pytest.raises(EventValidationError):
        EpisodeEvent.from_dict(event(action={"name": "ACTION1", "id": 1, "requested": False}))
