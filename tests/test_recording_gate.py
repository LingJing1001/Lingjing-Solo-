import pytest

from models.aop.recording_gate import RecordingValidationError, validate_rows


def valid(action="CLICK"):
    return {
        "episode_id": "ep-1",
        "requested_action": {"name": action, "id": 6},
        "observation": {
            "tick": 0, "state": "NOT_FINISHED",
            "legal_actions": ["CLICK"], "levels_completed": 0,
        },
        "data": {
            "tick": 1, "state": "NOT_FINISHED",
            "legal_actions": ["CLICK"], "levels_completed": 0,
        },
    }


def test_accepts_settled_action_and_reports_counts():
    report = validate_rows([valid()])
    assert report.rows == 1
    assert report.episodes == 1
    assert report.actions == 1
    assert report.resets == 0


def test_reset_is_allowed_but_not_an_action_label():
    reset = valid("RESET")
    report = validate_rows([reset, valid()])
    assert report.resets == 1
    assert report.actions == 1


@pytest.mark.parametrize("mutation", [
    lambda r: r["data"].update(tick=0),
    lambda r: r["observation"].pop("legal_actions"),
    lambda r: r["requested_action"].update(name="UNKNOWN"),
    lambda r: (r["observation"].pop("state"), r["observation"].pop("levels_completed")),
])
def test_rejects_untrustworthy_recording(mutation):
    row = valid()
    mutation(row)
    with pytest.raises(RecordingValidationError):
        validate_rows([row])


def test_rejects_empty_and_reset_only_recordings():
    with pytest.raises(RecordingValidationError):
        validate_rows([])
    with pytest.raises(RecordingValidationError):
        validate_rows([valid("RESET")])
