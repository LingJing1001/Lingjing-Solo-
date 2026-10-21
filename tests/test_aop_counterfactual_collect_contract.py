import json

from tools.aop.counterfactual_collect import (
    action_record,
    make_action,
    recorded_action_record,
    recording_transition,
)


class Frame:
    def model_dump(self, mode="json"):
        return {"game_id": "ar25-example"}


def test_action6_record_preserves_serialized_sdk_payload():
    action = make_action(6, Frame(), (15, 31))
    assert action_record(action) == {
        "name": "ACTION6",
        "id": 6,
        "payload": {"game_id": "ar25-example", "x": 15, "y": 31},
    }


def test_simple_action_record_has_explicit_sdk_payload():
    action = make_action(1, Frame())
    assert action_record(action) == {"name": "ACTION1", "id": 1, "payload": {"game_id": ""}}


def test_recorded_action_record_uses_authoritative_server_payload():
    record = {"action_input": {"id": "ACTION1", "data": {"game_id": "ar25-server"}}}
    assert recorded_action_record(record, 1) == {
        "name": "ACTION1", "id": 1, "payload": {"game_id": "ar25-server"}
    }


def test_recording_transition_uses_marked_final_candidate_not_prefix_match(tmp_path):
    card = "card-1"
    recording_dir = tmp_path / "remote_recordings" / card
    recording_dir.mkdir(parents=True)
    records = [
        {"data": {"marker": "reset"}},
        {"data": {"marker": "prefix-before", "action_input": {
            "id": "ACTION1", "reasoning": {"probe": "counterfactual replay", "group": "g0"}
        }}},
        {"data": {"marker": "final-before", "action_input": {
            "id": "ACTION2", "reasoning": {"probe": "counterfactual replay", "group": "g0"}
        }}},
        {"data": {"marker": "final-after", "action_input": {
            "id": "ACTION1", "data": {"game_id": "ar25-example"},
            "reasoning": {"probe": "counterfactual candidate", "group": "g0", "candidate": 1}
        }}},
    ]
    (recording_dir / "recording.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    transition = recording_transition(tmp_path, card, 1, "g0")
    assert transition is not None
    before, after = transition
    assert before["marker"] == "final-before"
    assert after["marker"] == "final-after"
