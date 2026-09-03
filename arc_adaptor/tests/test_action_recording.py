from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


class DummyRecorder:
    def __init__(self):
        self.records = []

    def record(self, payload):
        self.records.append(payload)


class DummyAgent(Agent):
    def is_done(self, frames, latest_frame):
        return False

    def choose_action(self, frames, latest_frame):
        return GameAction.ACTION1


def test_recording_keeps_requested_action_separate_from_response_action():
    agent = DummyAgent(
        card_id="card",
        game_id="ls20-9607627b",
        agent_name="dummy",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=None,
    )
    agent.recorder = DummyRecorder()
    agent._last_requested_action = GameAction.ACTION3
    frame = FrameData(
        game_id="ls20-9607627b",
        frame=[[[1]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=7,
        available_actions=[GameAction.ACTION1, GameAction.ACTION2],
    )

    agent.append_frame(frame)

    assert agent.recorder.records[0]["requested_action"] == {
        "name": "ACTION3",
        "id": 3,
    }
