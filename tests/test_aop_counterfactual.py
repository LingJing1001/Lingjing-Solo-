import torch

from tools.aop.counterfactual_collect import make_action, recording_action_id, step_action
from tools.aop.counterfactual_qa import validate_groups
from tools.aop.train import counterfactual_rank_loss


def row(group, action, progressed=False, effective=True, visual_delta=None):
    return {
        'counterfactual_group_id': group,
        'requested_action': {'name': f'ACTION{action}'},
        'legal_actions': [1, 2],
        'replay_verified': True,
        'replay_state_hash': 'same',
        'state_delta': {'progressed': progressed, 'action_effective': effective, 'visual_delta': visual_delta},
    }


def test_counterfactual_group_validation_requires_same_state_and_legal_action():
    report = validate_groups([row('g', 1), row('g', 2)])
    assert report['pass'] is True
    assert report['comparable_groups'] == 1
    assert report['informative_groups'] == 0


def test_counterfactual_rank_loss_skips_uninformative_groups():
    labels = [row('g', 1), row('g', 2)]
    logits = torch.zeros(2, 2, requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = counterfactual_rank_loss(logits, targets, labels)
    assert float(loss.detach()) == 0.0


def test_counterfactual_rank_loss_uses_visual_delta_when_progress_is_equal():
    labels = [row('g', 1, visual_delta=2), row('g', 2, visual_delta=12)]
    logits = torch.zeros(2, 2, requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = counterfactual_rank_loss(logits, targets, labels)
    assert float(loss.detach()) > 0.0


def test_counterfactual_rank_loss_uses_progress_outcome():
    labels = [row('g', 1, progressed=False), row('g', 2, progressed=True)]
    logits = torch.zeros(2, 2, requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = counterfactual_rank_loss(logits, targets, labels)
    assert float(loss.detach()) > 0.0


def test_recording_action_id_normalizes_numeric_and_named_payloads():
    assert recording_action_id({'id': 6, 'name': 'ACTION6'}) == 6
    assert recording_action_id({'id': 'ACTION7'}) == 7
    assert recording_action_id(5) == 5


def test_action6_requires_payload_and_serializes_complex_action():
    import pytest

    with pytest.raises(ValueError, match='requires an explicit x,y payload'):
        make_action(6)
    action = make_action(6, action6_xy=(15, 31))
    assert action.action_data.x == 15
    assert action.action_data.y == 31


def test_step_action_forwards_complex_payload_to_sdk_boundary():
    class FakeEnv:
        def step(self, action, data=None, reasoning=None):
            self.received = (action, data, reasoning)
            return 'ok'

    env = FakeEnv()
    action = make_action(6, action6_xy=(15, 31))
    assert step_action(env, action, {'probe': 'test'}) == 'ok'
    assert env.received[1] is not None
    assert env.received[1]['x'] == 15
    assert env.received[1]['y'] == 31
