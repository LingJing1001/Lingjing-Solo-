import torch

from tools.aop.counterfactual_qa import validate_groups
from tools.aop.train import counterfactual_rank_loss


def row(group, action, progressed=False, effective=True):
    return {
        'counterfactual_group_id': group,
        'requested_action': {'name': f'ACTION{action}'},
        'legal_actions': [1, 2],
        'replay_verified': True,
        'replay_state_hash': 'same',
        'state_delta': {'progressed': progressed, 'action_effective': effective},
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


def test_counterfactual_rank_loss_uses_progress_outcome():
    labels = [row('g', 1, progressed=False), row('g', 2, progressed=True)]
    logits = torch.tensor([[2.0, 0.0], [0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = counterfactual_rank_loss(logits, targets, labels)
    assert float(loss.detach()) > 0.0
