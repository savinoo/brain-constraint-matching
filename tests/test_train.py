import torch
from brain.data import collect_oracle_dataset
from brain.train import pretrain_cortex, train_cotrain, train_freeze_adapt, train_monolith


def _data():
    return collect_oracle_dataset(n_episodes=20, seed=0)


def test_pretrain_cortex_lowers_goal_error():
    data = _data()
    cortex, losses = pretrain_cortex(data, epochs=50, seed=0)
    assert losses[-1] < losses[0]
    for p in cortex.parameters():
        assert p.requires_grad is False


def test_cotrain_lowers_bc_loss():
    data = _data()
    (cortex, controller), losses = train_cotrain(data, epochs=50, seed=0)
    assert losses[-1] < losses[0]


def test_freeze_adapt_lowers_bc_loss_and_keeps_cortex_frozen():
    data = _data()
    cortex, _ = pretrain_cortex(data, epochs=50, seed=0)
    before = [p.detach().clone() for p in cortex.parameters()]
    (cortex2, adapter, controller), losses = train_freeze_adapt(data, cortex, epochs=50, seed=0)
    assert losses[-1] < losses[0]
    for p0, p1 in zip(before, cortex2.parameters()):
        assert torch.allclose(p0, p1)


def test_monolith_lowers_bc_loss():
    data = _data()
    monolith, losses = train_monolith(data, epochs=50, seed=0)
    assert losses[-1] < losses[0]
