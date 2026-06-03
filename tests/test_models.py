import torch
from brain.models import (Cortex, CortexPretrainHead, Adapter, Controller, Monolith,
                          SCENE_DIM, POS_DIM, GOAL_DIM, Z_DIM, ACTION_DIM)


def test_forward_shapes():
    B = 7
    scene = torch.randn(B, SCENE_DIM)
    pos = torch.randn(B, POS_DIM)
    cortex = Cortex()
    z = cortex(scene)
    assert z.shape == (B, Z_DIM)
    assert CortexPretrainHead()(z).shape == (B, GOAL_DIM)
    assert Adapter()(z).shape == (B, Z_DIM)
    assert Controller()(pos, z).shape == (B, ACTION_DIM)
    assert Monolith()(scene, pos).shape == (B, ACTION_DIM)
