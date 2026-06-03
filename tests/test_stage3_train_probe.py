import torch
from brain.data import collect_oracle_dataset
from brain.train import train_freeze_adapt_llm
from brain.models import H_DIM, AdapterLLM, Controller


def test_train_freeze_adapt_llm_runs_with_hidden():
    data = collect_oracle_dataset(n_episodes=20, seed=0)
    H = torch.randn(data["scene"].shape[0], H_DIM)
    (adapter, controller), losses = train_freeze_adapt_llm(H, data, epochs=40, seed=0)
    assert losses[-1] < losses[0]
    assert isinstance(adapter, AdapterLLM) and isinstance(controller, Controller)
