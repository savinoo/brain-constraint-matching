import torch
from brain.models import AdapterLLM, H_DIM, Z_DIM


def test_adapter_llm_shape_and_l2norm():
    a = AdapterLLM()
    h = torch.randn(5, H_DIM)
    z = a(h)
    assert z.shape == (5, Z_DIM)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-4)
