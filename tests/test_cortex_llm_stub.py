import numpy as np
from brain.cortex_llm import LLMCortex
from brain.models import H_DIM


def test_stub_backend_returns_hidden_and_latency():
    cx = LLMCortex(backend="stub", stub_latency_ms=5.0, seed=0)
    scene = np.zeros(16, dtype=np.float32)
    h = cx.hidden(scene)
    assert h.shape == (H_DIM,)
    assert cx.lat_ms >= 5.0
    assert np.allclose(h, cx.hidden(scene))


def test_scene_to_prompt_is_short_text():
    cx = LLMCortex(backend="stub")
    p = cx.scene_to_prompt(np.arange(16, dtype=np.float32) / 16.0)
    assert isinstance(p, str) and "scene" in p
