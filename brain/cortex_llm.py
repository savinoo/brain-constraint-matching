import time
import numpy as np
import torch
from brain.models import H_DIM


def quantize_scene(scene):
    """16-D float -> texto curto e estruturado (inteiros = poucos tokens)."""
    q = (np.asarray(scene, dtype=np.float32) * 100).round().astype(int)
    return "scene: " + " ".join(str(int(v)) for v in q)


class LLMCortex:
    """Cortex congelado. backend='stub' (MLP fixo + sleep calibrado, roda no servidor
    sem LLM) | 'mps' (Qwen2.5-0.5B real no Mac). Sempre CONGELADO; mede lat_ms."""

    def __init__(self, backend="stub", model_name="Qwen/Qwen2.5-0.5B",
                 layer=None, pooling="last", stub_latency_ms=200.0, seed=0):
        self.backend = backend
        self.pooling = pooling
        self.lat_ms = 0.0
        self._stub_lat = stub_latency_ms / 1000.0
        if backend == "stub":
            g = torch.Generator().manual_seed(seed)
            self._proj = torch.randn(16, H_DIM, generator=g) / np.sqrt(16)
        elif backend == "mps":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, output_hidden_states=True,
            ).to("mps").eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            n = self.model.config.num_hidden_layers
            self.layer = layer if layer is not None else n // 2
        else:
            raise ValueError(backend)

    def scene_to_prompt(self, scene):
        return quantize_scene(scene)

    @torch.no_grad()
    def hidden(self, scene):
        t0 = time.perf_counter()
        if self.backend == "stub":
            x = torch.tensor(np.asarray(scene, dtype=np.float32))
            h = torch.tanh(x @ self._proj)            # funcao fixa de scene
            if self._stub_lat:
                time.sleep(self._stub_lat)
            out = h.numpy().astype(np.float32)
        else:
            enc = self.tok(self.scene_to_prompt(scene), return_tensors="pt").to("mps")
            hs = self.model(**enc).hidden_states[self.layer][0]   # (T, H)
            if self.pooling == "mean":
                h = hs.mean(0)
            else:
                idx = int(enc["attention_mask"].sum().item()) - 1
                h = hs[idx]
            out = h.float().cpu().numpy().astype(np.float32)
        self.lat_ms = (time.perf_counter() - t0) * 1000.0
        return out
