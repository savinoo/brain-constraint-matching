"""Indice esparso (Indexing Theory): guarda (chave_DG, ponteiro_para_conteudo, meta).
Busca exata por produto interno (recomendado na escala do de-risco; ANN so se N >> 1e4)."""
import torch


class EpisodicStore:
    def __init__(self):
        self.keys = []       # chaves DG esparsas (D,)
        self.content = []    # conteudo episodico (embedding e) (d,) — o "ponteiro" resolvido
        self.meta = []       # {t, surprise}

    def add(self, key, content, t=None, surprise=0.0):
        self.keys.append(key)
        self.content.append(content)
        self.meta.append({"t": len(self.meta) if t is None else t, "surprise": float(surprise)})

    def keys_matrix(self):
        return torch.stack(self.keys) if self.keys else torch.empty(0)

    def search_exact(self, query_key, k):
        """top-k por produto interno sobre as chaves."""
        X = self.keys_matrix()
        sims = X @ query_key
        k = min(k, len(self.keys))
        return torch.topk(sims, k).indices.tolist()

    def __len__(self):
        return len(self.keys)
