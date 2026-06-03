"""Hipocampo — memoria episodica one-shot, sem retreino, para a LLM congelada.
Cadeia: DG (pattern separation) -> CA3 (pattern completion) -> indice esparso.
Escrita gateada por surpresa; leitura por similaridade U contiguidade temporal.
Flags de ablacao (use_dg/use_ca3/use_contiguity/use_gate) para o experimento causal."""
import torch
from .dg import DG
from .ca3 import ca3_complete
from .index import EpisodicStore
from .contiguity import temporal_neighbors
from .surprise_gate import SurpriseGate


def _norm(x):
    return x / (x.norm() + 1e-8)


class Hippocampus:
    def __init__(self, d, expand=4, sparsity=0.05, beta=8.0, seed=0,
                 use_dg=True, use_ca3=True, use_contiguity=True, use_gate=True,
                 gamma=1.0, k_s=8, k_c=4, n_neighbors=1):
        self.d = d
        self.use_dg = use_dg
        self.use_ca3 = use_ca3
        self.use_contiguity = use_contiguity
        self.use_gate = use_gate
        self.beta = beta
        self.k_s, self.k_c, self.n_neighbors = k_s, k_c, n_neighbors
        self.dg = DG(d, expand, sparsity, seed)
        self.store = EpisodicStore()
        self.gate = SurpriseGate(gamma=gamma)

    def _key(self, e):
        return self.dg(e) if self.use_dg else e

    def write(self, e, surprise=0.0):
        e = _norm(e)
        if self.use_gate and not self.gate.should_write(surprise):
            return False
        self.store.add(self._key(e), e, surprise=surprise)
        return True

    def read(self, cue):
        """Retorna lista de conteudos (embeddings) recuperados, em ordem de prioridade."""
        if len(self.store) == 0:
            return []
        cue = _norm(cue)
        key = self._key(cue)
        X = self.store.keys_matrix()
        if self.use_ca3:
            key = ca3_complete(key, X, self.beta)   # completa a chave (pista parcial -> atrator)
        sims = X @ key
        k = min(self.k_s, len(self.store))
        sim_ids = torch.topk(sims, k).indices.tolist()
        ids = list(sim_ids)
        if self.use_contiguity:
            ids = ids + temporal_neighbors(sim_ids, len(self.store), self.n_neighbors, self.k_c)
        seen, out = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return [self.store.content[i] for i in out]

    def recall_top1_id(self, cue):
        """id da memoria de maior prioridade (para medir recall@1)."""
        if len(self.store) == 0:
            return None
        cue = _norm(cue)
        key = self._key(cue)
        X = self.store.keys_matrix()
        if self.use_ca3:
            key = ca3_complete(key, X, self.beta)
        return int(torch.argmax(X @ key).item())
