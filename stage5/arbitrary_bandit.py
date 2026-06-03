"""Bandit contextual com mapeamento ARBITRARIO contexto->acao-certa. Arbitrario de
proposito: o prior semantico da LLM nao pode adivinhar — so o reward (online) revela.
Isola a contribuicao do aprendizado por reforco (a restricao que a LLM congelada nao tem)."""
import numpy as np


class ArbitraryBandit:
    def __init__(self, n_contexts=12, n_actions=4, d=32, seed=0):
        rng = np.random.default_rng(seed)
        self.n_contexts = n_contexts
        self.n_actions = n_actions
        self.d = d
        Z = rng.standard_normal((n_contexts, d)).astype("float32")
        self.Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)        # embedding (cortex) por contexto
        self.correct = rng.integers(0, n_actions, size=n_contexts)   # mapeamento arbitrario
        self.rng = rng

    def sample_context(self):
        c = int(self.rng.integers(self.n_contexts))
        return c, self.Z[c]

    def reward(self, c, a):
        return 1.0 if a == self.correct[c] else 0.0
