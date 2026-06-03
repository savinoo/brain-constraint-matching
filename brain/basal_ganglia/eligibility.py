"""Traços de elegibilidade — TD(lambda) tabular. Mecanismo cérebro-fiel de atribuição de
crédito temporal: cada par (estado, ação) visitado deixa uma marca decadente, e uma
recompensa distante atualiza tudo o que a precedeu de uma vez. lambda=0 = TD de um passo
(baseline competente); lambda>0 = traços. Base: Sutton & Barto; elegibilidade sináptica +
plasticidade gateada por dopamina (gânglios da base)."""
import numpy as np
from collections import defaultdict


class TDLambdaQ:
    def __init__(self, n_actions, alpha=0.3, gamma=0.97, lam=0.9, seed=0):
        self.nA = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.rng = np.random.default_rng(seed)
        self.e = defaultdict(lambda: np.zeros(n_actions))

    def act(self, s, n_actions, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(n_actions))
        return int(np.argmax(self.Q[s][:n_actions]))

    def episode_reset(self):
        self.e = defaultdict(lambda: np.zeros(self.nA))

    def step(self, s, a, r, s2, a2, done):
        q_next = 0.0 if done else self.Q[s2][a2]
        delta = r + self.gamma * q_next - self.Q[s][a]
        self.e[s][a] += 1.0
        for st, ev in list(self.e.items()):
            self.Q[st] += self.alpha * delta * ev
            self.e[st] = ev * (self.gamma * self.lam)
