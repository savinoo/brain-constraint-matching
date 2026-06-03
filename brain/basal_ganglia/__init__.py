"""Gânglios da base — seleção de ação por aprendizado por reforço (RL dopaminérgico).
Aprende Q(z, a) sobre o embedding do córtex (congelado) a partir do REWARD, online —
a capacidade que uma LLM congelada NÃO tem (não aprende do reward). Seleção por argmax
dos valores aprendidos (análogo Go/NoGo: realça a ação de maior valor, suprime o resto)."""
import numpy as np
import torch
import torch.nn as nn


class BasalGanglia(nn.Module):
    def __init__(self, d, n_actions, lr=0.1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_actions = n_actions
        self.q = nn.Linear(d, n_actions)
        self.opt = torch.optim.SGD(self.q.parameters(), lr=lr)

    def values(self, z):
        with torch.no_grad():
            return self.q(torch.as_tensor(z, dtype=torch.float32))

    def act(self, z, epsilon=0.1, rng=None):
        if rng is not None and rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        return int(torch.argmax(self.values(z)).item())

    def update(self, z, a, r):
        """TD de bandit: Q(z,a) -> E[r | z,a]. r-Q e o erro de predicao de recompensa (dopamina)."""
        z = torch.as_tensor(z, dtype=torch.float32)
        q_sa = self.q(z)[a]
        loss = (q_sa - float(r)) ** 2
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())
