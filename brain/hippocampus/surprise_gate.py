"""Gating por surpresa: grava so eventos salientes. Limiar movel S_t > mu + gamma*sigma
sobre uma janela (padrao EM-LLM; Fountas et al. 2025)."""
import numpy as np


class SurpriseGate:
    def __init__(self, gamma=1.0, window=32):
        self.gamma = gamma
        self.window = window
        self.buf = []

    def should_write(self, s):
        s = float(s)
        if len(self.buf) < 3:
            self.buf.append(s)
            return True
        w = self.buf[-self.window:]
        mu, sigma = float(np.mean(w)), float(np.std(w) + 1e-6)
        write = s > mu + self.gamma * sigma
        self.buf.append(s)
        return write
