"""CA3 — pattern completion via Hopfield moderno, que e LITERALMENTE a atencao do
Transformer (Ramsauer et al. 2020, arXiv:2008.02217): k_hat = X . softmax(beta . X^T . cue).
UM update. beta alto -> recupera 1 padrao isolado (regime de ponto-fixo desejado)."""
import torch


def ca3_complete(cue, X, beta=8.0):
    """cue: (D,) pista parcial/ruidosa; X: (N, D) chaves armazenadas. Retorna (D,) completada."""
    if X.shape[0] == 0:
        return cue
    p = torch.softmax(beta * (X @ cue), dim=0)   # (N,) — pesos de atencao sobre as memorias
    return X.T @ p                                # (D,) — chave completada (atrator)
