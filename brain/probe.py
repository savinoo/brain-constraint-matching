"""Sonda linear: quanta informação da variável da tarefa (objetivo) a representação
congelada carrega, decodável LINEARMENTE? É o instrumento barato proposto para o
estágio 2 (medir decodabilidade da representação de uma LLM/V-JEPA congelada).

Aqui validamos, no brinquedo, que a R² da sonda PREVÊ o sucesso do congelamento:
representação que carrega o objetivo (R²→1) => congelar funciona; que o destrói
(R²→0) => congelar falha.
"""
import torch


def linear_probe_r2(frozen_cortex, data, test_frac=0.2, seed=0):
    """R² out-of-sample de prever o objetivo a partir do z congelado via regressão linear."""
    with torch.no_grad():
        Z = frozen_cortex(data["scene"])
    G = data["goal"]
    n = Z.shape[0]
    n_test = max(1, int(n * test_frac))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    te, tr = perm[:n_test], perm[n_test:]
    ones = lambda idx: torch.ones(len(idx), 1)
    Xtr = torch.cat([Z[tr], ones(tr)], dim=1)
    Xte = torch.cat([Z[te], ones(te)], dim=1)
    sol = torch.linalg.lstsq(Xtr, G[tr]).solution
    pred = Xte @ sol
    ss_res = ((G[te] - pred) ** 2).sum()
    ss_tot = ((G[te] - G[te].mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)
