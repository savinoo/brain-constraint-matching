"""Contiguidade temporal: vizinhos no tempo sobem junto com os recuperados por similaridade
(efeito de contiguidade humano; Howard & Kahana 2002; copiado do EM-LLM)."""


def temporal_neighbors(ids, n_total, n_neighbors=1, k_c=4):
    out = []
    for i in ids:
        for dd in range(-n_neighbors, n_neighbors + 1):
            j = i + dd
            if dd != 0 and 0 <= j < n_total:
                out.append(j)
    seen, res = set(), []
    for j in out:
        if j not in seen:
            seen.add(j)
            res.append(j)
    return res[:k_c]
