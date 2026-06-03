"""DG — pattern separation: projecao aleatoria FIXA (nao-treinavel) de expansao + k-WTA.
Ortogonaliza episodios parecidos antes de armazenar. Fiel ao espirito (DG real usa
inibicao aprendida; aqui e projecao fixa, marcada como simplificacao)."""
import torch


def make_projection(d, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D, d, generator=g) / d ** 0.5


def kwta(h, k):
    """Mantem o top-k por |valor| (retendo o valor), zera o resto. Inibicao lateral."""
    k = max(1, int(k))
    k = min(k, h.shape[-1])
    idx = torch.topk(h.abs(), k, dim=-1).indices
    out = torch.zeros_like(h)
    out.scatter_(-1, idx, h.gather(-1, idx))
    return out


class DG:
    def __init__(self, d, expand=4, sparsity=0.05, seed=0):
        self.d = d
        self.D = int(expand * d)
        self.k = max(1, int(sparsity * self.D))
        self.P = make_projection(d, self.D, seed)

    def __call__(self, e):
        # e: (d,) ou (B, d), idealmente L2-normalizado
        h = e @ self.P.T          # expansao d -> D
        return kwta(h, self.k)
