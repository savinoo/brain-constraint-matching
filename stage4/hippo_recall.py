"""Experimento Tarefa A — recall associativo de UMA tentativa sob pista corrompida e
interferencia. Ablacao causal: RAG puro -> +DG -> +DG+CA3. Mede QUAL peca causa ganho
e em que regime. CPU, >=5 seeds, REPORT.md/JSON. Estilo study.py/stage2/stage3.
"""
import json
import os
import statistics
import torch
from brain.hippocampus import Hippocampus

BETA = 8.0
CONDS = {
    "rag":     dict(use_dg=False, use_ca3=False),   # cosine sobre embedding cru (sem DG/CA3)
    "dg":      dict(use_dg=True,  use_ca3=False),   # + pattern separation
    "dg_ca3":  dict(use_dg=True,  use_ca3=True),    # + pattern completion (cadeia)
}


def make_episodes(d, n_clusters, per_cluster, sigma_intra, seed):
    """Episodios em clusters: itens do mesmo cluster sao parecidos (interferencia).
    sigma_intra pequeno = alta interferencia (itens quase identicos)."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, d, generator=g)
    centers = centers / centers.norm(dim=1, keepdim=True)
    embs = []
    for c in range(n_clusters):
        for _ in range(per_cluster):
            e = centers[c] + sigma_intra * torch.randn(d, generator=g)
            embs.append(e / e.norm())
    return torch.stack(embs)


def recall_at1(embs, cond, corruption, d, seed):
    hip = Hippocampus(d, use_contiguity=False, use_gate=False, beta=BETA, seed=seed, **cond)
    for e in embs:
        hip.write(e)
    g = torch.Generator().manual_seed(seed + 777)
    hits = 0
    for i, e in enumerate(embs):
        mask = (torch.rand(d, generator=g) > corruption).float()   # zera fracao das dims
        cue = e * mask
        if hip.recall_top1_id(cue) == i:
            hits += 1
    return hits / len(embs)


def sweep(seeds, d=64, n_clusters=20, per_cluster=4, sigma_intra=0.5,
          corruptions=(0.0, 0.3, 0.5, 0.7, 0.9)):
    out = {c: {} for c in CONDS}
    for c, cond in CONDS.items():
        for p in corruptions:
            vals = [recall_at1(make_episodes(d, n_clusters, per_cluster, sigma_intra, s),
                               cond, p, d, s) for s in seeds]
            out[c][p] = (statistics.mean(vals), statistics.pstdev(vals))
    return out, list(corruptions)


def sweep_interference(seeds, d=64, n_clusters=20, per_cluster=4, corruption=0.5,
                       sigmas=(0.2, 0.4, 0.8, 1.5)):
    """Fixa a corrupcao, varre a interferencia (sigma_intra): hipotese -> DG ganha
    quando a interferencia e alta (sigma pequeno)."""
    out = {c: {} for c in CONDS}
    for c, cond in CONDS.items():
        for sg in sigmas:
            vals = [recall_at1(make_episodes(d, n_clusters, per_cluster, sg, s),
                               cond, corruption, d, s) for s in seeds]
            out[c][sg] = (statistics.mean(vals), statistics.pstdev(vals))
    return out, list(sigmas)


def main():
    seeds = (0, 1, 2, 3, 4)
    corr, corrs = sweep(seeds)
    intf, sigmas = sweep_interference(seeds)
    os.makedirs("results", exist_ok=True)

    def table(res, xs, xlabel):
        lines = ["| condição | " + " | ".join(f"{xlabel}={x}" for x in xs) + " |",
                 "|---|" + "---|" * len(xs)]
        for c in CONDS:
            row = f"| {c} | " + " | ".join(f"{res[c][x][0]:.2f}±{res[c][x][1]:.2f}" for x in xs) + " |"
            lines.append(row)
        return "\n".join(lines)

    report = f"""# Hipocampo — recall one-shot sob corrupção e interferência (ablação causal)

5 seeds (média±desvio). recall@1 = a memória de maior prioridade é a certa.
Episódios em clusters (itens do mesmo cluster são parecidos = interferência).

## Recall × corrupção da pista (sigma_intra=0.5)
{table(corr, corrs, "corr")}

## Recall × interferência (corrupção=0.5; sigma_intra menor = mais interferência)
{table(intf, sigmas, "sig")}

## Leitura
- **rag** = cosine sobre o embedding cru (sem DG/CA3). **dg** = +pattern separation. **dg_ca3** = +pattern completion.
- Hipóteses: CA3 ganha sob **corrupção alta** (completa a pista parcial); DG ganha sob **interferência alta** (sigma pequeno).
- Falsificável: se rag empata/ganha em todo regime, a cadeia cérebro-fiel não se justifica aqui (achado honesto).
"""
    open("results/hippocampus.md", "w").write(report)
    json.dump({"corruption": {c: corr[c] for c in CONDS},
               "interference": {c: intf[c] for c in CONDS}},
              open("results/hippocampus.json", "w"), indent=2)

    print("=== recall × corrupção ===")
    print(table(corr, corrs, "corr"))
    print("\n=== recall × interferência ===")
    print(table(intf, sigmas, "sig"))
    print("\nresults/hippocampus.md + .json salvos")


if __name__ == "__main__":
    main()
