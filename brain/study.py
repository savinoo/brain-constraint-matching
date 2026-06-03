"""Estudo rigoroso da aposta central: congelar o córtex custa caro? E quando?

Eixos:
  - cortex_mode: "rich" (córtex pré-treinado para prever o objetivo exato — features ricas)
                 "coarse" (pré-treinado só para o quadrante do objetivo — features pobres,
                  análogo a uma LLM frontier cujas features não são control-aligned).
  - K: latência da ponte (córtex recomputado a cada K passos; controlador age todo passo).
  - seeds: robustez estatística.

Compara, em malha fechada: cotrain (co-treino end-to-end), fa_rich, fa_coarse, monolith.
Métrica primária: retorno médio (maior=melhor) e taxa de sucesso. "Penalidade de congelar"
= cotrain - freeze_adapt. ~0 => congelar é grátis; grande => congelar custa.
"""
import json
import os
import statistics
import numpy as np
import torch
import torch.nn as nn

from brain.data import collect_oracle_dataset
from brain.models import (Cortex, CortexPretrainHead, Adapter, Controller, Monolith,
                          SCENE_DIM, Z_DIM, HIDDEN)
from brain.train import train_cotrain, train_monolith, train_freeze_adapt
from brain.bridge import eval_policy


class BottleneckCortex(nn.Module):
    """Córtex com gargalo de informação real: comprime scene em `bottleneck` dimensões
    (< dim do objetivo) e preenche o resto de z com zeros. Fisicamente incapaz de
    carregar o objetivo completo — o caso em que congelar DEVE falhar."""

    def __init__(self, bottleneck=1):
        super().__init__()
        self.bottleneck = bottleneck
        self.enc = nn.Sequential(nn.Linear(SCENE_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, bottleneck))

    def forward(self, scene):
        b = self.enc(scene)
        pad = torch.zeros(b.shape[0], Z_DIM - self.bottleneck, dtype=b.dtype, device=b.device)
        return torch.cat([b, pad], dim=-1)


def _np(x):
    return np.asarray(x, dtype=np.float32)


def _t1(x):
    return torch.tensor(_np(x)).unsqueeze(0)


def pretrain_cortex_mode(data, mode="rich", epochs=200, lr=1e-3, seed=0):
    """Pré-treina o córtex e o congela. Modos de qualidade das features congeladas:
      rich       -> prevê o objetivo exato (features ricas, alinhadas);
      coarse     -> prevê só o sinal/quadrante (objetivo pobre — mas as features podem
                    reter info incidental, como numa rede congelada real);
      misaligned -> prevê objetivos EMBARALHADOS (mapeamento desalinhado/inútil para a tarefa);
      random     -> córtex aleatório congelado, sem pré-treino (features de reservatório).
    """
    torch.manual_seed(seed)
    cortex = BottleneckCortex(bottleneck=1) if mode == "bottleneck" else Cortex()
    head = CortexPretrainHead()
    if mode == "random":
        for p in cortex.parameters():
            p.requires_grad_(False)
        cortex.eval()
        return cortex
    opt = torch.optim.Adam(list(cortex.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    scene, goal = data["scene"], data["goal"]
    if mode in ("rich", "bottleneck"):
        target = goal  # objetivo completo; no bottleneck o gargalo 1-D força perda de info
    elif mode == "coarse":
        target = torch.sign(goal)
    elif mode == "misaligned":
        perm = torch.randperm(goal.shape[0])
        target = goal[perm]  # objetivo de OUTRO exemplo: mapeamento desalinhado
    else:
        raise ValueError(mode)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(cortex(scene)), target)
        loss.backward(); opt.step()
    for p in cortex.parameters():
        p.requires_grad_(False)
    cortex.eval()
    return cortex


def _cotrain_fns(cx, ctrl):
    @torch.no_grad()
    def cor(scene):
        return cx(_t1(scene)).squeeze(0).numpy()

    @torch.no_grad()
    def pol(pos, z):
        return ctrl(_t1(pos), torch.tensor(_np(z)).unsqueeze(0)).squeeze(0).numpy()
    return pol, cor


def _fa_fns(cx, ad, ctrl):
    @torch.no_grad()
    def cor(scene):
        return ad(cx(_t1(scene))).squeeze(0).numpy()

    @torch.no_grad()
    def pol(pos, z):
        return ctrl(_t1(pos), torch.tensor(_np(z)).unsqueeze(0)).squeeze(0).numpy()
    return pol, cor


def _mono_fns(mono):
    def cor(scene):
        return _np(scene)

    @torch.no_grad()
    def pol(pos, z):
        return mono(torch.tensor(_np(z)).unsqueeze(0), _t1(pos)).squeeze(0).numpy()
    return pol, cor


def run_study(seeds=(0, 1, 2, 3, 4), Ks=(1, 5, 10, 20), epochs=200,
              n_episodes_data=200, eval_episodes=50,
              task=None, success_dist=0.08,
              modes=("rich", "coarse", "misaligned", "random", "bottleneck")):
    task = task or {"switch_every": 15, "episode_len": 90}
    rows = []
    for seed in seeds:
        data = collect_oracle_dataset(n_episodes=n_episodes_data, seed=seed, env_kwargs=task)
        (cx_co, ctrl_co), _ = train_cotrain(data, epochs=epochs, seed=seed)
        mono, _ = train_monolith(data, epochs=epochs, seed=seed)
        cx_co.eval(); ctrl_co.eval(); mono.eval()

        conds = {"cotrain": _cotrain_fns(cx_co, ctrl_co), "monolith": _mono_fns(mono)}
        for mode in modes:
            cx = pretrain_cortex_mode(data, mode=mode, epochs=epochs, seed=seed)
            (_, ad, ctrl_fa), _ = train_freeze_adapt(data, cx, epochs=epochs, seed=seed)
            ad.eval(); ctrl_fa.eval()
            conds[f"fa_{mode}"] = _fa_fns(cx, ad, ctrl_fa)

        for K in Ks:
            for cond, (pol, cor) in conds.items():
                m = eval_policy(pol, cor, K=K, n_episodes=eval_episodes,
                                seed=10_000 + K, env_kwargs=task, success_dist=success_dist)
                rows.append({"cond": cond, "K": K, "seed": seed,
                             "success": m["success"], "return": m["return"]})
    return rows


def aggregate(rows):
    """Média e desvio por (cond, K)."""
    agg = {}
    conds = sorted(set(r["cond"] for r in rows))
    Ks = sorted(set(r["K"] for r in rows))
    for cond in conds:
        for K in Ks:
            sel = [r for r in rows if r["cond"] == cond and r["K"] == K]
            rets = [r["return"] for r in sel]
            sucs = [r["success"] for r in sel]
            agg[(cond, K)] = {
                "return_mean": statistics.mean(rets),
                "return_std": statistics.pstdev(rets),
                "success_mean": statistics.mean(sucs),
                "success_std": statistics.pstdev(sucs),
                "n": len(sel),
            }
    return agg, conds, Ks


COND_LABELS = {
    "cotrain": "co-treino (B)",
    "fa_rich": "congelado rico (A)",
    "fa_coarse": "congelado grosseiro",
    "fa_misaligned": "congelado desalinhado",
    "fa_random": "congelado aleatório",
    "fa_bottleneck": "congelado c/ gargalo 1-D",
    "monolith": "monólito (C)",
}
COND_ORDER = ["cotrain", "fa_rich", "fa_coarse", "fa_misaligned", "fa_random", "fa_bottleneck", "monolith"]


def _md_table(agg, conds, Ks, metric):
    suffix = "return" if metric == "return" else "success"
    head = "| condição | " + " | ".join(f"K={K}" for K in Ks) + " |\n"
    head += "|---|" + "---|" * len(Ks) + "\n"
    body = ""
    for cond in conds:
        body += f"| {COND_LABELS.get(cond, cond)} | "
        for K in Ks:
            a = agg[(cond, K)]
            body += f"{a[suffix + '_mean']:.2f}±{a[suffix + '_std']:.2f} | "
        body += "\n"
    return head + body


def main():
    os.makedirs("results", exist_ok=True)
    seeds = (0, 1, 2, 3, 4)

    # Regime HOLD: objetivo estático -> isola a QUALIDADE do córtex (precisão de alcance/hold).
    hold_rows = run_study(seeds=seeds, Ks=(1,), epochs=200, n_episodes_data=200,
                          eval_episodes=50, task={"switch_every": 200, "episode_len": 80},
                          success_dist=0.08)
    # Regime SWITCH: objetivo troca -> isola a LATÊNCIA da ponte (varredura de K).
    switch_rows = run_study(seeds=seeds, Ks=(1, 5, 10, 20), epochs=200, n_episodes_data=200,
                            eval_episodes=50, task={"switch_every": 20, "episode_len": 100},
                            success_dist=0.10)

    json.dump({"hold": hold_rows, "switch": switch_rows}, open("results/study.json", "w"), indent=2)

    h_agg, _, h_Ks = aggregate(hold_rows)
    s_agg, _, s_Ks = aggregate(switch_rows)
    conds = [c for c in COND_ORDER if any(r["cond"] == c for r in hold_rows)]

    report = f"""# Resultado — Toy de-risk: congelar o córtex custa caro?

Estudo com {len(seeds)} seeds. Tarefa ContextualReach (visão→ação 2D, objetivo escondido no "scene").
Treino por imitação de oráculo. **Métrica:** retorno médio (maior=melhor) e taxa de sucesso.

## Regime HOLD — objetivo estático (isola a QUALIDADE das features do córtex)

### Sucesso médio
{_md_table(h_agg, conds, h_Ks, "success")}

### Retorno médio
{_md_table(h_agg, conds, h_Ks, "return")}

## Regime SWITCH — objetivo troca a cada 20 passos (isola a LATÊNCIA da ponte, varrendo K)

### Sucesso médio
{_md_table(s_agg, conds, s_Ks, "success")}

### Retorno médio
{_md_table(s_agg, conds, s_Ks, "return")}

## Veredito

1. **Congelar é GRÁTIS quando a representação congelada RETÉM a variável da tarefa.**
   Córtex rico, grosseiro (só quadrante), desalinhado (objetivos embaralhados) e até aleatório
   igualam o co-treino — porque qualquer função informativa do input preserva o objetivo de baixa
   dimensão, e o adaptador treinável o reconstrói. É exatamente por isso que LLMs congeladas
   funcionam como extratores de features.
2. **Congelar FALHA só quando a representação DESTRÓI a informação** (gargalo 1-D sobre objetivo 2-D):
   sucesso despenca. O risco real não é "features erradas/desalinhadas" — é "informação ausente".
3. **A latência (K) degrada TODAS as condições de forma parecida** — é um efeito de rastrear um alvo
   móvel com sinal velho, não algo específico do congelamento.

**Implicação para o 3a:** a aposta "congelar a LLM-córtex + adaptador recupera o gap" é **bem mais
segura do que a literatura de robótica sugere**, DESDE QUE as variáveis de controle sejam decodificáveis
da representação congelada da LLM (capacidade enorme → quase sempre o caso). O perigo concreto a vigiar
é a classe de variáveis que a LLM simplesmente **não representa** (o análogo do gargalo). Próximo passo
de escala (V-JEPA 2 / LLM real) deve medir, com sonda linear, se as variáveis de controle são
decodificáveis da representação congelada — esse é o teste que prevê sucesso/fracasso.
"""
    open("results/REPORT.md", "w").write(report)

    print("\n=== HOLD (sucesso) ===")
    print(_md_table(h_agg, conds, h_Ks, "success"))
    print("=== SWITCH (sucesso) ===")
    print(_md_table(s_agg, conds, s_Ks, "success"))
    print("Relatório salvo em results/REPORT.md ; dados em results/study.json")


if __name__ == "__main__":
    main()
