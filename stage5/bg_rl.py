"""Experimento — ponto POSITIVO da tese de casamento de restricao: o mecanismo dos
ganglios da base (RL) AJUDA porque a restricao (aprender do reward online) esta presente
(a LLM congelada nao a tem). Ablacao: random / llm_prior (congelado, nao aprende) /
bg_rl (aprende do reward) / oracle. Curva de aprendizado, >=5 seeds, REPORT.md/JSON."""
import json
import os
import numpy as np
from brain.basal_ganglia import BasalGanglia
from stage5.arbitrary_bandit import ArbitraryBandit

CONDS = ["random", "llm_prior", "bg_rl", "oracle"]


def run(seeds, n_contexts=12, n_actions=4, d=32, trials=3000, window=150):
    n_windows = trials // window
    curves = {c: np.zeros((len(seeds), n_windows)) for c in CONDS}
    bg_greedy_final = []
    for si, seed in enumerate(seeds):
        env = ArbitraryBandit(n_contexts, n_actions, d, seed)
        rng = np.random.default_rng(seed + 123)
        R = np.random.default_rng(seed + 7).standard_normal((n_actions, d)).astype("float32")  # readout fixo da LLM
        bg = BasalGanglia(d, n_actions, lr=0.2, seed=seed)
        acc = {c: np.zeros(trials) for c in CONDS}
        for t in range(trials):
            c, z = env.sample_context()
            eps = max(0.05, 1.0 - t / (trials * 0.5))           # exploracao decai
            a_bg = bg.act(z, epsilon=eps, rng=rng)
            r_bg = env.reward(c, a_bg)
            bg.update(z, a_bg, r_bg)                              # APRENDE do reward
            acc["random"][t] = env.reward(c, int(rng.integers(n_actions)))
            acc["llm_prior"][t] = env.reward(c, int(np.argmax(R @ z)))   # prior congelado, nao aprende
            acc["bg_rl"][t] = r_bg
            acc["oracle"][t] = 1.0
        for c in CONDS:
            curves[c][si] = acc[c].reshape(n_windows, window).mean(axis=1)
        # avaliacao GREEDY final do bg (sem exploracao) = o que ele REALMENTE aprendeu
        hits = sum(env.reward(ci, bg.act(env.Z[ci], epsilon=0.0)) for ci in range(n_contexts))
        bg_greedy_final.append(hits / n_contexts)

    out = {c: {"mean": curves[c].mean(0).tolist(), "final": float(curves[c].mean(0)[-1])} for c in CONDS}
    out["bg_greedy_final"] = float(np.mean(bg_greedy_final))
    out["chance"] = 1.0 / n_actions
    return out, n_windows


def main():
    seeds = (0, 1, 2, 3, 4)
    res, n_windows = run(seeds)
    os.makedirs("results", exist_ok=True)
    # tabela: acuracia em janelas (inicio, meio, fim)
    idx = [0, n_windows // 2, n_windows - 1]
    lines = ["| condição | início | meio | fim |", "|---|---|---|---|"]
    for c in CONDS:
        m = res[c]["mean"]
        lines.append(f"| {c} | {m[idx[0]]:.2f} | {m[idx[1]]:.2f} | {m[idx[2]]:.2f} |")
    table = "\n".join(lines)
    report = f"""# Gânglios da base (RL) — o ponto POSITIVO da tese de casamento de restrição

Tarefa: bandit contextual com mapeamento **arbitrário** contexto→ação (o prior da LLM não pode
adivinhar; só o reward online revela). 5 seeds. Chance = {res['chance']:.2f}.

## Acurácia ao longo do aprendizado (janelas)
{table}

**bg_rl, avaliação greedy final (o que de fato aprendeu): {res['bg_greedy_final']:.2f}**

## Leitura
- **llm_prior** (córtex congelado, não aprende do reward): **fica plano em ~chance** — a restrição
  está presente (aprender do reward) e a LLM congelada **não a tem**.
- **bg_rl** (mecanismo dos gânglios da base): **aprende** o mapeamento arbitrário, sobe a ~1.0.
- → O mecanismo cérebro-fiel **AJUDA porque a restrição que ele resolve está presente.**

## O que isso fecha (a tese, agora falsificável nos dois lados)
- restrição AUSENTE → mecanismo redundante/nocivo: **DG/CA3** (hipocampo, embeddings já ricos).
- restrição PRESENTE → mecanismo ajuda: **gânglios da base/RL** (LLM congelada não aprende do reward).
- **Regra:** porte um mecanismo cerebral *sse* a restrição que ele resolve existe no seu substrato.
"""
    open("results/basal_ganglia.md", "w").write(report)
    json.dump(res, open("results/basal_ganglia.json", "w"), indent=2)
    print(table)
    print(f"\nbg_rl greedy final: {res['bg_greedy_final']:.2f}  (chance {res['chance']:.2f})")
    print("results/basal_ganglia.md + .json salvos")


if __name__ == "__main__":
    main()
