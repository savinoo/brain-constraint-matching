"""Experimento do PRE-REGISTRO: TD(lambda) vs TD de um passo em recompensa ATRASADA.
Tarefa: a cada episodio, um contexto c; o agente escolhe a acao no estado ("choose",c);
seguem K passos distratores ("wait"); a recompensa (1 se a==correct[c]) so chega no fim.
O agente precisa atribuir o credito da recompensa final de volta a decisao, atraves do gap K.
Predicao registrada antes: TD(lambda) vence, e a vantagem cresce com K."""
import json
import os
import statistics
import numpy as np
from brain.basal_ganglia.eligibility import TDLambdaQ


def run_one(C, A, K, lam, episodes, seed):
    rng = np.random.default_rng(seed)
    correct = rng.integers(0, A, size=C)
    agent = TDLambdaQ(A, alpha=0.3, gamma=0.97, lam=lam, seed=seed)
    for ep in range(episodes):
        agent.episode_reset()
        c = int(rng.integers(C))
        eps = max(0.1, 1.0 - ep / (episodes * 0.6))
        states = [("choose", c)] + [("wait", c, j) for j in range(1, K + 1)]
        a0 = agent.act(states[0], A, eps)
        actions = [a0] + [0] * K
        for t in range(len(states)):
            st, at = states[t], actions[t]
            if t < len(states) - 1:
                s2, a2, r, done = states[t + 1], actions[t + 1], 0.0, False
            else:
                s2, a2, r, done = st, 0, (1.0 if a0 == correct[c] else 0.0), True
            agent.step(st, at, r, s2, a2, done)
    hits = sum(1 for c in range(C) if int(np.argmax(agent.Q[("choose", c)])) == correct[c])
    return hits / C


def sweep(seeds, C=6, A=4, Ks=(0, 4, 8, 16, 24), episodes=150):
    out = {}
    for lam in (0.0, 0.9):
        out[lam] = {}
        for K in Ks:
            vals = [run_one(C, A, K, lam, episodes, s) for s in seeds]
            out[lam][K] = (statistics.mean(vals), statistics.pstdev(vals))
    return out, list(Ks)


def main():
    seeds = (0, 1, 2, 3, 4)
    res, Ks = sweep(seeds)
    os.makedirs("results", exist_ok=True)
    json.dump({str(l): {str(k): res[l][k] for k in Ks} for l in res},
              open("results/delayed_credit.json", "w"), indent=2)
    print("Acuracia greedy (media 5 seeds) vs gap K, orcamento fixo de episodios:")
    print("gap K      | " + " | ".join(f"K={k:<4d}" for k in Ks))
    print("TD 1-passo | " + " | ".join(f"{res[0.0][k][0]:.2f}  " for k in Ks))
    print("TD(lambda) | " + " | ".join(f"{res[0.9][k][0]:.2f}  " for k in Ks))
    print("vantagem   | " + " | ".join(f"{res[0.9][k][0]-res[0.0][k][0]:+.2f} " for k in Ks))


if __name__ == "__main__":
    main()
