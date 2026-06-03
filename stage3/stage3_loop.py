import json
import os
import numpy as np
import torch
from brain.data import collect_oracle_dataset
from brain.cortex_llm import LLMCortex
from brain.train import train_freeze_adapt_llm
from brain.probe import _probe_r2_on_features
from brain.bridge_async import eval_with_latents
from brain.envs.contextual_reach import ContextualReach


def run_stage3(backend="stub", model_name="Qwen/Qwen2.5-0.5B", layer=None,
               pooling="last", n_episodes_data=200, epochs=200, eval_episodes=50,
               ds=(0, 2, 5, 10, 15, 20, 30, 50), task=None, success_dist=0.10, seed=0):
    task = task or {"switch_every": 20, "episode_len": 100}
    stub_lat = 0.0 if backend == "stub" else 200.0
    cortex = LLMCortex(backend=backend, model_name=model_name, layer=layer,
                       pooling=pooling, stub_latency_ms=stub_lat, seed=seed)
    data = collect_oracle_dataset(n_episodes=n_episodes_data, seed=seed, env_kwargs=task)

    # hidden congelado do treino (uma chamada por amostra) + latencia medida + gatekeeper
    H_list, lats = [], []
    for s in data["scene"]:
        H_list.append(cortex.hidden(s.numpy()))
        lats.append(cortex.lat_ms)
    H = torch.tensor(np.stack(H_list), dtype=torch.float32)
    latency_ms = {"L50": float(np.percentile(lats, 50)), "L95": float(np.percentile(lats, 95))}
    gate_r2 = _probe_r2_on_features(H, data["goal"])

    # treina adapter+controller sobre o hidden congelado
    (adapter, controller), _ = train_freeze_adapt_llm(H, data, epochs=epochs, seed=seed)
    adapter.eval()
    controller.eval()

    @torch.no_grad()
    def policy_fn(pos, z):
        return controller(torch.tensor(pos).unsqueeze(0),
                          torch.tensor(z, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    # PRE-COMPUTA latentes por (episodio, passo) UMA vez (a cena independe das acoes,
    # entao o latente vale para qualquer condicao/atraso no mesmo seed)
    SEED_EVAL = 20000
    latents = []
    for ep in range(eval_episodes):
        env = ContextualReach(seed=SEED_EVAL + ep, **task)
        obs = env.reset()
        done = False
        Lz = []
        with torch.no_grad():
            while not done:
                h = torch.tensor(cortex.hidden(obs["scene"])).unsqueeze(0)
                Lz.append(adapter(h).squeeze(0).numpy())
                obs, _, done = env.step(np.zeros(2, dtype=np.float32))
        latents.append(np.stack(Lz))

    curve = {c: {} for c in ("oracle", "naive", "rtc")}
    for cond in curve:
        for d in ds:
            m = eval_with_latents(latents, policy_fn, condition=cond, d=d,
                                  n_episodes=eval_episodes, seed_base=SEED_EVAL,
                                  env_kwargs=task, success_dist=success_dist)
            curve[cond][d] = m["success"]

    res = {"backend": backend, "latency_ms": latency_ms, "gate_r2": round(gate_r2, 3),
           "curve": curve, "ds": list(ds)}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/stage3.json", "w"), indent=2)
    return res


def main():
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "stub"
    if backend == "mps":
        # escala reduzida: cada cortex.hidden() e um forward REAL da LLM (caro)
        res = run_stage3(backend=backend, n_episodes_data=25, epochs=300,
                         eval_episodes=12, ds=(0, 2, 5, 10, 20, 40),
                         task={"switch_every": 15, "episode_len": 70})
    else:
        res = run_stage3(backend=backend)
    print(json.dumps({k: res[k] for k in ("backend", "latency_ms", "gate_r2")}, indent=2))
    print("\nsucesso x atraso (d):")
    for cond in ("oracle", "naive", "rtc"):
        print(f"  {cond:7s} " + "  ".join(f"d={d}:{res['curve'][cond][d]:.2f}" for d in res["ds"]))
    print("\nresults/stage3.json salvo")


if __name__ == "__main__":
    main()
