"""Estagio 3 — versao com DINAMICA (2a ordem). A tarefa tem inercia, entao mudancas
bruscas de alvo custam (overshoot) — o regime onde o RTC pode ajudar. Espelha
stage3_loop, mas usa env inertia=True, ControllerDyn (ve pos+vel) e eval ciente de vel.
"""
import json
import os
import numpy as np
import torch
from brain.data import collect_oracle_dataset
from brain.cortex_llm import LLMCortex
from brain.train import train_freeze_adapt_llm_dyn
from brain.probe import _probe_r2_on_features
from brain.bridge_async import eval_with_latents_dyn
from brain.envs.contextual_reach import ContextualReach


def run_stage3_dyn(backend="stub", model_name="Qwen/Qwen2.5-0.5B", layer=None,
                   pooling="last", n_episodes_data=40, epochs=300, eval_episodes=20,
                   ds=(0, 2, 5, 10, 20, 40), task=None, success_dist=0.10, seed=0):
    base = {"switch_every": 25, "episode_len": 80, "inertia": True}
    if task:
        base.update(task)
    task = base
    stub_lat = 0.0 if backend == "stub" else 0.0  # latencia injetada via d
    cortex = LLMCortex(backend=backend, model_name=model_name, layer=layer,
                       pooling=pooling, stub_latency_ms=stub_lat, seed=seed)
    data = collect_oracle_dataset(n_episodes=n_episodes_data, seed=seed, env_kwargs=task)

    H_list, lats = [], []
    for s in data["scene"]:
        H_list.append(cortex.hidden(s.numpy()))
        lats.append(cortex.lat_ms)
    H = torch.tensor(np.stack(H_list), dtype=torch.float32)
    latency_ms = {"L50": float(np.percentile(lats, 50)), "L95": float(np.percentile(lats, 95))}
    gate_r2 = _probe_r2_on_features(H, data["goal"])

    (adapter, controller), _ = train_freeze_adapt_llm_dyn(H, data, epochs=epochs, seed=seed)
    adapter.eval()
    controller.eval()

    @torch.no_grad()
    def policy_fn(pos, vel, z):
        return controller(torch.tensor(pos).unsqueeze(0), torch.tensor(vel).unsqueeze(0),
                          torch.tensor(z, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    # pre-computa latentes por (episodio, passo) — a cena independe das acoes
    SEED_EVAL = 30000
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
            m = eval_with_latents_dyn(latents, policy_fn, condition=cond, d=d,
                                      n_episodes=eval_episodes, seed_base=SEED_EVAL,
                                      env_kwargs=task, success_dist=success_dist)
            curve[cond][d] = m["success"]

    res = {"backend": backend, "dynamics": True, "latency_ms": latency_ms,
           "gate_r2": round(gate_r2, 3), "curve": curve, "ds": list(ds)}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/stage3_dyn.json", "w"), indent=2)
    return res


def main():
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "stub"
    if backend == "mps":
        res = run_stage3_dyn(backend=backend, n_episodes_data=18, epochs=400,
                             eval_episodes=12, ds=(0, 2, 5, 10, 20, 40))
    else:
        res = run_stage3_dyn(backend=backend)
    print("LAT", json.dumps(res["latency_ms"]), "GATE_R2", res["gate_r2"], flush=True)
    for c in ("oracle", "naive", "rtc"):
        line = "  ".join("d=%d:%.2f" % (d, res["curve"][c][d]) for d in res["ds"])
        print("%-7s %s" % (c, line), flush=True)
    print("salvo results/stage3_dyn.json", flush=True)


if __name__ == "__main__":
    main()
