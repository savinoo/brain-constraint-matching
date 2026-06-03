import numpy as np
from brain.envs.contextual_reach import ContextualReach
from brain.rtc import rtc_blend

SUCCESS_DIST = 0.10


def eval_policy_async(policy_fn, cortex_fn, condition="rtc", d=5, tau=3.0,
                      n_episodes=50, seed=999, pass_env=False,
                      env_kwargs=None, success_dist=SUCCESS_DIST):
    """Loop de duas velocidades, latencia modelada como atraso de d passos na
    chegada do latente (versao deterministica para CI; a versao com thread real
    vive no runner do Mac). Condicoes:
      oracle -> latente sempre fresco (d efetivo 0);
      naive  -> usa o latente que chegou ha d passos, troca abrupta (ZOH);
      rtc    -> blend exponencial entre o latente em uso e o que chegou (suaviza a troca).
    """
    env = ContextualReach(seed=seed, **(env_kwargs or {}))
    total_ret, succ = 0.0, 0
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        hist = []          # latente computado a cada passo (um por passo)
        z_used_prev = None
        step = 0
        ep_ret = 0.0
        reached = False
        while not done:
            z_now = np.asarray(cortex_fn(obs["scene"]), dtype=np.float32)
            hist.append(z_now)
            if condition == "oracle":
                z_eff = z_now
            else:
                z_arrived = hist[max(0, step - d)]       # latente atrasado de d passos
                if condition == "naive" or z_used_prev is None:
                    z_eff = z_arrived                    # troca abrupta (zero-order hold)
                else:                                    # rtc: suaviza a transicao
                    z_eff = rtc_blend(z_used_prev, z_arrived, i=1, d=0, tau=tau)
            z_used_prev = z_eff
            a = policy_fn(obs["pos"], z_eff, env) if pass_env else policy_fn(obs["pos"], z_eff)
            obs, r, done = env.step(a)
            ep_ret += r
            if -r <= success_dist:
                reached = True
            step += 1
        total_ret += ep_ret
        succ += int(reached)
    return {"success": succ / n_episodes, "return": total_ret / n_episodes}


def eval_with_latents_dyn(latents, policy_fn, condition="rtc", d=5, tau=3.0,
                          n_episodes=None, seed_base=20000, env_kwargs=None,
                          success_dist=SUCCESS_DIST):
    """Como eval_with_latents, mas para a tarefa de 2a ordem: a politica ve
    (pos, vel, z). env_kwargs deve incluir inertia=True."""
    n_episodes = n_episodes if n_episodes is not None else len(latents)
    total_ret, succ = 0.0, 0
    for ep in range(n_episodes):
        env = ContextualReach(seed=seed_base + ep, **(env_kwargs or {}))
        obs = env.reset()
        Lz = latents[ep]
        done = False
        step = 0
        z_used_prev = None
        ep_ret = 0.0
        reached = False
        while not done:
            if condition == "oracle":
                z_eff = Lz[min(step, len(Lz) - 1)]
            else:
                z_arrived = Lz[max(0, step - d)]
                if condition == "naive" or z_used_prev is None:
                    z_eff = z_arrived
                else:
                    z_eff = rtc_blend(z_used_prev, z_arrived, i=1, d=0, tau=tau)
            z_used_prev = z_eff
            a = policy_fn(obs["pos"], obs["vel"], z_eff)
            obs, r, done = env.step(a)
            ep_ret += r
            if -r <= success_dist:
                reached = True
            step += 1
        total_ret += ep_ret
        succ += int(reached)
    return {"success": succ / n_episodes, "return": total_ret / n_episodes}


def eval_with_latents(latents, policy_fn, condition="rtc", d=5, tau=3.0,
                      n_episodes=None, seed_base=20000, env_kwargs=None,
                      success_dist=SUCCESS_DIST):
    """Avalia usando latentes PRE-COMPUTADOS por (episodio, passo) — evita rodar o
    cortex por passo (inviavel com LLM real). As cenas independem das acoes
    (scene = W@goal + ruido), entao latents[ep][t] vale para qualquer politica no
    mesmo seed. Condicoes: oracle (latente do passo t) / naive (atraso d, hold) /
    rtc (atraso d + suavizacao)."""
    n_episodes = n_episodes if n_episodes is not None else len(latents)
    total_ret, succ = 0.0, 0
    for ep in range(n_episodes):
        env = ContextualReach(seed=seed_base + ep, **(env_kwargs or {}))
        obs = env.reset()
        Lz = latents[ep]
        done = False
        step = 0
        z_used_prev = None
        ep_ret = 0.0
        reached = False
        while not done:
            if condition == "oracle":
                z_eff = Lz[min(step, len(Lz) - 1)]
            else:
                z_arrived = Lz[max(0, step - d)]
                if condition == "naive" or z_used_prev is None:
                    z_eff = z_arrived
                else:
                    z_eff = rtc_blend(z_used_prev, z_arrived, i=1, d=0, tau=tau)
            z_used_prev = z_eff
            a = policy_fn(obs["pos"], z_eff)
            obs, r, done = env.step(a)
            ep_ret += r
            if -r <= success_dist:
                reached = True
            step += 1
        total_ret += ep_ret
        succ += int(reached)
    return {"success": succ / n_episodes, "return": total_ret / n_episodes}
