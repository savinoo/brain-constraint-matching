import numpy as np
from brain.envs.contextual_reach import ContextualReach

SUCCESS_DIST = 0.15  # alcançou se chegou a essa distância em algum passo do episódio


def eval_policy(policy_fn, cortex_fn, K=1, n_episodes=50, seed=999, pass_env=False,
                env_kwargs=None, success_dist=SUCCESS_DIST):
    """Avalia em malha fechada com ponte de duas velocidades.

    cortex_fn(scene) -> z é recomputado SÓ a cada K passos (lento);
    policy_fn(pos, z[, env]) -> ação roda TODO passo (rápido) com o z possivelmente velho.
    Mede retorno médio e taxa de sucesso. K>1 simula a latência do córtex.
    env_kwargs configura a dificuldade da tarefa; success_dist o limiar de alcance.
    """
    env = ContextualReach(seed=seed, **(env_kwargs or {}))
    total_return, successes = 0.0, 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        step = 0
        z = None
        ep_ret = 0.0
        reached = False
        while not done:
            if step % K == 0 or z is None:
                z = cortex_fn(obs["scene"])  # atualização lenta do córtex
            if pass_env:
                a = policy_fn(obs["pos"], z, env)
            else:
                a = policy_fn(obs["pos"], z)
            obs, r, done = env.step(a)
            ep_ret += r
            if -r <= success_dist:
                reached = True
            step += 1
        total_return += ep_ret
        successes += int(reached)
    return {"return": total_return / n_episodes, "success": successes / n_episodes}
