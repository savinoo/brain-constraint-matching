import numpy as np
import torch
from brain.envs.contextual_reach import ContextualReach


def collect_oracle_dataset(n_episodes=200, seed=0):
    """Rola o oráculo e coleta (scene, pos, goal, ação-do-oráculo) para imitação."""
    env = ContextualReach(seed=seed)
    scenes, poss, goals, acts = [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = env.oracle_action()
            scenes.append(obs["scene"]); poss.append(obs["pos"])
            goals.append(obs["goal"]); acts.append(a)
            obs, _, done = env.step(a)
    to_t = lambda xs: torch.tensor(np.array(xs, dtype=np.float32))
    return {"scene": to_t(scenes), "pos": to_t(poss), "goal": to_t(goals), "action": to_t(acts)}
