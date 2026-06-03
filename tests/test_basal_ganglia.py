import numpy as np
from brain.basal_ganglia import BasalGanglia
from stage5.arbitrary_bandit import ArbitraryBandit


def test_bandit_arbitrary_mapping_and_reward():
    env = ArbitraryBandit(n_contexts=5, n_actions=3, d=8, seed=0)
    c, z = env.sample_context()
    assert z.shape == (8,)
    correct = int(env.correct[c])
    assert env.reward(c, correct) == 1.0
    assert env.reward(c, (correct + 1) % 3) == 0.0


def test_bg_learns_arbitrary_mapping_from_reward():
    env = ArbitraryBandit(n_contexts=6, n_actions=4, d=16, seed=0)
    bg = BasalGanglia(d=16, n_actions=4, lr=0.2, seed=0)
    rng = np.random.default_rng(1)
    for t in range(4000):
        c, z = env.sample_context()
        eps = max(0.05, 1.0 - t / 2000)
        a = bg.act(z, epsilon=eps, rng=rng)
        bg.update(z, a, env.reward(c, a))
    # avaliacao greedy: aprendeu o mapeamento arbitrario bem acima do acaso (0.25)
    acc = np.mean([env.reward(ci, bg.act(env.Z[ci], epsilon=0.0)) for ci in range(6)])
    assert acc >= 0.8
