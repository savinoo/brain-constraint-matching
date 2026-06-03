import numpy as np
from brain.envs.contextual_reach import ContextualReach
from brain.bridge import eval_policy


def test_oracle_policy_gets_high_success_sync():
    cortex_fn = lambda scene: None
    def policy_fn(pos, z, env):
        return np.clip(env.goal - pos, -1.0, 1.0)
    out = eval_policy(policy_fn, cortex_fn, K=1, n_episodes=20, seed=999, pass_env=True)
    assert out["success"] > 0.8


def test_returns_dict_with_keys():
    cortex_fn = lambda scene: np.zeros(8, dtype=np.float32)
    policy_fn = lambda pos, z: np.zeros(2, dtype=np.float32)
    out = eval_policy(policy_fn, cortex_fn, K=1, n_episodes=3, seed=0)
    assert set(out.keys()) == {"return", "success"}


def test_staleness_K_does_not_crash_and_zero_policy_fails():
    cortex_fn = lambda scene: np.zeros(8, dtype=np.float32)
    policy_fn = lambda pos, z: np.zeros(2, dtype=np.float32)
    out = eval_policy(policy_fn, cortex_fn, K=5, n_episodes=5, seed=1)
    assert out["success"] < 0.5
