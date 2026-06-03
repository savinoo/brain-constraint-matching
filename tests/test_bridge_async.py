import numpy as np
from brain.bridge_async import eval_policy_async


def _oracle_policy(pos, z, env):
    return np.clip(env.goal - pos, -1.0, 1.0)


def test_three_conditions_run_and_oracle_high():
    out = {}
    for cond in ("oracle", "naive", "rtc"):
        out[cond] = eval_policy_async(
            policy_fn=_oracle_policy, cortex_fn=lambda scene: np.zeros(2, dtype=np.float32),
            condition=cond, d=5, n_episodes=10, seed=7, pass_env=True,
            env_kwargs={"switch_every": 200, "episode_len": 60})
    assert out["oracle"]["success"] >= 0.8
    assert set(out["rtc"].keys()) == {"success", "return"}
