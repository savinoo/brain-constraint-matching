import numpy as np
from brain.envs.contextual_reach import ContextualReach, SCENE_DIM, GOAL_DIM, POS_DIM, ACTION_DIM


def test_reset_obs_shapes():
    env = ContextualReach(seed=0)
    obs = env.reset()
    assert obs["scene"].shape == (SCENE_DIM,)
    assert obs["pos"].shape == (POS_DIM,)
    assert obs["goal"].shape == (GOAL_DIM,)
    assert np.allclose(obs["pos"], 0.0)


def test_oracle_reduces_distance():
    env = ContextualReach(seed=1)
    obs = env.reset()
    d0 = np.linalg.norm(obs["pos"] - obs["goal"])
    for _ in range(10):
        obs, r, done = env.step(env.oracle_action())
    d1 = np.linalg.norm(obs["pos"] - obs["goal"])
    assert d1 < d0


def test_goal_switches_on_schedule():
    env = ContextualReach(seed=2, switch_every=5)
    obs = env.reset()
    g_start = obs["goal"].copy()
    for _ in range(4):
        obs, r, done = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert np.allclose(obs["goal"], g_start)
    obs, r, done = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert not np.allclose(obs["goal"], g_start)


def test_scene_encodes_goal_deterministically_up_to_noise():
    env = ContextualReach(seed=3, noise=0.0)
    obs = env.reset()
    assert np.allclose(obs["scene"], env.W @ obs["goal"], atol=1e-5)
