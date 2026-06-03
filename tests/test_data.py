from brain.data import collect_oracle_dataset
from brain.models import SCENE_DIM, POS_DIM, GOAL_DIM, ACTION_DIM


def test_dataset_shapes():
    data = collect_oracle_dataset(n_episodes=5, seed=0)
    n = data["scene"].shape[0]
    assert n == 5 * 100  # episode_len padrão = 100
    assert data["scene"].shape == (n, SCENE_DIM)
    assert data["pos"].shape == (n, POS_DIM)
    assert data["goal"].shape == (n, GOAL_DIM)
    assert data["action"].shape == (n, ACTION_DIM)
    assert data["scene"].dtype.is_floating_point
