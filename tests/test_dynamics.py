import numpy as np
import torch
from brain.envs.contextual_reach import ContextualReach
from brain.data import collect_oracle_dataset
from brain.models import ControllerDyn, H_DIM, ACTION_DIM
from brain.train import train_freeze_adapt_llm_dyn
from stage3.dynamics_loop import run_stage3_dyn


def test_inertia_oracle_reaches_and_obs_has_vel():
    env = ContextualReach(seed=0, inertia=True, switch_every=200, episode_len=100)
    obs = env.reset()
    assert "vel" in obs and obs["vel"].shape == (2,)
    reached = False
    done = False
    while not done:
        obs, r, done = env.step(env.oracle_action())
        if -r <= 0.10:
            reached = True
    assert reached  # PD alcanca o alvo estatico


def test_controller_dyn_and_train_runs():
    data = collect_oracle_dataset(n_episodes=15, seed=0, env_kwargs={"inertia": True})
    assert "vel" in data
    H = torch.randn(data["scene"].shape[0], H_DIM)
    (adapter, controller), losses = train_freeze_adapt_llm_dyn(H, data, epochs=40, seed=0)
    assert losses[-1] < losses[0]
    assert isinstance(controller, ControllerDyn)
    out = controller(torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 8))
    assert out.shape == (3, ACTION_DIM)


def test_run_stage3_dyn_stub_smoke():
    res = run_stage3_dyn(backend="stub", n_episodes_data=12, epochs=40,
                         eval_episodes=8, ds=(0, 5, 20), seed=0)
    assert res["dynamics"] is True
    for cond in ("oracle", "naive", "rtc"):
        assert set(res["curve"][cond].keys()) == {0, 5, 20}
