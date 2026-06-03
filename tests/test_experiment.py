from brain.experiment import run_experiment


def test_run_experiment_smoke():
    res = run_experiment(n_episodes_data=10, epochs=40, eval_episodes=20, Ks=(1, 5), seed=0)
    assert set(res.keys()) == {"cotrain", "freeze_adapt", "monolith", "verdict"}
    for cond in ("cotrain", "freeze_adapt", "monolith"):
        assert "success" in res[cond][1]
    assert set(res["verdict"]["recovery_by_K"].keys()) == {1, 5}
