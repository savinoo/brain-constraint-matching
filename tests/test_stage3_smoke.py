from stage3.stage3_loop import run_stage3


def test_run_stage3_stub_smoke():
    res = run_stage3(backend="stub", n_episodes_data=20, epochs=40,
                     eval_episodes=15, ds=(0, 5, 15), seed=0)
    assert set(res.keys()) >= {"latency_ms", "gate_r2", "curve"}
    for cond in ("oracle", "naive", "rtc"):
        assert cond in res["curve"]
        assert set(res["curve"][cond].keys()) == {0, 5, 15}
