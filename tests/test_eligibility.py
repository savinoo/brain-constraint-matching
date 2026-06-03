from stage5.delayed_credit import run_one


def test_td_lambda_beats_one_step_under_delay():
    # com atraso K=8, tracos (lambda=0.9) aprendem; um passo (lambda=0) fica para tras
    acc_traces = sum(run_one(C=6, A=4, K=8, lam=0.9, episodes=150, seed=s) for s in range(3)) / 3
    acc_onestep = sum(run_one(C=6, A=4, K=8, lam=0.0, episodes=150, seed=s) for s in range(3)) / 3
    assert acc_traces > 0.9
    assert acc_traces > acc_onestep + 0.4


def test_no_delay_ties():
    # sem atraso (K=0) os dois empatam (restricao ausente -> sem vantagem)
    a_tr = sum(run_one(C=6, A=4, K=0, lam=0.9, episodes=150, seed=s) for s in range(3)) / 3
    a_os = sum(run_one(C=6, A=4, K=0, lam=0.0, episodes=150, seed=s) for s in range(3)) / 3
    assert a_tr > 0.9 and a_os > 0.9
