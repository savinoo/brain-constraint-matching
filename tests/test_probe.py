from brain.data import collect_oracle_dataset
from brain.study import pretrain_cortex_mode
from brain.probe import linear_probe_r2


def test_probe_high_for_rich_low_for_bottleneck():
    data = collect_oracle_dataset(n_episodes=60, seed=0)
    cx_rich = pretrain_cortex_mode(data, mode="rich", epochs=150, seed=0)
    cx_bottle = pretrain_cortex_mode(data, mode="bottleneck", epochs=150, seed=0, bottleneck=1)
    r2_rich = linear_probe_r2(cx_rich, data)
    r2_bottle = linear_probe_r2(cx_bottle, data)
    assert r2_rich > 0.9           # representação rica carrega o objetivo
    assert r2_bottle < r2_rich     # gargalo perde informação
