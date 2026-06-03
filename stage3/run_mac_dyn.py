"""Runner DINAMICO do estagio 3 no Mac (Qwen real, MPS). Escala reduzida p/ caber no tempo."""
import json
from stage3.dynamics_loop import run_stage3_dyn

r = run_stage3_dyn(backend="mps", n_episodes_data=12, epochs=400, eval_episodes=8,
                   ds=(0, 2, 5, 10, 20, 40), task={"switch_every": 20, "episode_len": 60})

print("LAT", json.dumps(r["latency_ms"]), "GATE_R2", r["gate_r2"], flush=True)
for c in ("oracle", "naive", "rtc"):
    line = "  ".join("d=%d:%.2f" % (d, r["curve"][c][d]) for d in r["ds"])
    print("%-7s %s" % (c, line), flush=True)
print("salvo results/stage3_dyn.json", flush=True)
