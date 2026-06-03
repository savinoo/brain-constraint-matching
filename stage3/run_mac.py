"""Runner do estagio 3 no Mac (Qwen real, MPS). Escala reduzida p/ caber no tempo.
Usa %-formatacao (sem f-string aninhada — compativel com Python 3.11)."""
import json
from stage3.stage3_loop import run_stage3

r = run_stage3(backend="mps", n_episodes_data=12, epochs=300, eval_episodes=8,
               ds=(0, 2, 5, 10, 20, 40), task={"switch_every": 15, "episode_len": 50})

print("LAT", json.dumps(r["latency_ms"]), "GATE_R2", r["gate_r2"], flush=True)
for c in ("oracle", "naive", "rtc"):
    line = "  ".join("d=%d:%.2f" % (d, r["curve"][c][d]) for d in r["ds"])
    print("%-7s %s" % (c, line), flush=True)
json.dump(r, open("results/stage3.json", "w"), indent=2)
print("salvo results/stage3.json", flush=True)
