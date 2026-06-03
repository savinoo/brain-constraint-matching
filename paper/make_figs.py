"""Gera as figuras do paper (PDF, para LaTeX) a partir dos resultados reais."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figs")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
C_RAG, C_BF = "#2c6fbb", "#c0392b"

# --- Fig 1: E1 capacity boundary (info >= task dimension) ---
dims = [1, 2, 3, 4]
succ = [0.25, 1.00, 1.00, 1.00]
fig, ax = plt.subplots(figsize=(4.0, 2.7))
ax.plot(dims, succ, "o-", color=C_RAG, lw=2, ms=7)
ax.axhline(0.25, ls=":", color="gray", lw=1)
ax.text(3.0, 0.30, "chance", color="gray", fontsize=9)
ax.set_xlabel("frozen bottleneck capacity (dims)")
ax.set_ylabel("recall@1")
ax.set_xticks(dims)
ax.set_ylim(0, 1.05)
ax.set_title("E1: freezing fails iff capacity < task dim (=2)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_e1.pdf"))

# --- Fig 2: E3 on REAL Qwen embeddings ---
e3 = json.load(open(os.path.join(ROOT, "stage4", "result-mac-real-e3.json")))
corrs = [0.5, 0.7, 0.9]
rag = [e3["corr=%.1f" % c]["RAG"] for c in corrs]
dgca3 = [e3["corr=%.1f" % c]["best_DG_CA3"] for c in corrs]
x = range(len(corrs))
fig, ax = plt.subplots(figsize=(4.0, 2.7))
w = 0.38
ax.bar([i - w / 2 for i in x], rag, w, label="RAG (raw cosine)", color=C_RAG)
ax.bar([i + w / 2 for i in x], dgca3, w, label="DG+CA3 (best $k$)", color=C_BF)
ax.set_xticks(list(x))
ax.set_xticklabels(["%.1f" % c for c in corrs])
ax.set_xlabel("cue corruption")
ax.set_ylabel("recall@1")
ax.set_ylim(0, 1.0)
ax.set_title("E3: real Qwen embeddings", fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_e3.pdf"))

# --- Fig 3: E5 pre-registered (eligibility traces vs one-step) ---
dc = json.load(open(os.path.join(ROOT, "results", "delayed_credit.json")))
Ks = sorted(int(k) for k in dc["0.0"])
one = [dc["0.0"][str(k)][0] for k in Ks]
lam = [dc["0.9"][str(k)][0] for k in Ks]
fig, ax = plt.subplots(figsize=(4.0, 2.7))
ax.plot(Ks, lam, "o-", color=C_BF, lw=2, ms=6, label=r"TD($\lambda$=0.9), traces")
ax.plot(Ks, one, "s--", color=C_RAG, lw=2, ms=6, label=r"TD($\lambda$=0), one-step")
ax.set_xlabel("delay gap $K$")
ax.set_ylabel("greedy accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("E5: pre-registered, delayed reward", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_e5.pdf"))

print("figuras salvas em", OUT, ":", os.listdir(OUT))
