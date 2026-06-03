"""E3 sobre embeddings REAIS de LLM (Qwen2.5-0.5B, MPS) — fecha o caveat do substrato
sintetico. Embeda 200 textos curtos, roda a ablacao recall RAG vs DG+CA3 (k-sweep) sob
corrupcao. Pergunta: o null do E3 (RAG >= DG/CA3) se mantem em embeddings reais, anisotropicos?"""
import json
import statistics
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from brain.hippocampus import Hippocampus

COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
ANIMALS = ["fox", "dog", "cat", "bird", "horse", "wolf", "bear", "mouse", "lion", "frog"]
VERBS = ["chased", "saw", "ate", "found", "carried", "watched", "followed", "ignored"]
OBJS = ["ball", "house", "river", "stone", "tree", "car", "box", "flower", "moon", "key"]


def gen_texts(n, seed=0):
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        t = "The %s %s %s the %s." % (rng.choice(COLORS), rng.choice(ANIMALS),
                                      rng.choice(VERBS), rng.choice(OBJS))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def embed(model, tok, texts, layer):
    embs = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to("mps")
        with torch.no_grad():
            hs = model(**enc).hidden_states[layer][0]
        embs.append(hs[-1].float().cpu().numpy())
    E = np.stack(embs)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return torch.tensor(E, dtype=torch.float32)


def recall(E, use_dg, sparsity, corruption, seed):
    d = E.shape[1]
    hip = Hippocampus(d, use_dg=use_dg, use_ca3=use_dg, use_contiguity=False,
                      use_gate=False, sparsity=sparsity, beta=8.0, seed=seed)
    for e in E:
        hip.write(e)
    g = torch.Generator().manual_seed(seed + 777)
    hits = 0
    for i, e in enumerate(E):
        mask = (torch.rand(d, generator=g) > corruption).float()
        if hip.recall_top1_id(e * mask) == i:
            hits += 1
    return hits / len(E)


def main():
    name = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, output_hidden_states=True).to("mps").eval()
    layer = model.config.num_hidden_layers // 2
    E = embed(model, tok, gen_texts(200), layer)
    print("embeddings reais:", tuple(E.shape), "(Qwen camada %d)" % layer)
    seeds = range(3)
    res = {}
    for corr in (0.5, 0.7, 0.9):
        rag = statistics.mean(recall(E, False, 0.05, corr, s) for s in seeds)
        best_dg, best_k = 0.0, None
        for k in (0.05, 0.25, 1.0):
            v = statistics.mean(recall(E, True, k, corr, s) for s in seeds)
            if v > best_dg:
                best_dg, best_k = v, k
        res["corr=%.1f" % corr] = {"RAG": round(rag, 3), "best_DG_CA3": round(best_dg, 3), "best_k": best_k}
        print("corr=%.1f: RAG=%.2f  melhor DG+CA3=%.2f (k=%s)  -> %s" %
              (corr, rag, best_dg, best_k, "RAG >= DG" if rag >= best_dg - 0.005 else "DG ganha"))
    json.dump(res, open("results/real_embed_e3.json", "w"), indent=2)
    print("salvo results/real_embed_e3.json")


if __name__ == "__main__":
    main()
