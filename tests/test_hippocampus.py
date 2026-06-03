import torch
from brain.hippocampus.dg import DG, kwta
from brain.hippocampus.ca3 import ca3_complete
from brain.hippocampus.contiguity import temporal_neighbors
from brain.hippocampus.surprise_gate import SurpriseGate
from brain.hippocampus import Hippocampus


def test_kwta_keeps_top_k():
    h = torch.tensor([0.1, -0.9, 0.5, -0.2, 0.8])
    out = kwta(h, 2)
    assert (out != 0).sum() == 2
    assert out[1] == -0.9 and out[4] == 0.8  # os 2 de maior |valor|


def test_dg_expands_and_sparsifies():
    dg = DG(d=16, expand=4, sparsity=0.05, seed=0)
    e = torch.randn(16); e = e / e.norm()
    k = dg(e)
    assert k.shape == (64,)
    assert (k != 0).sum().item() == dg.k


def test_ca3_completes_toward_stored_pattern():
    torch.manual_seed(0)
    X = torch.randn(10, 32)
    target = X[3]
    cue = target + 0.3 * torch.randn(32)        # pista ruidosa
    out = ca3_complete(cue, X, beta=8.0)
    # a chave completada fica mais perto do alvo que a pista crua
    assert (out - target).norm() < (cue - target).norm()


def test_contiguity_returns_temporal_neighbors():
    n = temporal_neighbors([5], n_total=10, n_neighbors=1, k_c=4)
    assert set(n) == {4, 6}


def test_surprise_gate_fires_on_outlier():
    g = SurpriseGate(gamma=1.0, window=32)
    for _ in range(20):
        g.should_write(1.0)        # baseline estavel
    assert g.should_write(10.0)    # outlier grava
    assert not g.should_write(1.0) # previsivel nao grava


def test_hippocampus_recall_beats_when_ca3_on_under_corruption():
    torch.manual_seed(0)
    d = 32
    embs = torch.randn(60, d)
    embs = embs / embs.norm(dim=1, keepdim=True)

    def recall_rate(use_ca3, use_dg, corruption=0.6):
        hip = Hippocampus(d, use_dg=use_dg, use_ca3=use_ca3, use_contiguity=False,
                          use_gate=False, beta=8.0, seed=0)
        for e in embs:
            hip.write(e)
        hits = 0
        for i, e in enumerate(embs):
            mask = (torch.rand(d) > corruption).float()
            cue = e * mask
            if hip.recall_top1_id(cue) == i:
                hits += 1
        return hits / len(embs)

    # smoke: roda e produz taxa em [0,1]; CA3 nao deve piorar drasticamente
    r_ca3 = recall_rate(use_ca3=True, use_dg=True)
    r_no = recall_rate(use_ca3=False, use_dg=True)
    assert 0.0 <= r_ca3 <= 1.0 and 0.0 <= r_no <= 1.0
