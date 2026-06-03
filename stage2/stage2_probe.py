"""Estágio 2 — sonda linear sobre um modelo de visão REAL congelado (no Mac, MPS).

Pergunta: variáveis de controle (posição, tamanho, cor) são decodificáveis LINEARMENTE
da representação congelada de uma ResNet18 pré-treinada? Se sim, congelar é seguro —
réplica, num modelo real, do achado do brinquedo.

Compara: pré-treinado (rico) · aleatório (mirror do 'random' do brinquedo) ·
gargalo (projeta para d dims -> curva de capacidade, mirror do limiar do brinquedo).
"""
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision.models import resnet18, ResNet18_Weights

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMG = 96
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VARS = ["x", "y", "tamanho", "R", "G", "B"]


def render(x, y, size, rgb):
    img = Image.new("RGB", (IMG, IMG), (127, 127, 127))
    d = ImageDraw.Draw(img)
    cx, cy = int(x * IMG), int(y * IMG)
    s = int(6 + size * 22)
    d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=tuple(int(c * 255) for c in rgb))
    return img


def make_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.2, 0.8, size=(n, 2))
    size = rng.uniform(0.0, 1.0, size=(n, 1))
    rgb = rng.uniform(0.0, 1.0, size=(n, 3))
    targets = np.concatenate([xy, size, rgb], axis=1).astype(np.float32)
    imgs = np.stack([np.asarray(render(t[0], t[1], t[2], t[3:6]), dtype=np.float32) / 255.0
                     for t in targets])
    x = torch.tensor(imgs).permute(0, 3, 1, 2)
    x = (x - MEAN) / STD
    return x, torch.tensor(targets)


def feature_body(kind):
    m = resnet18(weights=ResNet18_Weights.DEFAULT) if kind == "pretrained" else resnet18(weights=None)
    body = nn.Sequential(*list(m.children())[:-2])  # até layer4 (B,512,3,3) p/ 96px
    body.eval().to(DEVICE)
    for p in body.parameters():
        p.requires_grad_(False)
    return body


@torch.no_grad()
def extract(body, x):
    feats = []
    for i in range(0, x.shape[0], 64):
        f = body(x[i:i + 64].to(DEVICE))
        f = torch.nn.functional.adaptive_avg_pool2d(f, (3, 3)).flatten(1)
        feats.append(f.cpu())
    return torch.cat(feats, 0)


def probe_r2(F, G, test_frac=0.25, seed=0):
    n = F.shape[0]
    nt = max(1, int(n * test_frac))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    te, tr = perm[:nt], perm[nt:]
    X = lambda idx: torch.cat([F[idx], torch.ones(len(idx), 1)], dim=1)
    sol = torch.linalg.lstsq(X(tr), G[tr]).solution
    pred = X(te) @ sol
    ss_res = ((G[te] - pred) ** 2).sum(0)
    ss_tot = ((G[te] - G[te].mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot)
    return [round(v, 3) for v in r2.tolist()], round(float(r2.mean()), 3)


def main():
    x, G = make_dataset(700, seed=0)
    out = {"device": DEVICE}

    Fp = extract(feature_body("pretrained"), x)
    pv, pm = probe_r2(Fp, G)
    out["pretrained"] = {"por_var": dict(zip(VARS, pv)), "medio": pm}

    Fr = extract(feature_body("random"), x)
    rv, rm = probe_r2(Fr, G)
    out["random"] = {"por_var": dict(zip(VARS, rv)), "medio": rm}

    # curva de capacidade: projeta as features pré-treinadas para d dims (destrói info)
    torch.manual_seed(0)
    cap = {}
    for d in (1, 2, 4, 8, 16, 32):
        proj = torch.randn(Fp.shape[1], d)
        _, m = probe_r2(Fp @ proj, G)
        cap[d] = m
    out["gargalo_capacidade_R2_medio"] = cap

    print(json.dumps(out, indent=2, ensure_ascii=False))
    json.dump(out, open("/Users/savino/stage2_probe/result.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
