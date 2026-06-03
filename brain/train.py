import torch
import torch.nn as nn
from brain.models import Cortex, CortexPretrainHead, Adapter, Controller, Monolith


def _seed(seed):
    torch.manual_seed(seed)


def pretrain_cortex(data, epochs=200, lr=1e-3, seed=0):
    """Pré-treina o córtex para prever o objetivo a partir do scene; depois CONGELA."""
    _seed(seed)
    cortex, head = Cortex(), CortexPretrainHead()
    opt = torch.optim.Adam(list(cortex.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    scene, goal = data["scene"], data["goal"]
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        pred = head(cortex(scene))
        loss = loss_fn(pred, goal)
        loss.backward(); opt.step()
        losses.append(loss.item())
    for p in cortex.parameters():
        p.requires_grad_(False)
    cortex.eval()
    return cortex, losses


def train_cotrain(data, epochs=200, lr=1e-3, seed=0):
    """(B) Co-treino: córtex + controlador treinados juntos, ponta a ponta."""
    _seed(seed)
    cortex, controller = Cortex(), Controller()
    opt = torch.optim.Adam(list(cortex.parameters()) + list(controller.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    scene, pos, act = data["scene"], data["pos"], data["action"]
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        pred = controller(pos, cortex(scene))
        loss = loss_fn(pred, act)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return (cortex, controller), losses


def train_freeze_adapt(data, frozen_cortex, epochs=200, lr=1e-3, seed=0):
    """(A) Congelado+adaptador: córtex congelado; treina só adaptador + controlador."""
    _seed(seed)
    adapter, controller = Adapter(), Controller()
    opt = torch.optim.Adam(list(adapter.parameters()) + list(controller.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    scene, pos, act = data["scene"], data["pos"], data["action"]
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        with torch.no_grad():
            z = frozen_cortex(scene)
        pred = controller(pos, adapter(z))
        loss = loss_fn(pred, act)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return (frozen_cortex, adapter, controller), losses


def train_monolith(data, epochs=200, lr=1e-3, seed=0):
    """(C) Monólito: uma rede só (scene, pos) -> ação."""
    _seed(seed)
    monolith = Monolith()
    opt = torch.optim.Adam(monolith.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scene, pos, act = data["scene"], data["pos"], data["action"]
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        pred = monolith(scene, pos)
        loss = loss_fn(pred, act)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return monolith, losses
