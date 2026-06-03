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


def train_freeze_adapt_llm(H, data, epochs=200, lr=1e-3, seed=0, delay_steps=0):
    """Treina AdapterLLM + Controller sobre hidden CONGELADO H (N, H_DIM), imitando
    o oraculo. delay_steps>0 serve o latente defasado durante o treino (robustez por
    construcao, truque do Helix)."""
    from brain.models import AdapterLLM, Controller
    _seed(seed)
    adapter, controller = AdapterLLM(), Controller()
    opt = torch.optim.Adam(list(adapter.parameters()) + list(controller.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    pos, act = data["pos"], data["action"]
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        z = adapter(H)
        if delay_steps > 0:
            z = torch.roll(z, shifts=delay_steps, dims=0)
        pred = controller(pos, z)
        loss = loss_fn(pred, act)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return (adapter, controller), losses


def train_freeze_adapt_llm_dyn(H, data, epochs=200, lr=1e-3, seed=0):
    """Treina AdapterLLM + ControllerDyn (ciente de velocidade) sobre hidden CONGELADO H,
    imitando o oraculo PD da tarefa de 2a ordem."""
    from brain.models import AdapterLLM, ControllerDyn
    _seed(seed)
    adapter, controller = AdapterLLM(), ControllerDyn()
    opt = torch.optim.Adam(list(adapter.parameters()) + list(controller.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    pos, vel, act = data["pos"], data["vel"], data["action"]
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        z = adapter(H)
        pred = controller(pos, vel, z)
        loss = loss_fn(pred, act)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return (adapter, controller), losses
