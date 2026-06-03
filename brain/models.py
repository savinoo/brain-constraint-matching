import torch
import torch.nn as nn

SCENE_DIM = 16
POS_DIM = 2
GOAL_DIM = 2
Z_DIM = 8
ACTION_DIM = 2
HIDDEN = 64


class Cortex(nn.Module):
    """scene -> latente z (gargalo). O 'córtex lento'."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(SCENE_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, Z_DIM))

    def forward(self, scene):
        return self.net(scene)


class CortexPretrainHead(nn.Module):
    """z -> estimativa de objetivo. Usada só no pré-treino do córtex."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(Z_DIM, GOAL_DIM)

    def forward(self, z):
        return self.head(z)


class Adapter(nn.Module):
    """z congelado -> z' adaptado. O 'tálamo-roteador' treinável."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(Z_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, Z_DIM))

    def forward(self, z):
        return self.net(z)


class Controller(nn.Module):
    """(pos, z) -> ação. O 'System 1' rápido."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(POS_DIM + Z_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, ACTION_DIM))

    def forward(self, pos, z):
        return self.net(torch.cat([pos, z], dim=-1))


class Monolith(nn.Module):
    """(scene, pos) -> ação. Baseline de rede única."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SCENE_DIM + POS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM),
        )

    def forward(self, scene, pos):
        return self.net(torch.cat([scene, pos], dim=-1))
