# Toy De-Risk: Congelado+Adaptador vs Co-treino — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir um arnês de brinquedo, CPU-runnable, que mede se "córtex congelado + adaptador treinável" recupera a performance de "co-treino end-to-end" numa tarefa mínima de visão→ação, sob uma ponte de duas velocidades (córtex lento, controlador rápido).

**Architecture:** Tarefa 2D `ContextualReach` (objetivo escondido, codificado num "scene" vetorial, que troca devagar). Três condições treinadas por imitação de um oráculo (behavior cloning, determinístico, sem RL): **(B) co-treino** (córtex+controlador juntos), **(A) congelado+adaptador** (córtex pré-treinado e congelado + adaptador + controlador), **(C) monólito** (uma rede só). Avaliação em malha fechada com uma ponte de duas velocidades (córtex recomputado a cada K passos, controlador age todo passo com latente possivelmente velho). Veredito = razão de performance A/B, sobretudo quando K cresce.

**Tech Stack:** Python 3.12, PyTorch (CPU), NumPy, pytest. Sem GPU, sem gymnasium (env próprio), sem LLM. Modelos = MLPs minúsculos.

---

## File Structure

- `requirements.txt` — deps (torch CPU, numpy, pytest).
- `brain/__init__.py` — pacote.
- `brain/envs/__init__.py`
- `brain/envs/contextual_reach.py` — a tarefa de brinquedo + oráculo.
- `brain/models.py` — Cortex, CortexPretrainHead, Adapter, Controller, Monolith.
- `brain/data.py` — coleta de dataset de imitação do oráculo.
- `brain/train.py` — pré-treino do córtex + 3 funções de treino (cotrain, freeze+adapt, monólito).
- `brain/bridge.py` — avaliação em malha fechada com a ponte de duas velocidades.
- `brain/experiment.py` — roda as 3 condições, computa o veredito, imprime tabela.
- `tests/` — testes espelhando cada módulo.
- `README.md` — como rodar.

Cada arquivo tem uma responsabilidade. Constantes de dimensão vivem em `brain/models.py` e são importadas onde preciso.

---

## Task 1: Scaffold do projeto + dependências

**Files:**
- Create: `requirements.txt`, `brain/__init__.py`, `brain/envs/__init__.py`, `tests/__init__.py`, `.gitignore`

- [ ] **Step 1: Inicializar git e estrutura**

```bash
cd /home/savino/projects/brain
git init
mkdir -p brain/envs tests
touch brain/__init__.py brain/envs/__init__.py tests/__init__.py
```

- [ ] **Step 2: Criar `.gitignore`**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
.pytest_cache/
results/
```

- [ ] **Step 3: Criar `requirements.txt`**

```
torch
numpy
pytest
```

- [ ] **Step 4: Criar venv e instalar (CPU)**

Run:
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pytest
```
Expected: instalação conclui sem erro. `python -c "import torch; print(torch.__version__)"` imprime a versão.

> Se a RAM for apertada (<3 GB), a roda CPU do torch ainda instala; o experimento usa modelos minúsculos. Se falhar por memória, rodar no Mac.

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt brain tests
git commit -m "chore: scaffold projeto toy de-risk (estrutura + deps)"
```

---

## Task 2: A tarefa `ContextualReach` + oráculo

**Files:**
- Create: `brain/envs/contextual_reach.py`
- Test: `tests/test_contextual_reach.py`

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_contextual_reach.py
import numpy as np
from brain.envs.contextual_reach import ContextualReach, SCENE_DIM, GOAL_DIM, POS_DIM, ACTION_DIM

def test_reset_obs_shapes():
    env = ContextualReach(seed=0)
    obs = env.reset()
    assert obs["scene"].shape == (SCENE_DIM,)
    assert obs["pos"].shape == (POS_DIM,)
    assert obs["goal"].shape == (GOAL_DIM,)
    assert np.allclose(obs["pos"], 0.0)

def test_oracle_reduces_distance():
    env = ContextualReach(seed=1)
    obs = env.reset()
    d0 = np.linalg.norm(obs["pos"] - obs["goal"])
    for _ in range(10):
        obs, r, done = env.step(env.oracle_action())
    d1 = np.linalg.norm(obs["pos"] - obs["goal"])
    assert d1 < d0  # oráculo aproxima do objetivo

def test_goal_switches_on_schedule():
    env = ContextualReach(seed=2, switch_every=5)
    obs = env.reset()
    g_start = obs["goal"].copy()
    for _ in range(4):
        obs, r, done = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert np.allclose(obs["goal"], g_start)   # ainda não trocou
    obs, r, done = env.step(np.zeros(ACTION_DIM, dtype=np.float32))  # 5º passo
    assert not np.allclose(obs["goal"], g_start)  # trocou

def test_scene_encodes_goal_deterministically_up_to_noise():
    env = ContextualReach(seed=3, noise=0.0)
    obs = env.reset()
    # com ruído zero, scene = W @ goal exatamente
    assert np.allclose(obs["scene"], env.W @ obs["goal"], atol=1e-5)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `. .venv/bin/activate && pytest tests/test_contextual_reach.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'brain.envs.contextual_reach'`.

- [ ] **Step 3: Implementar a tarefa**

```python
# brain/envs/contextual_reach.py
import numpy as np

GOAL_DIM = 2
POS_DIM = 2
SCENE_DIM = 16
ACTION_DIM = 2

# Encoder fixo (mesmo em todas as instâncias): scene = W @ goal + ruído.
_W = np.random.default_rng(123).standard_normal((SCENE_DIM, GOAL_DIM)).astype(np.float32)


class ContextualReach:
    """Reach 2D com objetivo escondido codificado num 'scene' que troca devagar."""

    def __init__(self, seed=0, switch_every=20, episode_len=100, dt=0.1, noise=0.05):
        self.rng = np.random.default_rng(seed)
        self.W = _W
        self.switch_every = switch_every
        self.episode_len = episode_len
        self.dt = dt
        self.noise = noise
        self.reset()

    def _new_goal(self):
        return self.rng.uniform(-1.0, 1.0, size=GOAL_DIM).astype(np.float32)

    def reset(self):
        self.t = 0
        self.pos = np.zeros(POS_DIM, dtype=np.float32)
        self.goal = self._new_goal()
        return self._obs()

    def _scene(self):
        n = self.noise * self.rng.standard_normal(SCENE_DIM).astype(np.float32)
        return (self.W @ self.goal + n).astype(np.float32)

    def _obs(self):
        return {"scene": self._scene(), "pos": self.pos.copy(), "goal": self.goal.copy()}

    def oracle_action(self):
        return np.clip(self.goal - self.pos, -1.0, 1.0).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.pos = (self.pos + action * self.dt).astype(np.float32)
        self.t += 1
        if self.t % self.switch_every == 0:
            self.goal = self._new_goal()
        reward = -float(np.linalg.norm(self.pos - self.goal))
        done = self.t >= self.episode_len
        return self._obs(), reward, done
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_contextual_reach.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add brain/envs/contextual_reach.py tests/test_contextual_reach.py
git commit -m "feat: tarefa ContextualReach + oráculo"
```

---

## Task 3: Os modelos (córtex, adaptador, controlador, monólito)

**Files:**
- Create: `brain/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_models.py
import torch
from brain.models import (Cortex, CortexPretrainHead, Adapter, Controller, Monolith,
                          SCENE_DIM, POS_DIM, GOAL_DIM, Z_DIM, ACTION_DIM)

def test_forward_shapes():
    B = 7
    scene = torch.randn(B, SCENE_DIM)
    pos = torch.randn(B, POS_DIM)
    cortex = Cortex()
    z = cortex(scene)
    assert z.shape == (B, Z_DIM)
    assert CortexPretrainHead()(z).shape == (B, GOAL_DIM)
    assert Adapter()(z).shape == (B, Z_DIM)
    assert Controller()(pos, z).shape == (B, ACTION_DIM)
    assert Monolith()(scene, pos).shape == (B, ACTION_DIM)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_models.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'brain.models'`.

- [ ] **Step 3: Implementar os modelos**

```python
# brain/models.py
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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_models.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add brain/models.py tests/test_models.py
git commit -m "feat: modelos (cortex, adapter, controller, monolith)"
```

---

## Task 4: Coleta do dataset de imitação do oráculo

**Files:**
- Create: `brain/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_data.py
from brain.data import collect_oracle_dataset
from brain.models import SCENE_DIM, POS_DIM, GOAL_DIM, ACTION_DIM

def test_dataset_shapes():
    data = collect_oracle_dataset(n_episodes=5, seed=0)
    n = data["scene"].shape[0]
    assert n == 5 * 100  # episode_len padrão = 100
    assert data["scene"].shape == (n, SCENE_DIM)
    assert data["pos"].shape == (n, POS_DIM)
    assert data["goal"].shape == (n, GOAL_DIM)
    assert data["action"].shape == (n, ACTION_DIM)
    assert data["scene"].dtype.is_floating_point
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_data.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'brain.data'`.

- [ ] **Step 3: Implementar a coleta**

```python
# brain/data.py
import numpy as np
import torch
from brain.envs.contextual_reach import ContextualReach


def collect_oracle_dataset(n_episodes=200, seed=0):
    """Rola o oráculo e coleta (scene, pos, goal, ação-do-oráculo) para imitação."""
    env = ContextualReach(seed=seed)
    scenes, poss, goals, acts = [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = env.oracle_action()
            scenes.append(obs["scene"]); poss.append(obs["pos"])
            goals.append(obs["goal"]); acts.append(a)
            obs, _, done = env.step(a)
    to_t = lambda xs: torch.tensor(np.array(xs, dtype=np.float32))
    return {"scene": to_t(scenes), "pos": to_t(poss), "goal": to_t(goals), "action": to_t(acts)}
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_data.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add brain/data.py tests/test_data.py
git commit -m "feat: coleta de dataset de imitação do oráculo"
```

---

## Task 5: Pré-treino do córtex + as 3 funções de treino

**Files:**
- Create: `brain/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_train.py
import torch
from brain.data import collect_oracle_dataset
from brain.train import pretrain_cortex, train_cotrain, train_freeze_adapt, train_monolith

def _data():
    return collect_oracle_dataset(n_episodes=20, seed=0)

def test_pretrain_cortex_lowers_goal_error():
    data = _data()
    cortex, losses = pretrain_cortex(data, epochs=50, seed=0)
    assert losses[-1] < losses[0]            # erro de objetivo caiu
    for p in cortex.parameters():
        assert p.requires_grad is False      # córtex sai congelado

def test_cotrain_lowers_bc_loss():
    data = _data()
    (cortex, controller), losses = train_cotrain(data, epochs=50, seed=0)
    assert losses[-1] < losses[0]

def test_freeze_adapt_lowers_bc_loss_and_keeps_cortex_frozen():
    data = _data()
    cortex, _ = pretrain_cortex(data, epochs=50, seed=0)
    before = [p.detach().clone() for p in cortex.parameters()]
    (cortex2, adapter, controller), losses = train_freeze_adapt(data, cortex, epochs=50, seed=0)
    assert losses[-1] < losses[0]
    # córtex não mudou (congelado)
    for p0, p1 in zip(before, cortex2.parameters()):
        assert torch.allclose(p0, p1)

def test_monolith_lowers_bc_loss():
    data = _data()
    monolith, losses = train_monolith(data, epochs=50, seed=0)
    assert losses[-1] < losses[0]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_train.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'brain.train'`.

- [ ] **Step 3: Implementar treino**

```python
# brain/train.py
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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_train.py -v`
Expected: 4 passed (roda em segundos em CPU).

- [ ] **Step 5: Commit**

```bash
git add brain/train.py tests/test_train.py
git commit -m "feat: pré-treino do córtex + 3 condições de treino"
```

---

## Task 6: A ponte de duas velocidades + avaliação em malha fechada

**Files:**
- Create: `brain/bridge.py`
- Test: `tests/test_bridge.py`

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_bridge.py
import numpy as np
from brain.envs.contextual_reach import ContextualReach
from brain.bridge import eval_policy

def test_oracle_policy_gets_high_success_sync():
    # Política = oráculo (usa pos e o goal verdadeiro via "z" = goal). Sucesso alto com K=1.
    cortex_fn = lambda scene: None  # oráculo não usa scene
    def policy_fn(pos, z, env):
        return np.clip(env.goal - pos, -1.0, 1.0)
    out = eval_policy(policy_fn, cortex_fn, K=1, n_episodes=20, seed=999, pass_env=True)
    assert out["success"] > 0.8

def test_returns_dict_with_keys():
    cortex_fn = lambda scene: np.zeros(8, dtype=np.float32)
    policy_fn = lambda pos, z: np.zeros(2, dtype=np.float32)
    out = eval_policy(policy_fn, cortex_fn, K=1, n_episodes=3, seed=0)
    assert set(out.keys()) == {"return", "success"}

def test_staleness_K_does_not_crash_and_zero_policy_fails():
    cortex_fn = lambda scene: np.zeros(8, dtype=np.float32)
    policy_fn = lambda pos, z: np.zeros(2, dtype=np.float32)  # não se move
    out = eval_policy(policy_fn, cortex_fn, K=5, n_episodes=5, seed=1)
    assert out["success"] < 0.5  # parado nunca alcança objetivo móvel
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_bridge.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'brain.bridge'`.

- [ ] **Step 3: Implementar a ponte**

```python
# brain/bridge.py
import numpy as np
from brain.envs.contextual_reach import ContextualReach

SUCCESS_DIST = 0.15  # alcançou se chegou a essa distância em algum passo do episódio


def eval_policy(policy_fn, cortex_fn, K=1, n_episodes=50, seed=999, pass_env=False):
    """Avalia em malha fechada com ponte de duas velocidades.

    cortex_fn(scene) -> z é recomputado SÓ a cada K passos (lento);
    policy_fn(pos, z[, env]) -> ação roda TODO passo (rápido) com o z possivelmente velho.
    Mede retorno médio e taxa de sucesso. K>1 simula a latência do córtex.
    """
    env = ContextualReach(seed=seed)
    total_return, successes = 0.0, 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        step = 0
        z = None
        ep_ret = 0.0
        reached = False
        while not done:
            if step % K == 0 or z is None:
                z = cortex_fn(obs["scene"])  # atualização lenta do córtex
            if pass_env:
                a = policy_fn(obs["pos"], z, env)
            else:
                a = policy_fn(obs["pos"], z)
            obs, r, done = env.step(a)
            ep_ret += r
            if -r <= SUCCESS_DIST:
                reached = True
            step += 1
        total_return += ep_ret
        successes += int(reached)
    return {"return": total_return / n_episodes, "success": successes / n_episodes}
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_bridge.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add brain/bridge.py tests/test_bridge.py
git commit -m "feat: ponte de duas velocidades + avaliação em malha fechada"
```

---

## Task 7: Runner do experimento + veredito

**Files:**
- Create: `brain/experiment.py`
- Test: `tests/test_experiment.py`

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_experiment.py
from brain.experiment import run_experiment

def test_run_experiment_smoke():
    # versão pequena pra rodar rápido no CI
    res = run_experiment(n_episodes_data=10, epochs=40, eval_episodes=20, Ks=(1, 5), seed=0)
    assert set(res.keys()) == {"cotrain", "freeze_adapt", "monolith", "verdict"}
    for cond in ("cotrain", "freeze_adapt", "monolith"):
        assert "success" in res[cond][1]   # K=1 result tem success
    # razão de recuperação está definida para cada K avaliado
    assert set(res["verdict"]["recovery_by_K"].keys()) == {1, 5}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_experiment.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'brain.experiment'`.

- [ ] **Step 3: Implementar o runner**

```python
# brain/experiment.py
import numpy as np
import torch
from brain.data import collect_oracle_dataset
from brain.train import pretrain_cortex, train_cotrain, train_freeze_adapt, train_monolith


def _np(x):
    return np.asarray(x, dtype=np.float32)


def _t1(x):
    """numpy 1D -> tensor (1, D)."""
    return torch.tensor(_np(x)).unsqueeze(0)


def run_experiment(n_episodes_data=200, epochs=200, eval_episodes=50, Ks=(1, 5, 10), seed=0):
    from brain.bridge import eval_policy

    data = collect_oracle_dataset(n_episodes=n_episodes_data, seed=seed)

    # treinar as 3 condições
    (cx_co, ctrl_co), _ = train_cotrain(data, epochs=epochs, seed=seed)
    cx_frozen, _ = pretrain_cortex(data, epochs=epochs, seed=seed)
    (_, adapter, ctrl_fa), _ = train_freeze_adapt(data, cx_frozen, epochs=epochs, seed=seed)
    monolith, _ = train_monolith(data, epochs=epochs, seed=seed)

    for m in (cx_co, ctrl_co, cx_frozen, adapter, ctrl_fa, monolith):
        m.eval()

    # funções córtex/política por condição (operam em vetores numpy 1D)
    @torch.no_grad()
    def cotrain_cortex(scene):
        return cx_co(_t1(scene)).squeeze(0).numpy()

    @torch.no_grad()
    def cotrain_policy(pos, z):
        return ctrl_co(_t1(pos), torch.tensor(_np(z)).unsqueeze(0)).squeeze(0).numpy()

    @torch.no_grad()
    def fa_cortex(scene):
        z = cx_frozen(_t1(scene))
        return adapter(z).squeeze(0).numpy()

    @torch.no_grad()
    def fa_policy(pos, z):
        return ctrl_fa(_t1(pos), torch.tensor(_np(z)).unsqueeze(0)).squeeze(0).numpy()

    # monólito: o "córtex" só repassa o scene (sofre staleness igual); a política usa scene+pos
    def mono_cortex(scene):
        return _np(scene)

    @torch.no_grad()
    def mono_policy(pos, scene_z):
        return monolith(torch.tensor(_np(scene_z)).unsqueeze(0), _t1(pos)).squeeze(0).numpy()

    conds = {
        "cotrain": (cotrain_policy, cotrain_cortex),
        "freeze_adapt": (fa_policy, fa_cortex),
        "monolith": (mono_policy, mono_cortex),
    }

    results = {}
    for name, (pol, cor) in conds.items():
        per_K = {}
        for K in Ks:
            per_K[K] = eval_policy(pol, cor, K=K, n_episodes=eval_episodes, seed=10_000 + K)
        results[name] = per_K

    # veredito: recuperação = (A - C) / (B - C) em retorno, por K. ~1.0 => congelar não custa.
    recovery = {}
    for K in Ks:
        a = results["freeze_adapt"][K]["return"]
        b = results["cotrain"][K]["return"]
        c = results["monolith"][K]["return"]
        denom = (b - c)
        recovery[K] = float((a - c) / denom) if abs(denom) > 1e-6 else float("nan")

    results["verdict"] = {
        "recovery_by_K": recovery,
        "note": "recovery≈1 => congelado+adaptador iguala co-treino; <0.5 => congelar custa caro; "
                "queda forte quando K cresce => fragilidade sob a ponte de duas velocidades.",
    }
    return results


def main():
    res = run_experiment()
    print("\n=== Retorno médio (maior = melhor) e sucesso por condição/K ===")
    for cond in ("cotrain", "freeze_adapt", "monolith"):
        for K, m in res[cond].items():
            print(f"{cond:13s} K={K:<3d} return={m['return']:8.3f}  success={m['success']:.2f}")
    print("\n=== VEREDITO — recuperação do gap (freeze_adapt vs co-treino) ===")
    for K, rec in res["verdict"]["recovery_by_K"].items():
        print(f"  K={K:<3d} recovery={rec:.2f}")
    print("  " + res["verdict"]["note"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_experiment.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add brain/experiment.py tests/test_experiment.py
git commit -m "feat: runner do experimento + veredito de recuperação do gap"
```

---

## Task 8: Rodar o experimento de verdade + README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Rodar a suíte completa**

Run: `pytest -v`
Expected: todos os testes passam.

- [ ] **Step 2: Rodar o experimento e observar o veredito**

Run: `python -m brain.experiment`
Expected: imprime a tabela de retorno/sucesso por condição e K, e as razões de recuperação. **Anotar os números** — esse é o resultado científico da primeira entrega. Leitura:
- `recovery ≈ 1.0` em todo K → congelar+adaptar iguala o co-treino: **a aposta passa no brinquedo**.
- `recovery < 0.5`, ou caindo forte com K → congelar custa caro / frágil sob a ponte: **sinal de alerta**, repensar antes de escalar.

- [ ] **Step 3: Criar `README.md`**

```markdown
# Toy de-risk: congelado+adaptador vs co-treino

Experimento mínimo (CPU) que testa a aposta central da arquitetura cérebro-fiel 3a:
um "córtex" congelado + adaptador treinável recupera a performance do co-treino
end-to-end, sob uma ponte de duas velocidades?

## Rodar
    python3 -m venv .venv && . .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install numpy pytest
    pytest -v
    python -m brain.experiment

## Ler o veredito
`recovery` por K = (freeze_adapt − monolito) / (co-treino − monolito) no retorno médio.
~1.0 => congelar não custa; <0.5 ou caindo com K => congelar é caro/frágil.

## Limites honestos
Modelos minúsculos, tarefa de brinquedo, imitação de oráculo (sem RL), sem LLM.
É um sinal DIRECIONAL barato — não um resultado sobre LLMs frontier. Se passar aqui,
escalar (V-JEPA 2 / LLM congelada / GPU) é a próxima entrega; se falhar, repensar
o congelamento antes de gastar.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README + resultado da primeira entrega (de-risk toy)"
```

---

## Self-Review (preenchido pelo autor)

**Spec coverage (§6 do design):**
- "visão JEPA-like / peças prontas" → **adiado de propósito**: este é o estágio toy (decidido com o usuário); V-JEPA 2 / LLM frontier vêm na entrega seguinte. O toy testa o MECANISMO da aposta (§6.1) com substitutos minúsculos, que é o objetivo desta fase.
- "congelado + adaptador vs co-treino vs modelo único" (§6.3) → Tasks 5–7.
- "assimetria física / ponte de duas velocidades" (§6.2/§6.3) → Task 6 (parâmetro K).
- "erro vem do sensório, nunca da LLM se autojulgando" (§10.2) → respeitado: o sinal de treino é imitação do oráculo (externo), nunca auto-julgamento.
- "instrumentar/benchmarkar desde o dia 1" (§10.2) → Task 7 (veredito quantitativo + ablação por K).
- "critério de aceite = veredito claro, falhar barato" (§6.4) → Task 8 Step 2.

**Placeholder scan:** sem TODO/TBD; todo passo tem código ou comando completo.

**Type consistency:** dims centralizadas em `brain/models.py` e importadas; `eval_policy` assina `(policy_fn, cortex_fn, K, n_episodes, seed, pass_env)` e é chamada assim no runner e nos testes; `run_experiment` retorna chaves `{cotrain, freeze_adapt, monolith, verdict}` conforme o teste.

**Nota de escopo:** esta é a entrega 1 de N. Escalar para modelos reais (V-JEPA 2, LLM congelada, GPU) será um plano separado, condicionado ao veredito desta.
```
