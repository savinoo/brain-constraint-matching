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
