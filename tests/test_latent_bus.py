import time
import numpy as np
from brain.latent_bus import LWWRegister, CortexThread


def test_lww_last_write_wins_and_prev():
    r = LWWRegister()
    assert r.read() == (None, None, 0)
    r.publish(np.array([1.0]))
    z, zp, gen = r.read()
    assert z[0] == 1.0 and zp[0] == 1.0 and gen == 1
    r.publish(np.array([2.0]))
    z, zp, gen = r.read()
    assert z[0] == 2.0 and zp[0] == 1.0 and gen == 2


def test_cortex_thread_publishes_without_blocking():
    scene_holder = {"scene": np.zeros(3, dtype=np.float32)}
    def get_scene():
        return scene_holder["scene"]
    def cortex_fn(scene):
        time.sleep(0.02)
        return scene + 1.0
    reg = LWWRegister()
    th = CortexThread(get_scene, cortex_fn, reg, cooldown=0.0)
    th.start()
    time.sleep(0.1)
    th.stop(); th.join(timeout=1)
    z, _, gen = reg.read()
    assert gen >= 1 and z is not None
