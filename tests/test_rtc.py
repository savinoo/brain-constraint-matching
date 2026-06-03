import numpy as np
from brain.rtc import rtc_blend


def test_freeze_first_d_then_decay():
    z_old = np.zeros(4, dtype=np.float32)
    z_new = np.ones(4, dtype=np.float32)
    assert np.allclose(rtc_blend(z_old, z_new, i=0, d=3, tau=2.0), z_old)
    assert np.allclose(rtc_blend(z_old, z_new, i=2, d=3, tau=2.0), z_old)
    out_d = rtc_blend(z_old, z_new, i=4, d=3, tau=2.0)
    assert (out_d > z_old).all() and (out_d < z_new).all()
    out_far = rtc_blend(z_old, z_new, i=50, d=3, tau=2.0)
    assert np.allclose(out_far, z_new, atol=1e-2)


def test_d_zero_is_immediate_transition():
    z_old = np.zeros(4, dtype=np.float32)
    z_new = np.ones(4, dtype=np.float32)
    out = rtc_blend(z_old, z_new, i=1, d=0, tau=1.0)
    assert (out > z_old).all()
