import numpy as np


def rtc_blend(z_old, z_new, i, d, tau=3.0):
    """Peso do z_new no passo i apos a chegada do z_new.
    i < d  -> congela (peso 0; mantem o que ja esta em execucao).
    i >= d -> w = 1 - exp(-(i-d)/tau), transicao suave (soft-mask estilo RTC).
    """
    z_old = np.asarray(z_old, dtype=np.float32)
    z_new = np.asarray(z_new, dtype=np.float32)
    if i < d:
        w = 0.0
    else:
        w = 1.0 - np.exp(-(i - d) / max(tau, 1e-6))
    return (1.0 - w) * z_old + w * z_new
