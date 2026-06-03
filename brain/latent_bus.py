import threading
import time


class LWWRegister:
    """Buffer de 1 slot, last-write-wins, thread-safe. Guarda z atual, z anterior
    (para o blend) e um contador de geracao."""

    def __init__(self):
        self._lock = threading.Lock()
        self._z = None
        self._z_prev = None
        self._gen = 0

    def publish(self, z):
        with self._lock:
            self._z_prev = self._z if self._z is not None else z
            self._z = z
            self._gen += 1

    def read(self):
        with self._lock:
            return self._z, self._z_prev, self._gen


class CortexThread(threading.Thread):
    """Roda o cortex lento em background: le a scene atual, computa o latente e
    publica no registrador. Nunca bloqueia o loop rapido."""

    def __init__(self, get_scene, cortex_fn, register, cooldown=0.0):
        super().__init__(daemon=True)
        self.get_scene = get_scene
        self.cortex_fn = cortex_fn
        self.register = register
        self.cooldown = cooldown
        self._stop_event = threading.Event()   # nao usar _stop: colide com metodo interno de Thread

    def run(self):
        while not self._stop_event.is_set():
            scene = self.get_scene()
            z = self.cortex_fn(scene)
            self.register.publish(z)
            if self.cooldown:
                time.sleep(self.cooldown)

    def stop(self):
        self._stop_event.set()
