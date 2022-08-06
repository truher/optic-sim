import cupy as cp
import time


class MyTimer:
    def __init__(self):
        self._t0 = time.monotonic_ns()

    def tick(self, label):
        return
        cp.cuda.Device().synchronize()
        t1 = time.monotonic_ns()
        print(f"{label} {t1 -self._t0}")
        self._t0 = t1
