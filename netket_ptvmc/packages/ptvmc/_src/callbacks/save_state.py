from netket.utils import struct

import netket_pro.distributed as nkpd
from advanced_drivers._src.callbacks.base import AbstractCallback

import os


class SaveState(AbstractCallback, mutable=True):
    _path: str = struct.field(pytree_node=False, serialize=False)
    _prefix: str = struct.field(pytree_node=False, serialize=False)
    _save_every: int = struct.field(pytree_node=False, serialize=False)

    def __init__(self, path: str, save_every: int, prefix: str = ""):
        self._path = path
        self._prefix = prefix
        self._save_every = save_every

    def on_run_start(self, step, driver, callbacks):
        if nkpd.is_master_process() and not os.path.exists(self._path):
            os.makedirs(self._path)

        path = f"{self._path}{self._prefix}_{driver.t:.5f}.nk"
        driver.state.save(path)

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            path = f"{self._path}{self._prefix}_{driver.t:.5f}.nk"
            driver.state.save(path)
