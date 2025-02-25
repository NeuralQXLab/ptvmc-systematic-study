import numpy as np

from typing import Any

from tqdm import tqdm

from netket.utils import struct

from netket_pro.distributed import is_master_process

from advanced_drivers._src.callbacks.base import AbstractCallback

"""
Here we generate the ProgressBarCallback for the ptvmc driver. We do not use the ProgressBarCallback implemented in advanced_drivers because:
1- We want the interface to display the time t of the simulation, not only the step number.
2- By taking the time t, we can account for a variable time step in the simulation.
"""


class DynamicsIntegrationProgressBarCallback(AbstractCallback, mutable=True):
    """
    Progress bar for the dynamics simulation.
    """

    _leave: bool = struct.field(pytree_node=False, serialize=False)

    _final_time: float = struct.field(pytree_node=False, serialize=True)
    _initial_time: float = struct.field(pytree_node=False, serialize=True)

    _pbar: Any = struct.field(pytree_node=False, serialize=False)

    def __init__(self, initial_time: float, final_time: float, leave: bool = True):
        """
        Args:
            initial_time: The initial time of the simulation t0.
            final_time: The final time of the simulation tend.
            leave: If True, the progress bar will leave the progress bar on the screen after the simulation is done.
        """
        self._pbar = None
        self._final_time = float(final_time)
        self._initial_time = float(initial_time)
        self._leave = leave

    def on_run_start(self, step, driver, callbacks):
        self._final_time = self._final_time

        self._pbar = tqdm(
            total=np.round(self._final_time, decimals=8),
            disable=not is_master_process(),
            dynamic_ncols=True,
            leave=self._leave,
        )
        # the time to evolve
        T = self._final_time - self._initial_time

        self._pbar.n = np.round(
            min(np.asarray(driver.t) - self._initial_time, T), decimals=8
        )
        self._pbar.set_postfix_str("n=" + str(driver._step_count))
        self._pbar.unpause()
        self._pbar.refresh()

    def on_step_end(self, step, log_data, driver):
        # the time to evolve
        T = self._final_time - self._initial_time

        self._pbar.n = np.round(
            min(np.asarray(driver.t) - self._initial_time, T), decimals=8
        )
        self._pbar.set_postfix_str("n=" + str(driver._step_count + 1))
        self._pbar.refresh()

    def on_run_end(self, step, driver):
        if self._pbar is not None:
            self._pbar.n = np.round(
                min(np.asarray(driver.t), self._final_time), decimals=8
            )
            self._pbar.set_postfix_str("n=" + str(driver._step_count))
            self._pbar.refresh()
            self._pbar.close()
            self._pbar = None

    def on_run_error(self, step, error, driver):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
