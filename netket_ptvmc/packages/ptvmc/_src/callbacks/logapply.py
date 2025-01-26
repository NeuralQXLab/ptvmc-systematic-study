from typing import Any, Callable
from functools import wraps

from netket.utils import struct

from advanced_drivers._src.callbacks.base import AbstractCallback


class DynamicLogApply(AbstractCallback, mutable=True):
    r"""
    Callback that applies a function to a logger at the end of each step.
    It can be used to dynamically plot the results of the simulation up to the current step.
    """

    logger: Any = struct.field(pytree_node=False, serialize=False)
    apply_fn: Callable = struct.field(pytree_node=False, serialize=False)

    def __init__(self, logger: Any, apply_fn: Callable):
        self.logger = logger
        self.apply_fn = apply_fn

    @property
    @wraps(AbstractCallback.callback_order)
    def callback_order(self) -> int:
        return 100

    def on_step_end(self, step, log_data, driver):
        self.apply_fn(self.logger)

    def on_run_end(self, step, driver):
        self.apply_fn(self.logger)
