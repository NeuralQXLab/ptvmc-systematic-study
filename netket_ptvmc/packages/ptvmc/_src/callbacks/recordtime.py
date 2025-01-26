from advanced_drivers._src.callbacks.base import AbstractCallback


class RecordTimeCallback(AbstractCallback, mutable=True):
    r"""
    Callback to log the time at each step of the simulation.
    """

    def on_legacy_run(self, step, log_data, driver):
        log_data["times"] = driver.t
