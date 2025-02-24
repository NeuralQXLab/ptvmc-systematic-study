from typing import Callable, Optional
from collections.abc import Iterable
import sys

import numbers

import jax.numpy as jnp

from netket_pro import distributed

from advanced_drivers._src.utils.itertools import to_iterable
from advanced_drivers._src.callbacks.callback_list import CallbackList
from advanced_drivers._src.callbacks.observables import ObservableCallback
from advanced_drivers._src.driver.abstract_variational_driver import (
    maybe_wrap_legacy_callback,
)
from advanced_drivers._src.callbacks.legacy_wrappers import (
    LegacyLoggerWrapper,
)
from advanced_drivers._src.driver.abstract_variational_driver import (
    AbstractVariationalDriver,
)

from netket.operator import AbstractOperator
from netket.operator._abstract_observable import AbstractObservable
from netket.logging import AbstractLog
from netket.utils import timing, struct
from netket.vqs import VariationalState

from ptvmc._src.solver.base import AbstractDiscretization
from ptvmc._src.compression.abstract_compression import AbstractStateCompression
from ptvmc._src.callbacks.progressbar import DynamicsIntegrationProgressBarCallback
from ptvmc._src.callbacks.recordtime import RecordTimeCallback
from ptvmc._src.callbacks.save_state import SaveState
from ptvmc._src.integrator.integrator import Integrator
from ptvmc._src.integrator.integration_params import IntegrationParameters


class PTVMCDriver(AbstractVariationalDriver):
    """
    The driver class for the p-tVMC algorithm. This class simulates the evolution of the system via a sequence of
    state compressions with varying target states. The definition of the target states and the specifics of the
    optimizations are handled by the compression, defined internally via the compression algorithm.

    See `L.Gravina, V.Savona, and F.Vicentini <https://arxiv.org/abs/2410.10720>`_
    and `A.Sinibaldi, C.Giuliani, G.Carleo, and F.Vicentini <https://arxiv.org/abs/2305.14294>`_ for a detailed description of the
    method.
    """

    _t0: float = struct.field(serialize=False)
    _integrator: AbstractDiscretization = struct.field(serialize=True)
    _compression_constructor: AbstractStateCompression = struct.field(serialize=False)
    _stop_count: int = struct.field(serialize=False)

    def __init__(
        self,
        generator: AbstractOperator,
        t0: float = 0.0,
        *,
        solver: AbstractDiscretization,
        integration_params: IntegrationParameters,
        compression_algorithm: AbstractStateCompression,
        variational_state: VariationalState,
    ):
        r"""
        Initializes the ptvmc driver.

        Args:
            generator: The generator of the dynamics.
            variational_state: The variational state.
            integrator: Solving algorithm used to perform a single update.
            t0: Initial time at the start of the time evolution.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schrödinger equation (SE), "imag" for the imaginary-time SE.
        """
        self._t0 = t0
        super().__init__(
            variational_state, optimizer=None, minimized_quantity_name="Generator"
        )

        integrator = Integrator(
            generator,
            solver,
            compression_algorithm,
            t0=t0,
            y0=variational_state,
            parameters=integration_params,
        )
        self._integrator = integrator

    @property
    def compression_algorithm(self):
        """
        The compression algorithm used to compress the state after each time step.
        """
        return self._integrator.compression_algorithm

    @property
    def integrator(self):
        """
        The underlying integrator algoritmh which informs the single time step.
        """
        return self._integrator

    # TODO: this method is here only bcause checkpoint callback calls it to ensure that
    # the driver has the correct state. This is a hack and should be removed.
    def _forward_and_backward(self):
        return None, None

    def run(
        self,
        T,
        out: Optional[Iterable[AbstractLog]] = (),
        obs: Optional[dict[str, AbstractObservable]] = None,
        *,
        show_progress: bool = True,
        save_path: Optional[str] = None,
        save_every: int = 1,
        save_prefix: str = "state",
        callback: Callable[
            [int, dict, "AbstractVariationalDriver"], bool
        ] = lambda *x: True,
        timeit: bool = False,
        obs_in_fullsum: bool = False,
    ):
        r"""
        Runs the driver responsible for performing the ptvmc evolution, updating the weights of the network stored in
        this driver and dumping values of the observables `obs` in the output `logger`.

        It is possible to control more specifically what quantities are logged, when to
        stop the optimisation, or to execute arbitrary code at every step by specifying
        one or more callbacks, which are passed as a list of functions to the keyword
        argument `callback`.

        Loggers are specified as an iterable passed to the keyword argument `out`. If only
        a string is specified, this will create by default a :class:`nk.logging.JsonLog`.
        To know about the output format check its documentation. The logger object is
        also returned at the end of this function so that you can inspect the results
        without reading the json output.

        Args:
            T: The time interval over which to evolve.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing all observables that should be computed
            show_progress: If true displays a progress bar (default=True)
            save_path: If not None, saves the state of the driver to the specified path.
            save_every: If save_path is not None, saves the state every `save_every` steps.
            timeit: If True, provide timing information.
            obs_in_fullsum: Whether to compute the observables in fullsum or not (default=False).
        """
        if not isinstance(T, numbers.Real):
            raise ValueError(
                "T, the first positional argument to `run`, must be a 'float' or 'int'."
            )

        if out is None:
            out = ()
        loggers = to_iterable(out)

        callback_list = [maybe_wrap_legacy_callback(c) for c in to_iterable(callback)]
        if obs is not None:
            callback_list.append(ObservableCallback(obs, fullsum=obs_in_fullsum))
        callback_list.extend(LegacyLoggerWrapper(log) for log in loggers)
        if show_progress:
            callback_list.append(
                DynamicsIntegrationProgressBarCallback(
                    initial_time=self.t, final_time=T
                )
            )

        callback_list.append(RecordTimeCallback())

        if save_path is not None:
            callback_list.append(SaveState(save_path, save_every, prefix=save_prefix))

        callbacks = CallbackList(callback_list)

        self._stop_run = False
        self._reject_step = False
        self._step_start = self.step_count
        self._step_end = self.step_count + T

        t_end = self.t + T
        with timing.timed_scope(force=timeit) as timer:
            try:
                callbacks.on_run_start(self.step_count, self, callbacks.callbacks)
                while self.t < t_end and not self._stop_run:
                    self._step_attempt = 0
                    step_log_data = {}

                    while True:
                        callbacks.on_step_start(self.step_count, step_log_data, self)

                        self.reset_step()

                        callbacks.on_compute_update_start(
                            self.step_count, step_log_data, self
                        )

                        _, info = self._integrator.step(max_dt=None)

                        callbacks.on_compute_update_end(
                            self.step_count, step_log_data, self
                        )

                        # Handle repeating a step
                        if self._reject_step:
                            self._reject_step = False
                            self._step_attempt += 1
                            continue
                        else:
                            break

                    # If we are here, we accepted the step
                    if info is not None:
                        step_log_data.update(info)

                    # Execute callbacks before loggers because they can append to log_data
                    self._log_additional_data(step_log_data)
                    callbacks.on_legacy_run(self.step_count, step_log_data, self)

                    callbacks.on_step_end(self.step_count, step_log_data, self)
                    self._step_count += 1

                callbacks.on_run_end(self.step_count, self)

            except KeyboardInterrupt as error:
                callbacks.on_run_error(self.step_count, error, self)
                if hasattr(sys, "ps1"):
                    print("Stopped by user.")
                else:
                    raise
            except Exception as error:
                callbacks.on_run_error(self.step_count, error, self)
                raise

        if timeit:
            self._timer = timer
            if distributed.is_master_process():
                print(timer)

        return loggers

    @property
    def _default_step_size(self):
        # Essentially means
        return None

    @property
    def step_value(self):
        return self.t

    @property
    def dt(self):
        """Current time step."""
        return self._integrator.dt

    @property
    def t(self):
        """Current time."""
        return self._integrator.t

    @t.setter
    def t(self, t):
        self._integrator.t = jnp.array(t, dtype=self._integrator.t)

    @property
    def t0(self):
        """
        The initial time set when the driver was created.
        """
        return self._t0

    @property
    def state(self):
        """
        Returns the machine that is optimized by this driver.
        """
        # self._variational_state = self._integrator.state.y
        return self._integrator._state.y  # self._variational_state

    def __repr__(self):
        return f"{type(self).__name__}(step_count={self.step_count}, t={self.t})"
