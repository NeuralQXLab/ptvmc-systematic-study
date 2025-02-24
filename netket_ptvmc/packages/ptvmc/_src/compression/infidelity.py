import copy

from netket.utils.types import PyTree
from netket.vqs import VariationalState
from netket.logging import RuntimeLog
from netket.operator import AbstractOperator

from advanced_drivers._src.driver.abstract_variational_driver import (
    AbstractVariationalDriver,
)
from advanced_drivers._src.callbacks.progressbar import ProgressBarCallback

from ptvmc._src.compression.abstract_compression import (
    AbstractStateCompression,
    CompressionState,
    CompressionResultT,
)


class InfidelityCompression(AbstractStateCompression):
    r"""
    Compression algorithm based on the minimisation of the Infidelity.

    Internally uses a driver of type `driver_class` to perform the compression, constructed
    with the parameters `build_parameters` and whose execution is controlled by the parameters
    `run_parameters`.

    This is used by Schroedinger's equation solver to perform an implicit step, i.e. to compute
    the state :math:`\ket{\psi(t+\Delta t)}` from the state :math:`\ket{\psi(t)}` as

    .. math::

        \ket{\psi(t+\Delta t)} = \hat{V}^{-1}\hat{U}\ket{\psi(t)}

    where :math:`\hat{U}` and :math:`\hat{V}` are given by the particular discretization
    scheme used by the solver.

    """

    def __init__(
        self,
        driver_class: AbstractVariationalDriver,
        build_parameters: PyTree,
        run_parameters: PyTree,
    ):
        r"""
        Constructs the compression algorithm with the given parameters.

        The driver will be constructed by calling the following function

        .. code-block:: python

            driver = driver_class(
                target_state=tstate,
                variational_state=copy.copy(vstate),
                U=U,
                V=V,
                **build_parameters
                )
            # and executed with the parameters
            driver.run(n_iter, out=logger, callback=callback, show_progress=False, **run_parameters)

        A reasonable example of Infidelity Compression is given by

        .. code-block:: python

            import advanced_drivers as advd
            import ptvmc
            import optax

            compression_alg = ptvmc.compression.InfidelityCompression(
                    driver_class=advd.driver.InfidelityOptimizerNG,
                    build_parameters={
                        "diag_shift": 1e-6,
                        "optimizer": optax.inject_hyperparams(optax.sgd)(learning_rate=0.05),
                        "linear_solver_fn": cholesky,
                        "proj_reg": None,
                        "momentum": None,
                        "chunk_size_bwd": None,
                        "collect_quadratic_model": False,
                        "use_ntk": False,
                        "cv_coeff": -0.5,
                        "resample_fraction": None,
                        "estimator": "cmc",
                    },
                    run_parameters={
                        "n_iter": 50,
                        "callback": [],
                    },
                )


        Args:
            driver_class: The driver class to use for the compression. This must be a subclass
                of :class:`advanced_drivers.driver.AbstractVariationalDriver`. The driver will be
                constructed as discussed above.
            build_parameters: a dictionary of keyword arguments to pass to the driver constructor.
            run_parameters: a dictionary of keyword arguments to pass to the driver's `run` method.
        """
        self.driver_class = driver_class
        self._build_parameters = build_parameters
        self._run_parameters = run_parameters

        if "n_iter" not in run_parameters:
            raise RuntimeError("Must specify the `n_iter` run parameter.")

    @property
    def build_parameters(self) -> PyTree:
        """Get the compression parameters."""
        return self._build_parameters

    @property
    def run_parameters(self) -> PyTree:
        """Get the compression parameters."""
        return self._run_parameters

    def init_state(
        self,
        vstate: VariationalState,
        tstate: VariationalState,
        U: AbstractOperator,
        V: AbstractOperator,
    ) -> CompressionState:
        driver = self.driver_class(
            target_state=tstate,
            variational_state=copy.copy(vstate),
            U=U,
            V=V,
            **self.build_parameters,
        )

        return driver

    def execute(self, driver: CompressionState) -> CompressionResultT:
        logger = RuntimeLog()
        kwargs = self.run_parameters.copy()

        n_iter = kwargs.pop("n_iter")
        callback = kwargs.pop("callback")

        if callback is None:
            callback = []
        elif not isinstance(callback, (list, tuple)):
            callback = [callback]
        else:
            # this creates a copy
            callback = list(callback)

        progress_cb = ProgressBarCallback(n_iter, leave=False)
        callback.append(progress_cb)

        driver.run(
            n_iter,
            out=logger,
            callback=callback,
            show_progress=False,
            **kwargs,
            _graceful_keyboard_interrupt=False,
        )
        return driver.state, driver, logger.data
