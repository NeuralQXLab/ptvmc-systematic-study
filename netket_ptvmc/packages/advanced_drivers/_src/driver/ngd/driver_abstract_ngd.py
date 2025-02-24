from typing import Any, Callable, Optional
from abc import abstractmethod

import jax
from jax.flatten_util import ravel_pytree

from netket import jax as nkjax
from netket.stats import statistics
from netket.optimizer.solver import cholesky
from netket.vqs import MCState, FullSumState
from netket.utils import timing, struct
from netket.utils.types import ScalarOrSchedule, Optimizer, Array, PyTree
from netket.jax._jacobian.default_mode import JacobianMode

from advanced_drivers._src.driver.abstract_variational_driver import (
    AbstractVariationalDriver,
)
from advanced_drivers._src.driver.ngd.sr_srt_common import sr, srt
from advanced_drivers._src.driver.ngd.srt_onthefly import srt_onthefly


ApplyFun = Callable[[PyTree, Array], Array]
KernelArgs = tuple[ApplyFun, PyTree, Array, tuple[Any, ...]]
KernelFun = Callable[[PyTree, Array], KernelArgs]
DeriativesArgs = tuple[ApplyFun, PyTree, PyTree, Array]


@jax.jit
def _flatten_samples(x):
    # return x.reshape(-1, x.shape[-1])
    return jax.lax.collapse(x, 0, x.ndim - 1)


class AbstractNGDDriver(AbstractVariationalDriver):
    r"""
    Abstract class for Natural Gradient Descent (NGD) drivers. This class is not meant to be used
    directly, but to be subclassed by specific NGD drivers. It provides the basic structure for
    the optimization loop, and the interface to the NGD solvers.

    The main method to be implemented by subclasses is `_get_local_estimators`, which should return
    the `local_gradient` and `local_loss` estimators.

    The class supports both the standard formulation of NGD and the kernel formulation. The two formulations
    compute the parameter updates as follows:

    - The standard formulation computes the updates as:

    .. math::
        \delta \theta = \tau (X^TX + \lambda \mathbb{I}_{N_P})^{-1} X^T f,

    where :math:`X \in R^{N_s \times N_p}` is the Jacobian of the log-wavefunction, with :math:`N_p` the number of parameters
    and :math:`N_s` the number of samples. The vector :math:`f` is the centered local estimator, corresponding, in the code, to ``local_gradient``.

    - The kernel formulation computes the updates as:

    .. math::
        \delta \theta = \tau X^T(XX^T + \lambda \mathbb{I}_{2N_s})^{-1} f,

    The matrix inversion is performed using a linear solver, which can be specified by the user.
    The regularization parameter :math:`\lambda` is the `diag_shift` parameter.
    The updates are then applied to the parameters using the `optimizer` which in general should be `optax.sgd`.

    See `R.Rende, L.L.Viteritti, L.Bardone, F.Becca and S.Goldt <https://arxiv.org/abs/2310.05715>`_
    for a detailed description of the derivation. A similar result can be obtained by minimizing the
    Fubini-Study distance with a specific constrain, see `A.Chen and M.Heyl <https://arxiv.org/abs/2302.01941>`_
    for details.

    When `momentum` is used, this driver implements the SPRING optimizer in
    `G.Goldshlager, N.Abrahamsen and L.Lin <https://arxiv.org/abs/2401.10190>`_
    to accumulate previous updates for better approximation of the exact SR with
    no significant performance penalty.
    """

    # Settings set by user
    diag_shift: ScalarOrSchedule = struct.field(serialize=False)
    r"""
    The diagonal shift :math:`\lambda` in the curvature matrix.

    This can be a scalar or a schedule. If it is a schedule, it should be
    a function that takes the current step as input and returns the value of the shift.
    """
    proj_reg: ScalarOrSchedule = struct.field(serialize=False)

    momentum: bool = struct.field(serialize=False, default=False)
    r"""
    Flag specifying whether to use momentum in the optimisation.

    If `True`, the optimizer will use momentum to accumulate previous updates
    following the SPRING optimizer from
    `G.Goldshlager, N.Abrahamsen and L.Lin <https://arxiv.org/abs/2401.10190>`_
    to accumulate previous updates for better approximation of the exact SR with
    no significant performance penalty.
    """

    _mode: str = struct.field(serialize=False)
    _chunk_size_bwd: Optional[int] = struct.field(serialize=False)
    _use_ntk: bool = struct.field(serialize=False)
    _on_the_fly: bool = struct.field(serialize=False)
    _collect_quadratic_model: bool = struct.field(serialize=False)
    _linear_solver_fn: Any = struct.field(serialize=False)

    # Internal things cached
    _unravel_params_fn: Any = struct.field(serialize=False)

    # Serialized state
    _old_updates: PyTree = None
    _dp: PyTree = struct.field(serialize=False)
    info: Optional[Any] = None
    """
    PyTree to pass on information from the solver,e.g, the quadratic model.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        diag_shift: ScalarOrSchedule,
        proj_reg: Optional[ScalarOrSchedule] = None,
        momentum: Optional[ScalarOrSchedule] = None,
        linear_solver_fn: Callable[[Array, Array], Array] = cholesky,
        variational_state: MCState = None,
        chunk_size_bwd: Optional[int] = None,
        collect_quadratic_model: bool = False,
        mode: Optional[JacobianMode] = None,
        use_ntk: bool = False,
        on_the_fly: bool | None = None,
        minimized_quantity_name: str = "Loss",
    ):
        r"""
        Initialize the driver.

        Args:
            optimizer: The optimizer to use for the parameter updates.
            diag_shift: The regularization parameter :math:`\lambda` for the NGD solver.
            proj_reg: The regularization parameter for the projection of the updates.
            momentum: The momentum parameter for the optimizer.
            linear_solver_fn: The linear solver function to use for the NGD solver.
            mode: The mode used to compute the jacobian of the variational state.
                Can be `'real'` or `'complex'`.
            on_the_fly: Whether to compute the QGT or NTK using lazy evaluation methods.
                This usually requires less memory.
            variational_state: The variational state to optimize.
            chunk_size_bwd: The number of rows of the NTK evaluated in a single sweep.
            collect_quadratic_model: Whether to collect the quadratic model of the loss.
            use_ntk: Wheter to compute the updates using the Neural Tangent Kernel (NTK)
            instead of the Quantum Geometric Tensor (QGT).
            minimized_quantity_name: The name of the minimized quantity.
        """
        super().__init__(
            variational_state,
            optimizer,
            minimized_quantity_name=minimized_quantity_name,
        )
        if isinstance(variational_state, FullSumState):
            raise TypeError(
                "NGD drivers do not support FullSumState. Please use 'standard' drivers with SR."
            )

        if on_the_fly is None:
            if use_ntk:
                on_the_fly = True
            else:
                on_the_fly = False

        self.diag_shift = diag_shift
        self.proj_reg = proj_reg
        self.momentum = momentum

        self.chunk_size_bwd = chunk_size_bwd
        self._use_ntk = use_ntk
        self.mode = mode
        self.on_the_fly = on_the_fly

        self._collect_quadratic_model = collect_quadratic_model
        self._linear_solver_fn = linear_solver_fn

        _, unravel_params_fn = ravel_pytree(self.state.parameters)
        self._unravel_params_fn = jax.jit(unravel_params_fn)

        self._old_updates: PyTree = None
        self._dp: PyTree = None

        # PyTree to pass on information from the solver, e.g, the quadratic model
        self.info = None

        params_structure = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state.parameters
        )
        if not nkjax.tree_ishomogeneous(params_structure):
            raise ValueError(
                "SRt only supports neural networks with all real or all complex parameters. "
                "Hybrid structures are not yet supported (but we would welcome contributions. Get in touch with us!)"
            )

    def _get_local_estimators_kernel_args(self) -> KernelArgs:
        r"""
        Get the arguments to pass to the local estimator kernel.

        This will be called by the method `local_estimators` to compute the local estimators following the schema below.

        .. code-block:: python

            afun, variables, σ, extra_args = self._get_local_estimators_kernel_args()
            local_grad, local_loss = self._kernel(afun, variables, σ, *extra_args)

        .. note::

            This method should be implemented by subclasses.

        Returns:
            A Tuple with 4 elements: afun, vars, σ, extra_args.
            The first three elements are the log-wavefunction, the variable and the samples of the variational state.
            The second to last element is any argument that can be fed as input to the local_kernel.
        """
        raise NotImplementedError()  # pragma: no cover

    @timing.timed
    def local_estimators(
        self, samples: Optional[Array] = None, parameters: Optional[PyTree] = None
    ) -> tuple[Array, Array]:
        r"""
        Compute the per-sample (local) estimators for the gradient and loss. This method should be implemented by subclasses.

        The two match when minimising the energy, but can differ, as is the case of some Infidelity estimators.

        .. note::

            This method uses the log-wavefunction, the variables and the samples obtained by calling the method
            `_get_local_estimators_kernel_args`. This method should be implemented by subclasses.

        Args:
            samples: The samples to use for the computation of the local estimators. If `None`, the current samples are used.
            parameters: Override the paramteers to compute the local estimators. If `None`, the current parameters are used.

        Returns:
                The method should return 2 jax arrays of shape `(Ns,)`:  `local_grad` and `local_loss`.
                `local_grad` should be such that the mean of its dot product with the jacobian equals the expectation value of the gradient of the loss.
                `local_loss` should be such that its mean equals the expectation value of the loss.
        """
        afun, variables, σ, extra_args = self._get_local_estimators_kernel_args()

        if parameters is not None:
            variables["params"] = parameters
        if samples is not None:
            σ = samples

        local_grad, local_loss = self._kernel(
            afun,
            variables,
            σ,
            *extra_args,
        )
        return local_grad, local_loss

    @property
    @abstractmethod
    def _kernel(self) -> KernelFun:
        r"""
        The kernel function to compute the local estimators.

        .. note::

            This method should be implemented by subclasses.

        This resulting kernel function should be called as follows:

        .. code-block:: python

            afun, variables, σ, extra_args = self._get_local_estimators_kernel_args()
            local_grad, local_loss = self._kernel(afun, variables, σ, *extra_args)

        Returns:
            A function that takes the log-wavefunction, the variables, the samples and any extra argument,
            and returns the local gradient and the local loss.
        """
        raise NotImplementedError()

    @timing.timed
    def _prepare_derivatives(self) -> DeriativesArgs:
        r"""
        Prepare the function and the samples for the computation of the jacobian, the neural tangent kernel,
        the vjp or jvp.

        This method difers from `_get_local_estimators_kernel_args` in that it is not used to compute the local
        estimators, but to compute the NGD update. In general, this should be the same function as the ones
        used to compute the local estimator of the gradient, but not necessarily.

        .. note::

            This method should be implemented by subclasses.

        Returns:
            A tuple containing the function, the parameters, the model state and the samples to be fed to the jacobian, the neural tangent kernel, the vjp or jvp.
        """
        raise NotImplementedError()

    @timing.timed
    def compute_loss_and_update(self):
        afun, params, model_state, samples = self._prepare_derivatives()
        local_grad, local_loss = self.local_estimators()

        self._loss_stats = statistics(local_loss)

        diag_shift = self.diag_shift
        proj_reg = self.proj_reg
        momentum = self.momentum
        if callable(diag_shift):
            diag_shift = diag_shift(self.step_count)
        if callable(proj_reg):
            proj_reg = proj_reg(self.step_count)
        if callable(momentum):
            momentum = momentum(self.step_count)

        self._dp, self._old_updates, self.info = self.update_fn(
            afun,
            local_grad,
            params,
            model_state,
            samples,
            diag_shift=diag_shift,
            solver_fn=self._linear_solver_fn,
            mode=self.mode,
            proj_reg=proj_reg,
            momentum=momentum,
            old_updates=self._old_updates,
            chunk_size=self.chunk_size_bwd,
            collect_quadratic_model=self.collect_quadratic_model,
        )

        return self._loss_stats, self._dp

    @timing.timed
    def _log_additional_data(self, log_dict: dict):
        """
        Method to be implemented in sub-classes of AbstractVariationalDriver to
        log additional data at every step.
        This method is called at every iteration when executing with `run`.

        Args:
            log_dict: The dictionary containing all logged data. It must be
                **modified in-place** adding new keys.
            step: the current step number.
        """
        # Always log the acceptance.
        if hasattr(self.state, "sampler_state"):
            acceptance = getattr(self.state.sampler_state, "acceptance", None)
            if acceptance is not None:
                log_dict["acceptance"] = acceptance

        # Log the quadratic model if requested.
        if self.info is not None:
            log_dict["info"] = self.info

    @property
    def mode(self) -> JacobianMode:
        """
        The mode used to compute the jacobian of the variational state. Can be `'real'`, `'complex'`, or 'onthefly'.

        - `'real'` mode truncates imaginary part of the wavefunction, useful for real-valued wf with a sign.
        - `'complex'` is the general implementation that always works.
        - `onthefly` uses a lazy implementation of the neural tangent kernel and does not compute the jacobian.

        This internally uses :func:`netket.jax.jacobian`. See that function for a more complete documentation.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str | JacobianMode | None):
        # TODO: Add support for 'onthefly' mode
        # At the moment, onthefly is only supported for use_ntk=True.
        # We raise a warning if the user tries to use it with use_ntk=False.
        if mode is None:
            mode = nkjax.jacobian_default_mode(
                self.state._apply_fun,
                self.state.parameters,
                self.state.model_state,
                self.state.hilbert.random_state(jax.random.key(1), 3),
                warn=False,
            )

        # TODO: Add support for 'holomorphic' mode
        # At the moment we only support 'real' and 'complex' modes for jacobian.
        # We raise an error if the user tries to use 'holomorphic' mode.
        if mode not in ["complex", "real"]:
            raise ValueError(
                "`mode` only supports 'jacobian_real' for real-valued wavefunctions, and 'jacobian_complex' for complex valued wave functions."
                "`holomorphic` is not yet supported, but could be contributed in the future. \n"
                f"You gave {mode}"
            )

        self._mode = mode

    @property
    def on_the_fly(self) -> bool:
        """
        Whether
        """
        return self._on_the_fly

    @on_the_fly.setter
    def on_the_fly(self, value: bool):
        if value and not self.use_ntk:
            raise ValueError(
                """
                `onthefly` is only supported when `use_ntk=True`.
                We plan to support this mode for the standard NGD in the future.
                In the meantime, use a standard VMC+SR QGTOnTheFly.
                """
            )
        self._on_the_fly = value

    @property
    def use_ntk(self) -> bool:
        r"""
        Whether to use the Neural Tangent Kernel (NTK) instead of the Quantum Geometric Tensor (QGT) to compute the update.
        """
        return self._use_ntk

    @property
    def update_fn(self) -> Callable:
        """Returns the function to compute the NGD update based on the evaluation mode."""
        if self.use_ntk:
            if self.on_the_fly:
                return srt_onthefly
            else:
                return srt
        else:
            if self.on_the_fly:
                raise NotImplementedError
            else:
                return sr

    @property
    def chunk_size_bwd(self) -> int:
        """
        Chunk size for backward-mode differentiation. This reduces memory pressure at a potential cost of higher computation time.

        If computing the jacobian, the jacobian is computed in blocks of `chunk_size_bwd` rows.
        If computing the NTK lazily, this is the number of rows of NTK evaluated in a single sweep.
        The chunk size does not affect the result, up to numerical precision.
        """
        return self._chunk_size_bwd

    @chunk_size_bwd.setter
    def chunk_size_bwd(self, value: int | None):
        if not isinstance(value, int | None):
            raise TypeError("chunk_size must be an integer or None")
        self._chunk_size_bwd = value

    @property
    def collect_quadratic_model(self) -> bool:
        r"""
        Whether to collect the quantities required to compute the quadratic model of the loss.

        These quantities, `linear_term` and `qudratic_term` are collected in the `info` attribute of the driver.
        They are computed during the estimation of the parameter updates by the functions `_compute_quadratic_model_sr`
        and `_compute_quadratic_model_srt`.

        The quadratic model is used to estimate if the update is small enough for the
        quadratic approximation on which NGD is based upon remains valid after the update.
        """
        return self._collect_quadratic_model

    @collect_quadratic_model.setter
    def collect_quadratic_model(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("collect_quadratic_model must be a boolean")
        self._collect_quadratic_model = value
