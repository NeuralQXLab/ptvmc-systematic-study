from typing import Optional

import jax.numpy as jnp

from netket.operator import AbstractOperator
from netket.vqs import VariationalState
from netket.utils.numbers import dtype as _dtype
from netket.utils import struct, history
from netket.utils.summation import KahanSum

# from ._solver import AbstractSolver
# from ._utils import (
#     maybe_jax_jit,
#     scaled_error,
#     propose_time_step,
#     set_flag_jax,
#     euclidean_norm,
#     maximum_norm,
# )

from netket_pro.utils import ensure_jax_operator

from ptvmc._src.compression.abstract_compression import AbstractStateCompression
from ptvmc._src.solver.base import AbstractDiscretization
from ptvmc._src.integrator.integrator_state import IntegratorState
from ptvmc._src.integrator.integration_params import IntegrationParameters


class Integrator(struct.Pytree, mutable=True):
    r"""
    Ordinary-Differential-Equation integrator.
    Given an ODE-function :math:`dy/dt = f(t, y)`, it integrates the derivatives to obtain the solution
    at the next time step :math:`y_{t+1}`.
    """

    generator: AbstractOperator = struct.field(pytree_node=False)
    """The generator of the dynamics."""

    _state: IntegratorState
    """The state of the integrator, containing informations about the solution."""
    _solver: AbstractDiscretization = struct.field(serialize=False)
    """The discretization algorithm."""
    _compression_algorithm: AbstractStateCompression = struct.field(serialize=False)
    """The compression algorithm."""
    _parameters: IntegrationParameters = struct.field(serialize=False)
    """The options of the integration."""

    # use_adaptive: bool = struct.field(pytree_node=False)
    # """Boolean indicating whether to use an adaptative scheme."""

    # norm: Callable = struct.field(pytree_node=False)
    # """The norm used to estimate the error."""

    def __init__(
        self,
        generator: AbstractOperator,
        solver: AbstractDiscretization,
        compression_algorithm: AbstractStateCompression,
        t0: float,
        y0: VariationalState,
        # use_adaptive: bool,
        parameters: IntegrationParameters,
        # norm: str | Callable = None,
    ):
        r"""
        Initializes the integrator with the given parameters.

        Args:
            generator: The generator of the dynamics.
            solver: The discretization algorithm.
            t0: The intial time.
            y0: The initial state.
            parameters: The suppleementary hyper-parameters of the integrator.
                This includes the values of :code:`dt`.
                See :code:`IntegrationParameters` for more details.
        """
        self.generator = ensure_jax_operator(generator)

        self._solver = solver
        self._parameters = parameters
        self._compression_algorithm = compression_algorithm

        # self.use_adaptive = use_adaptive

        # if norm is not None:
        #     if isinstance(norm, str):
        #         norm = norm.lower()
        #         if norm == "euclidean":
        #             norm = euclidean_norm
        #         elif norm == "maximum":
        #             norm = maximum_norm
        #         else:
        #             raise ValueError(
        #                 f"The error norm must either be 'euclidean' or 'maximum', instead got {norm}."
        #             )
        #     if not isinstance(norm, Callable):
        #         raise ValueError(
        #             f"The error norm must be a callable, instead got a {type(norm)}."
        #         )
        # else:
        #     norm = euclidean_norm
        # self.norm = norm

        self._state = self._init_state(t0, y0)

    def _init_state(self, t0: float, y0: VariationalState) -> IntegratorState:
        r"""
        Initializes the `IntegratorState` structure containing the solver and state,
        given the necessary information.

        Args:
            t0: The initial time of evolution
            y0: The solution at initial time `t0`

        Returns:
            An :code:`Integrator` instance intialized with the passed arguments
        """
        dt = self._parameters.dt

        t_dtype = jnp.result_type(_dtype(t0), _dtype(dt))

        return IntegratorState(
            dt=jnp.array(dt, dtype=t_dtype),
            y=y0,
            t=jnp.array(t0, dtype=t_dtype),
            solver=self.solver,
            # compression_algorithm=self._compression_algorithm,
            # last_norm=0.0 if self.use_adaptive else None,
            # last_scaled_error=0.0 if self.use_adaptive else None,
            # flags=IntegratorFlags(0),
        )

    def step(self, max_dt: float = None, callback=None) -> bool:
        """
        Performs one full step by :code:`min(self.dt, max_dt)`.

        Args:
            max_dt: The maximal value for the time step `dt`.
            callback: A callable function that is called after each step.

        Returns:
            A boolean indicating whether the step was successful or
            was rejected by the step controller and should be retried, and
            the `info` dictionary containing the relevant information about the step.
        """
        # if not self.use_adaptive:
        self._state, info = self._step_fixed(max_dt=max_dt, callback=callback)
        # else:
        #     self._state = self._step_adaptive(
        #         solver=self._solver,
        #         f=self.f,
        #         state=self._state,
        #         max_dt=max_dt,
        #         parameters=self._parameters,
        #         norm_fn=self.norm,
        #     )

        return True, info  # self._state.accepted

    @property
    def t(self) -> float:
        """The actual time."""
        if isinstance(self._state.t, KahanSum):
            return self._state.t.value
        else:
            return self._state.t

    @property
    def y(self) -> VariationalState:
        """The actual state."""
        return self._state.y

    @property
    def dt(self) -> float:
        """The actual time-step size."""
        return self._state.dt

    @property
    def solver(self) -> AbstractDiscretization:
        """The discretization algoritm."""
        return self._solver

    @property
    def compression_algorithm(self) -> AbstractStateCompression:
        """The compression algorithm."""
        return self._compression_algorithm

    # def _get_integrator_flags(self, intersect=IntegratorFlags.NONE) -> IntegratorFlags:
    #     r"""Returns the currently set flags of the integrator, intersected with `intersect`."""
    #     # _state.flags is turned into an int-valued DeviceArray by JAX,
    #     # so we convert it back.
    #     return IntegratorFlags(int(self._state.flags) & intersect)

    # @property
    # def errors(self) -> IntegratorFlags:
    #     r"""Returns the currently set error flags of the integrator."""
    #     return self._get_integrator_flags(IntegratorFlags.ERROR_FLAGS)

    # @property
    # def warnings(self) -> IntegratorFlags:
    #     r"""Returns the currently set warning flags of the integrator."""
    #     return self._get_integrator_flags(IntegratorFlags.WARNINGS_FLAGS)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(solver={self.solver}, state={self._state})"

    def _step_fixed(self, max_dt: Optional[float], callback=None) -> IntegratorState:
        r"""
        Performs one step of fixed size from current time.
        Args:
            solver: Instance that solves the ODE.
                The solver should contain a method :code:`step(f,dt,t,y_t,solver_state)`
            f: A callable ODE function.
                Given a time `t` and a state `y_t`, it should return the partial
                derivatives in the same format as `y_t`. The function should also accept
                supplementary arguments, such as :code:`stage`.
            state: IntegratorState containing the current state (t,y), the solver_state and stability information.
            max_dt: The maximal value for the time step `dt`.
            parameters: The integration parameters.

        Returns:
            Updated state of the integrator.
        """
        state = self._state
        current_vstate = state.y
        solver_state = state.solver_state

        if max_dt is None:
            actual_dt = state.dt
        else:
            actual_dt = jnp.minimum(self.state.dt, max_dt)

        # loop on substeps
        infos = None
        while True:
            _vstate, _tstate, _U, _V = self.solver.get_substep(
                self.generator, actual_dt, state.t, current_vstate, solver_state
            )

            compression_result, compression_state, info = (
                self.compression_algorithm.init_and_execute(
                    vstate=_vstate,
                    tstate=_tstate,
                    U=_U,
                    V=_V,
                )
            )
            infos = history.accum_histories_in_tree(
                infos, info, step=solver_state.stage
            )

            current_vstate, solver_state = self.solver.finish_substep(
                self.generator, actual_dt, state.t, compression_result, solver_state
            )
            if solver_state.stage == self.solver.stages:
                break

        solver_state = self.solver.reset(solver_state)

        return (
            state.replace(
                step_no=state.step_no + 1,
                step_no_total=state.step_no_total + 1,
                t=state.t + actual_dt,
                y=current_vstate,
                solver_state=solver_state,
                # flags=IntegratorFlags.INFO_STEP_ACCEPTED,
            ),
            infos,
        )

    # @staticmethod
    # @partial(maybe_jax_jit, static_argnames=["f", "norm_fn"])
    # def _step_adaptive(
    #     solver: AbstractSolver,
    #     f: Callable,
    #     state: IntegratorState,
    #     max_dt: Optional[float],
    #     parameters: IntegrationParameters,
    #     norm_fn: Callable,
    # ) -> IntegratorState:
    #     r"""
    #     Performs one adaptive step from current time.
    #     Args:
    #         solver: Instance that solves the ODE
    #             The solver should contain a method :code:`step_with_error(f,dt,t,y_t,solver_state)`
    #         f: A callable ODE function.
    #             Given a time `t` and a state `y_t`, it should return the partial
    #             derivatives in the same format as `y_t`. The function should also accept
    #             supplementary arguments, such as :code:`stage`.
    #         state: IntegratorState containing the current state (t,y), the solver_state and stability information.
    #         norm_fn: The function used for the norm of the error.
    #             By default, we use euclidean_norm.
    #         parameters: The integration parameters.
    #         max_dt: The maximal value for the time-step size `dt`.

    #     Returns:
    #         Updated state of the integrator
    #     """
    #     flags = IntegratorFlags(0)

    #     if max_dt is None:
    #         actual_dt = state.dt
    #     else:
    #         actual_dt = jnp.minimum(state.dt, max_dt)

    #     # Perform the solving step
    #     y_tp1, y_err, solver_state = solver.step_with_error(
    #         f, actual_dt, state.t.value, state.y, state.solver_state
    #     )

    #     scaled_err, norm_y = scaled_error(
    #         y_tp1,
    #         y_err,
    #         parameters.atol,
    #         parameters.rtol,
    #         last_norm_y=state.last_norm,
    #         norm_fn=norm_fn,
    #     )

    #     # Propose the next time step, but limited within [0.1 dt, 5 dt] and potential
    #     # global limits in dt_limits. Not used when actual_dt < state.dt (i.e., the
    #     # integrator is doing a smaller step to hit a specific stop).
    #     dt_min, dt_max = parameters.dt_limits
    #     next_dt = propose_time_step(
    #         actual_dt,
    #         scaled_err,
    #         solver.error_order,
    #         limits=(
    #             (jnp.maximum(0.1 * state.dt, dt_min) if dt_min else 0.1 * state.dt),
    #             (jnp.minimum(5.0 * state.dt, dt_max) if dt_max else 5.0 * state.dt),
    #         ),
    #     )

    #     # check if next dt is NaN
    #     flags = set_flag_jax(
    #         ~jnp.isfinite(next_dt), flags, IntegratorFlags.ERROR_INVALID_DT
    #     )

    #     # check if we are at lower bound for dt
    #     if dt_min is not None:
    #         is_at_min_dt = jnp.isclose(next_dt, dt_min)
    #         flags = set_flag_jax(is_at_min_dt, flags, IntegratorFlags.WARN_MIN_DT)
    #     else:
    #         is_at_min_dt = False
    #     if dt_max is not None:
    #         is_at_max_dt = jnp.isclose(next_dt, dt_max)
    #         flags = set_flag_jax(is_at_max_dt, flags, IntegratorFlags.WARN_MAX_DT)

    #     # accept if error is within tolerances or we are already at the minimal step
    #     accept_step = jnp.logical_or(scaled_err < 1.0, is_at_min_dt)
    #     # accept the time step iff it is accepted by all MPI processes
    #     accept_step, _ = mpi_all_jax(accept_step)

    #     return jax.lax.cond(
    #         accept_step,
    #         # step accepted
    #         lambda _: state.replace(
    #             step_no=state.step_no + 1,
    #             step_no_total=state.step_no_total + 1,
    #             y=y_tp1,
    #             t=state.t + actual_dt,
    #             dt=jax.lax.cond(
    #                 actual_dt == state.dt,
    #                 lambda _: next_dt,
    #                 lambda _: state.dt,
    #                 None,
    #             ),
    #             last_norm=norm_y.astype(state.last_norm.dtype),
    #             last_scaled_error=scaled_err.astype(state.last_scaled_error.dtype),
    #             solver_state=solver_state,
    #             flags=flags | IntegratorFlags.INFO_STEP_ACCEPTED,
    #         ),
    #         # step rejected, repeat with lower dt
    #         lambda _: state.replace(
    #             step_no_total=state.step_no_total + 1,
    #             dt=next_dt,
    #             flags=flags,
    #         ),
    #         state,
    #     )
