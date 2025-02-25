from typing import Optional

import jax.numpy as jnp

from netket.operator import AbstractOperator
from netket.vqs import VariationalState
from netket.utils.numbers import dtype as _dtype
from netket.utils import struct, history
from netket.utils.summation import KahanSum

from netket_pro._src.operator.jax_utils import to_jax_operator

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

    def __init__(
        self,
        generator: AbstractOperator,
        solver: AbstractDiscretization,
        compression_algorithm: AbstractStateCompression,
        t0: float,
        y0: VariationalState,
        parameters: IntegrationParameters,
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
        self.generator = to_jax_operator(generator)

        self._solver = solver
        self._parameters = parameters
        self._compression_algorithm = compression_algorithm

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
            # Useful to precompute some things later on...
            generator=self.generator,
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
        self._state, info = self._step_fixed(max_dt=max_dt, callback=callback)

        return True, info

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
        vstate = state.y
        solver_state = state.solver_state

        if max_dt is None:
            actual_dt = state.dt
        else:
            actual_dt = jnp.minimum(self.state.dt, max_dt)

        # briefing of the loop on substeps
        infos = None
        vstate, solver_state = self.solver.start_step(
            self.generator, actual_dt, state.t, vstate, solver_state
        )

        # loop on substeps
        while True:
            _vstate, _tstate, _U, _V, solver_state = self.solver.get_substep(
                self.generator, actual_dt, state.t, vstate, solver_state
            )
            vstate, compression_state, info = (
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
            vstate, solver_state = self.solver.finish_substep(
                self.generator, actual_dt, state.t, vstate, solver_state
            )
            if solver_state.stage == self.solver.stages:
                break

        # debriefing of the loop on substeps
        vstate, solver_state = self.solver.finish_step(
            self.generator, actual_dt, state.t, vstate, solver_state
        )
        solver_state = self.solver.reset(solver_state)

        return (
            state.replace(
                step_no=state.step_no + 1,
                step_no_total=state.step_no_total + 1,
                t=state.t + actual_dt,
                y=vstate,
                solver_state=solver_state,
            ),
            infos,
        )
