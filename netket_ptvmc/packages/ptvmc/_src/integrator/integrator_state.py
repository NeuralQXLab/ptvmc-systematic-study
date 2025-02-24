from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from netket.operator import DiscreteJaxOperator
from netket.utils import struct, KahanSum
from netket.vqs import VariationalState

from netket_pro import distributed


if TYPE_CHECKING:
    from ptvmc._src.integrator.integrator_state import AbstractIntegratorState


class IntegratorState(struct.Pytree):
    r"""
    Dataclass containing the state of an ODE solver.
    In particular, it stores the current state of the system, former usefull values
    and information about integration (number of step, errors, etc)
    """

    solver_state: "AbstractIntegratorState"
    """The state of the solver."""

    # compression_algorithm: "AbstractStateCompression"
    # """The compression algorithm used to compress the state."""

    step_no: int
    """The number of successful steps since the start of the iteration."""
    step_no_total: int
    """The number of steps since the start of the iteration, including rejected steps."""
    t: KahanSum
    """The current time."""
    y: VariationalState
    """The solution at current time."""
    dt: float
    """The current time-step size."""
    # last_norm: float | None = None
    # """The solution norm at previous time step."""
    # last_scaled_error: float | None = None
    # """The error of the TDVP integrator at the last time step."""
    # flags: IntegratorFlags = IntegratorFlags.INFO_STEP_ACCEPTED
    # """The flags containing information on the solver state."""

    def __init__(
        self,
        dt: float,
        y,
        t,
        solver,
        *,
        step_no=0,
        step_no_total=0,
        generator: DiscreteJaxOperator = None,
    ):
        r"""
        Initialize the state of the Integrator.

        Args:
            dt: The current time-step size.
            y: The solution at current time.
            t: The current time.
            solver: The ODE solver.

            step_no: The number of successful steps since the start of the iteration.
                defaults to :code:`0`.
            step_no_total: The number of steps since the start of the iteration, including rejected steps.
                defaults to :code:`0`.
            last_norm: The solution norm at previous time step.
            last_scaled_error: The error of the TDVP integrator at the last time step.
            flags: The flags containing information on the solver state.
        """
        step_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32

        self.step_no = distributed.declare_replicated_array(
            jnp.asarray(step_no, dtype=step_dtype)
        )
        self.step_no_total = distributed.declare_replicated_array(
            jnp.asarray(step_no_total, dtype=step_dtype)
        )

        if not isinstance(t, jax.Array):
            t = jnp.asarray(t)
        self.t = distributed.declare_replicated_array(t)
        if not isinstance(dt, jax.Array):
            dt = jnp.asarray(dt)
        self.dt = distributed.declare_replicated_array(dt)

        self.y = y

        self.solver_state = solver.init_state(generator, y, t, dt)
