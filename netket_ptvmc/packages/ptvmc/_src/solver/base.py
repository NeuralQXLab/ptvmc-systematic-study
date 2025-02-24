from typing import Union, Callable

from netket.utils import struct
from netket.operator import AbstractOperator
from netket.vqs import VariationalState

from ptvmc._src.compression.abstract_compression import AbstractStateCompression


class AbstractDiscretizationState(struct.Pytree):
    """
    Base class holding the state of a discretization.
    """

    stage: int = struct.field(default=0)

    def __repr__(self):
        return f"{type(self).__name__}(stage={self.stage})"


StepReturnT = tuple[VariationalState, AbstractDiscretizationState, list[dict]]


class AbstractDiscretization(struct.Pytree):
    r"""
    Abstract base class for PTVMC solvers.

    The solvers implement the following methods:

    - :meth:`~ptvmc._src.solver.base.AbstractDiscretization.get_substep`
    - :meth:`~ptvmc._src.solver.base.AbstractDiscretization.finish_substep`
    - :meth:`~ptvmc._src.solver.base.AbstractDiscretization.finish_step`
    - :meth:`~ptvmc._src.solver.base.AbstractDiscretization.reset`

    Which are called inside of a discretization step as follows:

    .. code-block:: python

        # Loop until we reach the total number of steps. The reason we use a while
        # is to be able to step individually
        while True:
            # Get substep should prepare what is needed to do a compression.
            _vstate, _tstate, _U, _V = solver.get_substep(
                self.generator, actual_dt, state.t, current_vstate, solver_state
            )

            compression_result, compression_state, info = (
                compression_algorithm.init_and_execute(
                    vstate=_vstate,
                    tstate=_tstate,
                    U=_U,
                    V=_V,
                )
            )

            infos = history.accum_histories_in_tree(
                infos, info, step=solver_state.stage
            )

            # This can be used to postprocesss the output of a compression step
            current_vstate, solver_state = self.solver.finish_substep(
                self.generator, actual_dt, state.t, compression_result, solver_state
            )

            if solver_state.stage == self.solver.stages:
                # this can be used to add some final postprocessing after all steps are done
                current_vstate = self.solver.finish_step(
                    self.generator, actual_dt, state.t, compression_result, solver_state
                )
                break

        # this should at least reset the solver state
        solver_state = self.solver.reset(solver_state)


    """

    def init_state(
        self,
        generator: AbstractOperator,
        state_t0: VariationalState,
        t0: float,
        dt: float,
    ) -> AbstractDiscretizationState:
        r"""
        Initializes the `DiscretizationState` structure containing supplementary information needed.

        Returns:
            An intialized `DiscretizationState` instance
        """
        return AbstractDiscretizationState()

    def step(
        self,
        generator: Union[Callable, AbstractOperator],
        dt: float,
        t: float,
        vstate: VariationalState,
        discretization_state: AbstractDiscretizationState,
        compression_algorithm: AbstractStateCompression,
    ) -> StepReturnT:
        r"""
        This method performs a single time step. We use it to define the integration
        scheme, consisting of nested calls to the compression algorithm. Each call to
        the compression algorithm differens by the choice of the transformations `U` and `V`,
        and possible other parameters, e.g. the variational state.

        The compression algorithms takes as inputs:
        - `vstate`: the current variational state
        - `tstate`: the target state
        - `V`: the transformation V acting on `vstate`
        - `U`: the transformation U acting on `tstate`
        - `compression_algorithm`: the compression algorithm with its parameters

        This method takes in input `vstate`.
        It should create a suitable `tstate`.
        It should use `generator`, `dt`, and `t`, to compute `U` and `V` and then call the compression algorithm.

        Args:
            vstate: te current variational state
            generator: the generator of the evolution
            dt: the time step
            t: the current time
            discretization_state: the state of the solver
            compression_algorithm: the compression algorithm

        Returns:
            The new variational state and the new solver state
        """
        raise NotImplementedError()  # pragma: no cover

    @property
    def stages(self) -> int:
        """
        Number of compressions.
        """
        raise NotImplementedError

    def reset(
        self,
        solver_state: AbstractDiscretizationState,
    ) -> AbstractDiscretizationState:
        """
        Resets the solver_state to start a step from scratch.

        Should be used at the end of a step, so after all substeps have been done.

        Args:
            solver_state: the current state of the solver

        Returns:
            A solver state ready to start a new step.
        """
        raise NotImplementedError

    def start_step(
        self,
        generator: AbstractOperator,
        dt: float,
        t: float,
        vstate: VariationalState,
        solver_state: AbstractDiscretizationState,
    ) -> tuple[VariationalState, AbstractDiscretizationState]:
        """
        Initial modifications to the solver state before all substeps are computed.

        This method is called before the first substep is computed.
        """
        return vstate, solver_state

    def get_substep(
        self,
        generator: AbstractOperator,
        dt: float,
        t: float,
        y_t: VariationalState,
        solver_state: AbstractDiscretizationState,
    ) -> tuple[VariationalState, VariationalState, AbstractOperator, AbstractOperator]:
        """
        Returns the arguments (psi, phi, U, V) needed to construct a compression for
        the current substep.
        """
        raise NotImplementedError

    def finish_substep(
        self,
        generator: AbstractOperator,
        dt: float,
        t: float,
        vstate: VariationalState,
        solver_state: AbstractDiscretizationState,
    ) -> tuple[VariationalState, AbstractDiscretizationState]:
        """
        Updates the solver state after a substep has been completed.
        """
        return vstate, solver_state.replace(stage=solver_state.stage + 1)

    def finish_step(
        self,
        generator: AbstractOperator,
        dt: float,
        t: float,
        vstate: VariationalState,
        solver_state: AbstractDiscretizationState,
    ) -> tuple[VariationalState, AbstractDiscretizationState]:
        """
        Final modifications to the solver state after all substeps have been completed.
        """
        return vstate, solver_state
