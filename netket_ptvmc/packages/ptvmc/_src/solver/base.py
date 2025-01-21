from typing import Union, Callable

from netket.utils import struct
from netket.operator import AbstractOperator
from netket.vqs import VariationalState

from ptvmc._src.compression.abstract_compression import AbstractStateCompression


class AbstractDiscretizationState(struct.Pytree):
    """
    Base class holding the state of a discretization.
    """

    def __repr__(self):
        return "DiscretizationState()"


StepReturnT = tuple[VariationalState, AbstractDiscretizationState, list[dict]]


class AbstractDiscretization(struct.Pytree):
    r"""
    Abstract base class for PTVMC solvers.
    """

    def _init_state(self, generator, t, dt) -> AbstractDiscretizationState:
        r"""
        Initializes the `DiscretizationState` structure containing supplementary information needed.

        Returns:
            An intialized `DiscretizationState` instance
        """
        return AbstractDiscretizationState()

    # def step(self, vstate, H, dt):
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
        compression_result: VariationalState,
        solver_state: AbstractDiscretizationState,
    ) -> tuple[VariationalState, AbstractDiscretizationState]:
        """
        Updates the solver state after a substep has been completed.
        """
        raise NotImplementedError
