from copy import copy
from typing import Optional

from netket.operator import AbstractOperator
from netket.utils import struct
from netket.vqs import VariationalState

from netket_pro.utils.split_hamiltonian import split_hamiltonian as split_generator

from ptvmc._src.solver.base import (
    AbstractDiscretization,
    AbstractDiscretizationState,
)


@struct.dataclass
class ProductExpansionState(AbstractDiscretizationState):
    r"""
    State of the AbstractDiscretization.
    `stage` is an integer that keeps track of the current stage of the product expansion.
    `Z` and `X` are the diagonal and off-diagonal parts of the generator respectively.
    """

    stage: int = 0
    X: Optional[AbstractOperator] = None
    Z: Optional[AbstractOperator] = None
    # compression_state: PyTree = None


class AbstractPE(AbstractDiscretization):
    r"""
    Abstract class for the Product Expansion (PE) method.
    """

    @property
    def coefficients(self):
        r"""
        Abstract property for the array of coefficients.
        Coefficients is a complex array of shape `(n_substeps, 2)`.
        """
        raise NotImplementedError()  # pragma: no cover

    @property
    def stages(self):
        return len(self.coefficients)

    def init_state(self) -> ProductExpansionState:
        r"""
        Initializes the `DiscretizationState` structure containing supplementary information needed.

        Returns:
            An intialized `DiscretizationState` instance
        """
        return ProductExpansionState()

    def reset(
        self,
        solver_state: ProductExpansionState,
    ):
        return solver_state.replace(stage=0)

    def get_substep(
        self,
        generator: AbstractOperator,
        dt: float,
        t: float,
        y_t: VariationalState,
        solver_state: ProductExpansionState,
    ):
        vstate = y_t
        tstate = copy(vstate)
        tstate.replace_sampler_seed()

        coeffs = self.coefficients[solver_state.stage]

        a, b = coeffs
        U, V = None, None
        if a is not None:
            U = 1 + a * generator * dt
        if b is not None:
            V = 1 + b * generator * dt

        return vstate, tstate, U, V


class AbstractSPE(AbstractDiscretization):
    @property
    def coefficients(self):
        r"""
        Abstract property for the array of coefficients.
        Coefficients is a complex array of shape `(n_substeps, 2)`.
        """
        raise NotImplementedError()  # pragma: no cover

    @property
    def stages(self):
        return len(self.coefficients)

    def init_state(self) -> ProductExpansionState:
        r"""
        Initializes the `DiscretizationState` structure containing supplementary information needed.

        Returns:
            An intialized `DiscretizationState` instance
        """
        return ProductExpansionState()

    def reset(
        self,
        solver_state: ProductExpansionState,
    ):
        return solver_state.replace(stage=0)

    def get_substep(
        self,
        generator: AbstractOperator,
        dt: float,
        t: float,
        y_t: VariationalState,
        solver_state: ProductExpansionState,
    ):
        coeffs = self.coefficients[solver_state.stage]
        a, b, c = coeffs

        # TODO: this is a hack to avoid recomputing the Hamiltonian at each substep
        # We should find a better way to handle this. For instance using `_init_state`
        # if solver_state.stage == 0:
        #     Z, X = split_generator(generator)
        #     solver_state = solver_state.replace(X=X, Z=Z)
        # Z, X = solver_state.Z, solver_state.X
        Z, X = split_generator(generator)

        vstate = y_t
        vstate.variables = vstate.model.apply_zz(
            vstate.variables, Z, scale=c * dt
        )  # exact diagonal evolution
        tstate = copy(vstate)
        tstate.replace_sampler_seed()

        # setup the off-diagonal evolution which is performed by the 'execute' method of the compression algorithm
        U, V = None, None
        if a is not None:
            U = 1 + a * X * dt
        if b is not None:
            V = 1 + b * X * dt

        return vstate, tstate, U, V
