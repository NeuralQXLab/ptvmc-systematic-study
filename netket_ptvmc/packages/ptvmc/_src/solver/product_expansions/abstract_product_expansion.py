from copy import copy

from netket.operator import AbstractOperator, DiscreteJaxOperator
from netket.vqs import VariationalState, MCState
from netket.utils import struct

from netket_pro.utils.diagonal_splitting import (
    split_hamiltonian as split_generator,
    decompose_paulioperator,
    check_if_allowed,
)

from ptvmc._src.solver.base import (
    AbstractDiscretization,
    AbstractDiscretizationState,
)


class ProductExpansionState(AbstractDiscretizationState):
    r"""
    State of the ProductExpansion.
    """

    # compression_state: PyTree = None


class SplitProductExpansionState(AbstractDiscretizationState):
    r"""
    State of the SplitProductExpansion.

    `split_generator` is a tuple containing the diagonal and off-diagonal parts of the generator, respectively.
    """

    split_generator: tuple[DiscreteJaxOperator, DiscreteJaxOperator] = struct.field(
        pytree_node=False, serialize=False
    )
    diagonal_decomposition: dict[str, tuple] = struct.field(
        pytree_node=False, serialize=False
    )
    """
    Stores the diagonal and non-diagonal parts of the generator, in that order.

    This is not defined if it cannot be precomputed, for example for time-depentent hamiltonians.
    """

    def __init__(self, split_generator=None, diagonal_decomposition=None):
        self.split_generator = split_generator
        self.diagonal_decomposition = diagonal_decomposition


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

    def init_state(
        self,
        generator: DiscreteJaxOperator,
        state_t0: VariationalState,
        t: float,
        dt: float,
    ) -> ProductExpansionState:
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
        if isinstance(tstate, MCState):
            tstate.replace_sampler_seed()

        coeffs = self.coefficients[solver_state.stage]

        a, b = coeffs
        U, V = None, None
        if a is not None:
            U = 1 + a * generator * dt
        if b is not None:
            V = 1 + b * generator * dt

        return vstate, tstate, U, V, solver_state


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

    def init_state(
        self,
        generator: DiscreteJaxOperator,
        state_t0: VariationalState,
        t: float,
        dt: float,
    ) -> ProductExpansionState:
        r"""
        Initializes the `DiscretizationState` structure containing supplementary information needed.

        Returns:
            An intialized `DiscretizationState` instance
        """
        model = state_t0.model
        if not hasattr(model, "supported_operations") or not hasattr(
            model, "apply_operations"
        ):
            raise TypeError(
                """
                Diagonal-split integration schemes require your Variational State to support the exact
                application of diagonal operations.

                This means that you should wrap your architecture into the `DiagonalWrapper`,
                or write yourself a nn Module which exposes the following fields:
                    - model.supported_operations -> should return a list/set of supported operations
                    - model.apply_operations(dictionary, variables, scale) -> should apply the list
                        of operations to the variables

                In general, you can just use the built in wrapper:

                ma = # your nn architecture nk.models.RBM()
                ma_with_diagonal = ptvmc.nn.DiagonalWrapper(ma, param_dtype=complex)
                vs = nk.vqs.MCState(sampler, ma_with_diagonal, ...)

                """
            )
        if isinstance(generator, DiscreteJaxOperator):
            Z, X = split_generator(generator)
            Z = Z.to_pauli_strings()
            check_if_allowed(Z, model.supported_operations)
            decomposition = decompose_paulioperator(
                Z, model.supported_operations, check=False
            )

            state = SplitProductExpansionState(
                split_generator=(Z, X), diagonal_decomposition=decomposition
            )
        else:
            state = ProductExpansionState()

        return state

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

        # If the hamiltonian is time-independent this should always be the case
        Z, X, diagonal_decomposition = get_decomposed_generator(
            generator, y_t.model.supported_operations, solver_state
        )
        # Apply the diagonal evolution to the parameters
        y_t.variables = y_t.model.apply_operations(
            diagonal_decomposition, variables=y_t.variables, scale=c * dt
        )

        tstate = copy(y_t)
        if isinstance(tstate, MCState):
            tstate.replace_sampler_seed()

        # setup the off-diagonal evolution which is performed by the 'execute' method of the compression algorithm
        U, V = None, None
        if a is not None:
            U = 1 + a * X * dt
        if b is not None:
            V = 1 + b * X * dt

        return y_t, tstate, U, V, solver_state

    def _apply_diagonal(self, vstate, Z, coeff, dt):
        # exact diagonal evolution
        decomposition = decompose_paulioperator(Z, vstate.model.supported_operations)
        vstate.variables = vstate.model.apply_operations(
            decomposition, variables=vstate.variables, scale=coeff * dt
        )
        return vstate


def get_decomposed_generator(generator, supported_operations, solver_state):
    """
    Returns the offdiagonal, diagonal and diagonal decomposition of the generator.
    If possible uses the precomputed values from the solver state, but if they
    are not available it will just decompose immediately.

    Returns:
        A tuple of 3 elements: the offidiagonal and diagonal operators, followed by
        the dictionary containing the decomposition of the diagonal part.
    """
    if solver_state.diagonal_decomposition is not None:
        Z, X = solver_state.split_generator
        diagonal_decomposition = solver_state.diagonal_decomposition
    else:
        if solver_state.split_generator is not None:
            Z, X = solver_state.split_generator
        else:
            Z, X = split_generator(generator)
        Z = Z.to_pauli_strings()
        diagonal_decomposition = decompose_paulioperator(Z, supported_operations)
    return Z, X, diagonal_decomposition
