import jax.numpy as jnp

from netket.operator import AbstractOperator
from netket.utils import struct
from netket.vqs import VariationalState

from ptvmc._src.solver.product_expansions.abstract_product_expansion import (
    AbstractSPE,
    ProductExpansionState,
    get_decomposed_generator,
)
from ptvmc._src.solver.product_expansions.lpe import append_docstring


sppe_docs = r"""
    Uses the Splitted Padé Product Expansion (SPPE) expression for approximating the exponential of the generator.
    The resulting expression is a product series of two terms, a linear term and the inverse of a linear term.
    Formally, the SPPE expansion is given by:
    .. math::

        e^{dt \hat{G}} = \prod_{i=1}^{s} (1 + b_i dt \hat{X})^{-1}(1 + a_i dt \hat{X}) e^{c_i dt \hat{Z}}

    where :math:`a_i` and `b_i` are the complex coefficients and :math:`s` is the number of stages.
    The number of stages :math:`s` is monotonically increasing in the order of the truncation.

    At every stage, one exact application of the diagonal gates is performed, and is followed by a
    compression performed with :math:`U=(1 + a_i dt \hat{X})` and :math:`V=(1 + b_i dt \hat{X})`.

    After all steps are performed, a final diagonal application of :math:`e^{c_s dt \hat{Z}}` is performed.

    This method was introduced by Gravina et al. in `ArXiV:24.10720 <https://arxiv.org/abs/2410.10720>`_ .
"""


@append_docstring(sppe_docs)
class SPPE(AbstractSPE):
    _a: jnp.ndarray = struct.field(pytree_node=True, serialize=False)
    _b: jnp.ndarray = struct.field(pytree_node=True, serialize=False)
    _c: jnp.ndarray = struct.field(pytree_node=True, serialize=False)

    def __init__(self, fwd_xcoeffs, bwd_xcoeffs, zcoeffs):
        r"""
        Initializes the SPPE class with the given coefficients.

        Args:
            fwd_xcoeffs: The forward coefficients of the off-diagonal part of the product expansion.
            bwd_xcoeffs: The backward coefficients of the off-diagonal part of the product expansion.
            zcoeffs: The coefficients of the diagonal part of the product expansion.
        """
        self._a = fwd_xcoeffs
        self._b = bwd_xcoeffs
        self._c = zcoeffs

    @property
    def coefficients(self):
        # size(_c) = size(_a) + 1. Therefore, the last coefficient of _c is not included in coefficients.
        # The diagonal update associated to this terms is performed in the finish_step method.
        return list(zip(self._a, self._b, self._c))

    def __repr__(self):
        return f"SPPE{self.stages}()"

    def finish_step(
        self,
        generator: AbstractOperator,
        dt: float,
        t: float,
        y_t: VariationalState,
        solver_state: ProductExpansionState,
    ) -> VariationalState:
        """
        Final diagonal evolution of SPPE.
        """
        c = self._c[-1]
        _, _, diagonal_decomposition = get_decomposed_generator(
            generator, y_t.model.supported_operations, solver_state
        )
        y_t.variables = y_t.model.apply_operations(
            diagonal_decomposition, variables=y_t.variables, scale=c * dt
        )
        return y_t, solver_state


@append_docstring(sppe_docs)
def SPPE2():
    r"""
    Second order splitted Padé Product Expansion of the exponential of the generator.
    """
    a1 = 0.5
    b1 = -0.5

    c1 = 0.5
    c2 = 0.5

    return SPPE(
        jnp.array([a1], dtype=jnp.float64),
        jnp.array([b1], dtype=jnp.float64),
        jnp.array([c1, c2], dtype=jnp.float64),
    )


@append_docstring(sppe_docs)
def SPPE3():
    r"""
    Third order splitted Padé Product Expansion of the exponential of the generator.
    """

    a1 = (3 - 1j * jnp.sqrt(3)) / 12
    a2 = (3 + 1j * jnp.sqrt(3)) / 12

    b1 = (+1 / 12) * (-3 + 1j * jnp.sqrt(3))
    b2 = (-1 / 12) * (+3 + 1j * jnp.sqrt(3))

    c1 = (3 - 1j * jnp.sqrt(3)) / 12
    c2 = 0.5
    c3 = (3 + 1j * jnp.sqrt(3)) / 12

    return SPPE(
        jnp.array([a1, a2], dtype=jnp.complex128),
        jnp.array([b1, b2], dtype=jnp.complex128),
        jnp.array([c1, c2, c3], dtype=jnp.complex128),
    )
