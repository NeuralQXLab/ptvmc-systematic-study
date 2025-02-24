import jax.numpy as jnp

from netket.utils import struct
from ptvmc._src.solver.product_expansions.abstract_product_expansion import AbstractSPE
from ptvmc._src.solver.product_expansions.lpe import append_docstring


slpe_docs = r"""
    Uses the Splitted Linear Product Expansion (SLPE) expression for the truncated taylor series,
    which is a product series composed of alernating linear terms in the off-diagonal part of the
    generator :math:`\hat{X}`, and exact exponentials of the diagonal part of the generator :math:`\hat{Z}`.

    .. math::

        e^{dt\hat{G}} = \prod_{i=1}^{s} (1 + a_i dt \hat{X}) e^{c_i dt \hat{Z}}

    where :math:`a_i` and :math:`c_i` are the complex coefficients. The number of stages :math:`s`
    is monotonically increasing in the order of the truncation.

    At every stage, one exact application of the diagonal gates is performed, and is followed by a
    compression performed with :math:`U=(1 + a_i dt \hat{X})` and :math:`V=1`.

    This method was introduced by Gravina et al. in `ArXiV:24.10720 <https://arxiv.org/abs/2410.10720>`_ .
"""


@append_docstring(slpe_docs)
class SLPE(AbstractSPE):
    _a: jnp.ndarray = struct.field(pytree_node=True, serialize=False)
    _c: jnp.ndarray = struct.field(pytree_node=True, serialize=False)

    def __init__(self, xcoeffs, zcoeffs):
        r"""
        Initializes the SLPE class with the given coefficients.

        Args:
            xcoeffs: The coefficients of the off-diagonal part of the linear product expansion.
            zcoeffs: The coefficients of the diagonal part of the linear product expansion.
        """
        self._a = xcoeffs
        self._c = zcoeffs

    @property
    def coefficients(self):
        return list(zip(self._a, [None for _ in self._a], self._c))

    def __repr__(self):
        return f"SLPE{self.stages}()"


@append_docstring(slpe_docs)
def SLPE1():
    r"""
    First order splitted truncation of the Taylor Series for the exponent of the generator.
    """
    a1 = 1.0
    c1 = 1.0
    return SLPE(jnp.array([a1], dtype=jnp.float64), jnp.array([c1], dtype=jnp.float64))


@append_docstring(slpe_docs)
def SLPE2():
    r"""
    Second order splitted truncation of the Taylor Series for the exponent of the generator.
    """
    a1 = (1.0 - 1.0j) * 0.5
    a2 = (1.0 + 1.0j) * 0.5
    c1 = (1.0 - 1.0j) * 0.5
    c2 = (1.0 + 1.0j) * 0.5
    return SLPE(
        jnp.array([a1, a2], dtype=jnp.complex128),
        jnp.array([c1, c2], dtype=jnp.complex128),
    )


@append_docstring(slpe_docs)
def SLPE3():
    r"""
    Third order splitted truncation of the Taylor Series for the exponent of the generator.
    """
    a1 = 0.1056624327025936 - 0.3943375672974064j
    a2 = 0.3943375672974064 + 0.1056624327025936j
    a3 = 0.3943375672974064 - 0.1056624327025936j
    a4 = 0.1056624327025936 + 0.3943375672974064j
    c1 = 0.1056624327025936 - 0.3943375672974064j
    c2 = 0.3943375672974064 + 0.1056624327025936j
    c3 = 0.3943375672974064 - 0.1056624327025936j
    c4 = 0.1056624327025936 + 0.3943375672974064j
    return SLPE(
        jnp.array([a1, a2, a3, a4], dtype=jnp.complex128),
        jnp.array([c1, c2, c3, c4], dtype=jnp.complex128),
    )
