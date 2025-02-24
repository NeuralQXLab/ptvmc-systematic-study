import jax.numpy as jnp

from netket.utils import struct
from ptvmc._src.solver.product_expansions.abstract_product_expansion import AbstractPE
from ptvmc._src.solver.product_expansions.lpe import append_docstring


ppe_docs = r"""
    Uses the Padé Product Expansion (PPE) expression for approximating the exponential of the generator.
    The resulting expression is a product series of two terms, a linear term and the inverse of a linear term.
    Formally, the PPE expansion is given by:
    .. math::

        e^{dt \hat{G}} = \prod_{i=1}^{s} (1 + b_i dt \hat{G})^{-1}(1 + a_i dt \hat{G})

    where :math:`a_i` and `b_i` are the complex coefficients and :math:`s` is the number of stages.
    The truncation is accurate to order :math:`2s`.

    At every stage, one compression is performed with :math:`U=(1 + a_i dt \hat{G})` and :math:`V=(1 + b_i dt \hat{G})`.

    This method was introduced by Gravina et al. in `ArXiV:24.10720 <https://arxiv.org/abs/2410.10720>`_ .
"""


@append_docstring(ppe_docs)
class PPE(AbstractPE):
    _a: jnp.ndarray = struct.field(pytree_node=True, serialize=False)
    _b: jnp.ndarray = struct.field(pytree_node=True, serialize=False)

    def __init__(self, fwd_coeffs, bwd_coeffs):
        r"""
        Initializes the PPE class with the given coefficients.

        Args:
            fwd_coeffs: The coefficients of the forward terms of the product expansion.
            bwd_coeffs: The coefficients of the backward terms of the product expansion.
        """
        self._a = fwd_coeffs
        self._b = bwd_coeffs

    @property
    def coefficients(self):
        return list(zip(self._a, self._b))

    def __repr__(self):
        return f"PPE{2*self.stages}()"


@append_docstring(ppe_docs)
def PPE2():
    r"""
    Second order Padé Product Expansion of the exponential of the generator.
    """
    a1 = +0.5
    b1 = -0.5
    return PPE(jnp.array([a1], dtype=jnp.float64), jnp.array([b1], dtype=jnp.float64))


@append_docstring(ppe_docs)
def PPE4():
    r"""
    Fourth order Padé Product Expansion of the exponential of the generator.
    """
    a1 = +(1 / 12) * (3 - 1j * jnp.sqrt(3))
    a2 = +(1 / 12) * (3 + 1j * jnp.sqrt(3))
    b1 = -(1 / 12) * (3 + 1j * jnp.sqrt(3))
    b2 = -(1 / 12) * (3 - 1j * jnp.sqrt(3))

    return PPE(
        jnp.array([a1, a2], dtype=jnp.complex128),
        jnp.array([b1, b2], dtype=jnp.complex128),
    )


@append_docstring(ppe_docs)
def PPE6():
    r"""
    Sixth order Padé Product Expansion of the exponential of the generator.
    """
    a1 = +0.2153144231161122
    a2 = +0.1423427884419439 - 1j * 0.1357999257081538
    a3 = +0.1423427884419439 + 1j * 0.1357999257081538
    b1 = -0.2153144231161122
    b2 = -0.1423427884419439 - 1j * 0.1357999257081538
    b3 = -0.1423427884419439 + 1j * 0.1357999257081538

    return PPE(
        jnp.array([a1, a2, a3], dtype=jnp.complex128),
        jnp.array([b1, b2, b3], dtype=jnp.complex128),
    )
