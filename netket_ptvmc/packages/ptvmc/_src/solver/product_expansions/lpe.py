import jax.numpy as jnp

from netket.utils import struct
from ptvmc._src.solver.product_expansions.abstract_product_expansion import AbstractPE


def append_docstring(doc):
    """
    Decorator that appends the string `doc` to the decorated function.

    This is needed here because docstrings cannot be f-strings or manipulated strings.
    """

    def _append_docstring(fun):
        fun_doc = fun.__doc__ if fun.__doc__ is not None else ""

        fun.__doc__ = fun_doc + doc
        return fun

    return _append_docstring


lpe_docs = r"""
    Uses the Linear Product Expansion (LPE) expression for the truncated taylor series,
    which is a product of linear terms in the generator.

    .. math::

        e^{dt \hat{G}} = \prod_{i=1}^{s} (1 + a_i dt \hat{G})

    where :math:`a_i` are the complex coefficients and :math:`s` is the number of stages,
    corresponding to the order of the truncation.

    At every stage, one compression is performed with :math:`U=(1 + a_i dt \hat{G})` and :math:`V=1`.

    This method was introduced by Gravina et al. in `ArXiV:24.10720 <https://arxiv.org/abs/2410.10720>`_ .
"""


@append_docstring(lpe_docs)
class LPE(AbstractPE):
    _a: jnp.ndarray = struct.field(pytree_node=True, serialize=False)

    def __init__(self, coeffs):
        r"""
        Initializes the LPE class with the given coefficients.

        Args:
            coeffs: The coefficients of the linear product expansion.
        """
        self._a = coeffs

    @property
    def coefficients(self):
        return list(zip(self._a, [None for _ in self._a]))

    def __repr__(self):
        return f"LPE{self.stages}()"


@append_docstring(lpe_docs)
def LPE1():
    r"""
    First order truncation of the Taylor Series for the exponent of the generator.
    """
    a1 = 1.0
    return LPE(jnp.array([a1], dtype=jnp.float64))


@append_docstring(lpe_docs)
def LPE2():
    r"""
    Second order truncation of the Taylor Series for the exponent of the generator.
    """
    a1 = (1.0 - 1.0j) * 0.5
    a2 = (1.0 + 1.0j) * 0.5
    return LPE(jnp.array([a1, a2], dtype=jnp.complex128))


@append_docstring(lpe_docs)
def LPE3():
    r"""
    Third order truncation of the Taylor Series for the exponent of the generator.
    """
    a1 = 0.6265382932707997
    a2 = 0.1867308533646001 - 0.4807738845503311j
    a3 = 0.1867308533646001 + 0.4807738845503311j
    return LPE(jnp.array([a1, a2, a3], dtype=jnp.complex128))


@append_docstring(lpe_docs)
def LPE4():
    r"""
    Fourth order truncation of the Taylor Series for the exponent of the generator.
    """
    a1 = 0.0426266565027024 - 0.3946329531721134j
    a2 = 0.0426266565027024 + 0.3946329531721134j
    a3 = 0.4573733434972976 + 0.2351004879985427j
    a4 = 0.4573733434972976 - 0.2351004879985427j
    return LPE(jnp.array([a1, a2, a3, a4], dtype=jnp.complex128))
