from typing import Optional

from netket.experimental.observable import AbstractObservable
from netket.operator import AbstractOperator
from netket.utils.types import DType
from netket.vqs import VariationalState

from netket_pro._src.operator.jax_utils import to_jax_operator


class InfidelityUVOperator(AbstractObservable):
    r"""
    Infidelity operator to compute the infidelity between

    .. math::

        \frac{\langle \psi_\theta | \hat{V}^\dagger \hat{U}| \phi_t \rangle}{\langle \psi_\theta | \phi_t \rangle}

    """

    def __init__(
        self,
        target_state: VariationalState,
        *,
        U: Optional[AbstractOperator] = None,
        V: Optional[AbstractOperator] = None,
        cv_coeff: Optional[float] = None,
        dtype: Optional[DType] = None,
        resample_fraction: Optional[float] = None,
    ):
        r"""
        Args:
            target_state: target variational state |ϕ⟩.
            U: operator U.
            U_dagger: dagger operator U^{\dagger}.
            cv_coeff: Control Variates coefficient c.
            dtype: The dtype of the output of expectation value and gradient.
            sample_Upsi: flag specifiying whether to sample from |ϕ⟩ or from U|ϕ⟩. If False with `is_unitary=False`, an error occurs.
        """
        if not isinstance(target_state, VariationalState):
            raise TypeError("The first argument should be a variational state.")

        super().__init__(target_state.hilbert)

        U_target = to_jax_operator(U) if U is not None else None
        V_state = to_jax_operator(V) if V is not None else None

        self._target = target_state
        self._dtype = dtype

        self._cv_coeff = cv_coeff
        self._U_target = U_target
        self._V_state = V_state

        self._resample_fraction = resample_fraction

    @property
    def target(self):
        return self._target

    @property
    def U_target(self):
        return self._U_target

    @property
    def V_state(self):
        return self._V_state

    @property
    def dtype(self):
        return self._dtype

    @property
    def cv_coeff(self):
        return self._cv_coeff

    def collect(self) -> AbstractObservable:
        return self

    @property
    def is_hermitian(self):
        return True

    @property
    def resample_fraction(self) -> Optional[float]:
        return self._resample_fraction

    def __repr__(self):
        return f"InfidelityUVOperator(target=U@{self.target}, U_target={self.U_target}, V_state={self.V_state}, cv_coeff={self.cv_coeff})"
