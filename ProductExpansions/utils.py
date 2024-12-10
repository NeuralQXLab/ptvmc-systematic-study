import numpy as np
import scipy.sparse as sp

from netket.operator import LocalOperator
from netket.utils.types import Array
from typing import Union


def _split_hamiltonian(H: Union[Array, LocalOperator]):
    if isinstance(H, Array):
        return _split_hamiltonian_dense(H)
    elif isinstance(H, LocalOperator):
        return _split_hamiltonian_netket(H)
    else:
        raise ValueError("Unknown type")

def _split_hamiltonian_dense(H: Array):
    H_diag = sp.diags(H.diagonal())
    H_offdiag = H - H_diag
    return H_diag, H_offdiag
    

def _split_hamiltonian_netket(H: LocalOperator):
    hi = H.hilbert
    H_diag = LocalOperator(hi, dtype=H.dtype)
    H_offdiag = LocalOperator(hi, dtype=H.dtype)

    for op, support in zip(H.operators, H.acting_on):
        # check if there is a diagonal part
        if not isinstance(op, np.ndarray):
            op = op.todense()

        if (not np.allclose(np.diagonal(op), 0)) and len(support) <= 2:
            # can be done analytically
            op_diag = np.diag(np.diagonal(op))
            H_diag += LocalOperator(hi, op_diag, support)
        else:
            op_diag = 0.0

        op_offdiag = op - op_diag
        if not np.allclose(op_offdiag, 0):
            H_offdiag += LocalOperator(hi, op_offdiag, support)

    return H_diag, H_offdiag