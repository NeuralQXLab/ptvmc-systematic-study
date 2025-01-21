import numpy as np
import scipy.sparse as sp

from netket.operator import LocalOperator
from netket.utils.types import Array
from typing import Union

from tqdm import tqdm
from copy import copy


def integrator(psi0, e_ops, tf, dt, f_apply, save=True):
    psi = copy(psi0)
    psi_list = []
    e_ops_list = []

    _step = f_apply(dt)

    t_l = np.arange(0, tf, dt)
    for t in tqdm(t_l):
        if save:
            psi_list.append(psi)
            e_ops_list.append(
                [
                    np.vdot(
                        psi / np.linalg.norm(psi), op.dot(psi / np.linalg.norm(psi))
                    )
                    for op in e_ops
                ]
            )

        psi = _step(psi)
        psi = psi / np.linalg.norm(psi)

    psi_list.append(psi)
    e_ops_list.append([np.vdot(psi, op.dot(psi)) for op in e_ops])

    return psi_list, np.array(e_ops_list)


def _split_hamiltonian(H: Union[Array, LocalOperator]):
    if isinstance(H, Union[Array, sp.spmatrix]):
        return _split_hamiltonian_array(H)
    elif isinstance(H, LocalOperator):
        return _split_hamiltonian_netket(H)
    else:
        raise ValueError("Unknown type: ", type(H))


def _split_hamiltonian_array(H: Union[Array, sp.spmatrix]):
    H_diag = H.diagonal()
    H_offdiag = H - sp.diags(H_diag)
    return H_offdiag, H_diag


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

    return H_offdiag, H_diag
