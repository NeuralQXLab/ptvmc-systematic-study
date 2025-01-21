import numpy as np

from netket.operator import LocalOperator


def split_hamiltonian(H: LocalOperator) -> tuple[LocalOperator, LocalOperator]:
    r"""
    Splits a Hamiltonian into a diagonal and off-diagonal part.

    Args:
        H: The Hamiltonian to split.

    Returns:
        A tuple containing the diagonal and off-diagonal parts of the Hamiltonian.
    """
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
