import numpy as np

from netket.operator import LocalOperator, PauliStrings, DiscreteOperator

"""
This file contains functions to handle the diagonal part of an Hamiltonian, and
split it in terms that we can apply as a jastrow - like term.
"""


def ensure_pauli_string(op: LocalOperator | PauliStrings) -> PauliStrings:
    if isinstance(op, PauliStrings):
        return op
    return op.to_pauli_strings()


def _check_if_known(arr: np.ndarray[np.str_], allowed_operations: set[str]):
    """
    Given an array of PauliStrings, and given a list of known operations, checks
    that the paulistrings have only the allowed operations.
    """
    # Transform strings like "IIIXXII" into "XX"
    terms = np.char.replace(arr, "I", "")
    if not np.isin(terms, list(allowed_operations)).all():
        raise ValueError(f"Unknown operation to act diagonally: {terms}")


def check_if_allowed(op: LocalOperator | PauliStrings, allowed_operations: set[str]):
    r"""
    This function checks if the operators in the PauliStrings are known operations.
    The identities are removed so that only the pauli operators are checked.

    For example, 'XIZI' is checked as 'XZ'. The two-qubit operation 'XZ' is not known and will raise error.
    """
    return _check_if_known(ensure_pauli_string(op).operators, allowed_operations)


def decompose_paulioperator(
    op: DiscreteOperator, operations: set[str] = {}, check: bool = True
):
    r"""
    This function decomposes a Pauli operator into a dictionary of couplings.

    The coupling is a string that represents an operation, e.g. 'ZZ'.
    If the two-qubit operation 'ZZ' is not known, an error is raised.

    If the operation is known, the PauliString implementing such operation are grouped under the key 'ZZ'.
    In general the implemented operator will be a sum of PauliStrings implementing the same operation on different qubits.
    From these PauliStrings we extract the qubits on which the operation is acting and the coefficients with which they appear.
    All this information, for all available operations/keys, are stored in the dictionary `decomposition`.

    Args:
        op: The operator to decompose.
        operations: The set of allowed operations.
        check: If True, checks that the operator can be fully decomposed into the given set of operations, and fail
               otherwise. If False, a partial decomposition might be returned.

    Returns:
        A dictionary containing the decomposition of the operator.
    """
    op = ensure_pauli_string(op)
    ops = op.operators
    weights = op.weights

    if check:
        if not operations:
            raise ValueError("No operations provided to decompose the operator.")

        check_if_allowed(op, operations)

    # Remove 'I' characters to form the coupling strings
    couplings = np.char.replace(ops, "I", "")

    # `np.unique` returns the unique couplings.
    # `return_inverse=True` gives the indices of the original array to which each unique element correpsonds.
    # For instance if `couplings` is ['XX', 'YY', 'XX'], then `unique_couplings` will be ['XX', 'YY'] and `inverse` will be [0, 1, 0].
    unique_couplings, inverse = np.unique(couplings, return_inverse=True)

    decomposition = {}
    for i, coupling in enumerate(unique_couplings):
        indices = np.nonzero(inverse == i)[0]

        decomposition[coupling] = {
            "indices": indices,
            "operators": ops[indices],
            "acting_on": _acting_on_from_pauli_strings(ops[indices]),
            "weights": weights[indices],
        }

    return decomposition


def _acting_on_from_pauli_strings(s: str | np.ndarray):
    ascii = ord("I")

    if isinstance(s, str):
        s_array = np.frombuffer(s.encode(), dtype=np.uint8)
        return np.where(s_array != ascii)[0]

    if (
        isinstance(s, np.ndarray) and s.dtype.kind == "U"
    ):  # 'U' means Unicode string dtype
        byte_matrix = np.char.encode(s, "utf-8").view(np.uint8).reshape(len(s), -1)

        row_indices, col_indices = np.where(byte_matrix != ascii)
        unique_rows, counts = np.unique(row_indices, return_counts=True)
        split_indices = np.split(col_indices, np.cumsum(counts)[:-1])

        indices_list = [()] * len(s)
        for i, row in zip(unique_rows, split_indices):
            indices_list[i] = tuple(row)

        return indices_list


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
