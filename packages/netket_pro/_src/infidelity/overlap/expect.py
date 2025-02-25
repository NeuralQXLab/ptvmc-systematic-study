from typing import Optional

from netket.vqs import (
    MCState,
    expect,
    expect_and_grad,
)

from .operator import InfidelityUVOperator
from .grad_hermitian import infidelity_hermitian


@expect.dispatch
def infidelity(
    vstate: MCState,
    op: InfidelityUVOperator,
    chunk_size: Optional[int],
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    # Compute the effective chunk size
    if chunk_size is not None:
        chunk_size = chunk_size // max(
            getattr(op.V_state, "max_conn_size", 1),
            getattr(op.U_target, "max_conn_size", 1),
        )

    return infidelity_hermitian(
        vstate, op, chunk_size=chunk_size, mutable=None, return_grad=False
    )


@expect_and_grad.dispatch
def infidelity(  # noqa: F811
    vstate: MCState,
    op: InfidelityUVOperator,
    chunk_size: Optional[int] = None,
    *,
    mutable,
):
    return infidelity_hermitian(
        vstate, op, chunk_size=chunk_size, mutable=mutable, return_grad=True
    )
