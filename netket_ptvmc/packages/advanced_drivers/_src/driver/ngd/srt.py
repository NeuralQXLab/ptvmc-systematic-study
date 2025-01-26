from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PositionalSharding

from netket import jax as nkjax
from netket.utils import mpi
from netket.utils.types import Union, Array

from netket_pro._src import distributed as distributed


@partial(
    jax.jit,
    static_argnames=(
        "solver_fn",
        "mode",
        "collect_quadratic_model",
    ),
)
def _compute_srt_update(
    O_L,
    dv,
    *,
    diag_shift: Union[float, Array],
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    proj_reg: Optional[Union[float, Array]] = None,
    momentum: Optional[Union[float, Array]] = None,
    old_updates: Optional[Array] = None,
    collect_quadratic_model: bool = False,
    params_structure,
):
    if momentum is not None:
        dv -= momentum * (O_L @ old_updates)

    # Equivalent to MPI.alltoall, shards the data across axis 1
    # (#ns, np) -> (ns, #np)
    O_LT, token = distributed.reshard(
        O_L, sharded_axis=0, out_sharded_axis=1, pad=True, pad_value=0.0
    )
    dv, token = distributed.allgather(dv, token=token)

    # This does the contraction (ns, #np) x (#np, ns) -> (ns, ns).
    # When using sharding the sum over #ns is done automatically.
    # When using MPI we need to do it manually with an allreduce_sum.
    matrix, token = mpi.mpi_reduce_sum_jax(O_LT @ O_LT.T, root=0, token=token)
    matrix_side = matrix.shape[-1]  # * it can be ns or 2*ns, depending on mode

    if mpi.rank == 0:
        matrix = matrix + diag_shift * jnp.eye(matrix_side, dtype=matrix.dtype)

        if proj_reg is not None:
            matrix += jnp.full_like(matrix, proj_reg / matrix_side)

        aus_vector = solver_fn(matrix, dv)

        # Some solvers return a tuple, some others do not.
        if isinstance(aus_vector, tuple):
            aus_vector, info = aus_vector
        else:
            info = {}

        aus_vector = aus_vector.reshape(mpi.n_nodes, -1)
        aus_vector, token = mpi.mpi_scatter_jax(aus_vector, root=0, token=token)
    else:
        aus_vector = jnp.zeros((int(matrix_side / mpi.n_nodes),), dtype=jnp.float64)
        aus_vector, token = mpi.mpi_scatter_jax(aus_vector, root=0, token=token)
        info = None

    if info is None:
        info = {}

    # (np, #ns) x (#ns) -> (np).
    # The sum over #ns is done automatically in sharding.
    # Under MPI we need to do it manually with an allreduce_sum.
    updates, token = mpi.mpi_allreduce_sum_jax(O_L.T @ aus_vector, token=token)
    if momentum is not None:
        updates += momentum * old_updates
        old_updates = updates

    if collect_quadratic_model:
        update_info, token = _compute_quadratic_model_srt(matrix, aus_vector, dv, token)
        info.update(update_info)

    # If complex mode and we have complex parameters, we need
    # To repack the real coefficients in order to get complex updates
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = updates.shape[-1] // 2
        updates = updates[:num_p] + 1j * updates[num_p:]

    if distributed.mode() == "sharding":
        out_shardings = (
            PositionalSharding(jax.devices()).replicate().reshape((1,) * updates.ndim)
        ).replicate()
        updates = jax.lax.with_sharding_constraint(updates, out_shardings)

    return updates, old_updates, info


def _compute_quadratic_model_srt(
    T: Array,  # (ns, ns)
    a: Array,  # (ns) on sharding or (#ns) on mpi
    dv: Array,  # (ns)
    token,
):
    r"""
    Computes the linear and quadratic terms of the SR update.
    The quadratic model reads:
    .. math::
        M(\delta) = h(\theta) + \delta^T \nabla h(\theta) + \frac{1}{2} \delta^T S \delta
    where :math:`h(\theta)` is the function to minimize. The linear and quadratic terms are:
    .. math::
        \text{linear_term} = \delta^T \nabla h(\theta)
    .. math::
        \text{quadratic_term} = \delta^T S \delta
    To make the computation more efficient, we use that :math:`\nabla h(\theta) = X^T \epsilon`, :math:`T = X X^T`, and :math:`S=X^T X`.
    The auxiliary vector :math:`a` is such that :math:`\delta = X^T a`. Therefore, the linear term is:
    .. math::
        \text{linear_term} = \delta^T \nabla h =  (X^T a)^T X^T \epsilon = a^T T \epsilon
    The quadratic term is:
    .. math::
        \text{quadratic_term} = \delta^T S \delta = (X^T a)^T X^T X X^T a = (T a)^T (T a)

    Args:
        T: The neural tangent kernel.
        a: The auxiliary vector.
        dv: The vector of local energies.

    Returns:
        A dictionary with the linear and quadratic terms.
    """
    # a : (#ns) -> (ns)
    if distributed.mode() == "mpi":
        a, token = distributed.allgather(a, token=token)

    # (ns)
    ϵ = dv.reshape(a.shape)

    # (ns, ns) x (ns) -> (ns)
    K = T @ a

    # (1, ns) x (ns, 1) -> (1, 1)
    linear = K.T @ ϵ

    # (1, ns) x (ns, 1) -> (1, 1)
    quadratic = K.T @ K

    return {"linear_term": linear, "quadratic_term": quadratic}, token
