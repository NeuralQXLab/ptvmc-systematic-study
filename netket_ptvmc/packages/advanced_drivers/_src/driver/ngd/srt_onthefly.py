from typing import Callable, Optional
from functools import partial

from einops import rearrange

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from netket import jax as nkjax
from netket.utils import mpi
from netket.utils.types import Union, Array

from netket_pro._src import distributed as distributed
from netket_pro._src.external import neural_tangents as nt


@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "chunk_size",
        "mode",
        "collect_quadratic_model",
        "rloo",
    ),
)
def srt_onthefly(
    log_psi,
    local_energies,
    parameters,
    model_state,
    samples,
    *,
    diag_shift: Union[float, Array],
    solver_fn: Callable[[Array, Array], Array],
    mode: nt.NtkImplementation,
    e_mean: Optional[Union[float, Array]] = None,
    proj_reg: Optional[Union[float, Array]] = None,
    momentum: Optional[Union[float, Array]] = None,
    old_updates: Optional[Array] = None,
    chunk_size: Optional[int] = None,
    collect_quadratic_model: bool = False,
    rloo: bool = False,
):
    N_mc = local_energies.size * mpi.n_nodes

    # Split all parameters into real and imaginary parts separately
    parameters_real, rss = nkjax.tree_to_real(parameters)

    # (Nmc) -> (Nmc,2) - splitting real and imaginary output like 2 classes
    def _apply_fn(parameters_real, samples):
        variables = {"params": rss(parameters_real), **model_state}
        log_amp = log_psi(variables, samples)

        re, im = log_amp.real, log_amp.imag
        return jnp.concatenate((re[:, None], im[:, None]), axis=-1)  # shape [N_mc,2]

    def jvp_f_chunk(parameters, vector, samples):
        r"""
        Creates the jvp of the function `_apply_fn` with respect to the parameters.
        This jvp is then evaluated in chunks of `chunk_size` samples.
        """
        f = lambda params: _apply_fn(params, samples)
        _, acc = jax.jvp(f, (parameters,), (vector,))
        return acc

    # compute rhs of the linear system
    if e_mean is not None:
        de = local_energies - e_mean
    de = de.flatten()

    # TODO: RLOO should be added to ntk calculation and to the final vjp.
    # At the moment the final vjp is centered by centering the auxiliary vector a.
    # This is the same as centering the jacobian but may have larger variance.
    RLOO = N_mc / (N_mc - 1) if rloo else 1.0
    dv = RLOO * 2.0 * de / jnp.sqrt(N_mc)  # shape [N_mc,]
    dv = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)  # shape [N_mc,2]

    token = None
    if momentum is not None:
        if old_updates is None:
            old_updates = tree_map(jnp.zeros_like, parameters_real)
        else:
            acc = nkjax.apply_chunked(
                jvp_f_chunk, in_axes=(None, None, 0), chunk_size=chunk_size
            )(parameters_real, old_updates, samples)

            avg, token = mpi.mpi_mean_jax(jnp.mean(acc, axis=0), token=token)
            acc = (acc - avg) / jnp.sqrt(N_mc)
            dv -= momentum * acc

    dv = jax.lax.collapse(dv, 0, 2)  # shape [2*N_mc,]
    dv, token = distributed.allgather(dv, token=token)  # shape [2*N_mc,]

    # Collect all samples on all MPI ranks, those label the columns of the T matrix
    all_samples, token = distributed.allgather(samples, token=token)

    _jacobian_contraction = nt.empirical_ntk_fn(
        f=_apply_fn,
        trace_axes=(),
        vmap_axes=0,
        implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION,
    )

    def jacobian_contraction(samples, all_samples, parameters_real):
        if chunk_size is None:
            # STRUCTURED_DERIVATIVES returns a complex array, but the imaginary part is zero
            # shape [N_mc/p.size, N_mc, 2, 2]
            return _jacobian_contraction(samples, all_samples, parameters_real).real
        else:
            _all_samples, _ = nkjax.chunk(all_samples, chunk_size=chunk_size)
            ntk_local = jax.lax.map(
                lambda batch_lattice: _jacobian_contraction(
                    samples, batch_lattice, parameters_real
                ).real,
                _all_samples,
            )
            return rearrange(ntk_local, "nbatches i j z w -> i (nbatches j) z w")

    # If we are sharding, use shard_map manually
    if distributed.mode() == "sharding":
        mesh = jax.make_mesh(
            (distributed.device_count(),), ("i",), devices=jax.devices()
        )
        # SAMPLES, ALL_SAMPLES PARAMETERS_REAL
        in_specs = (P("i", None), P(), P())
        out_specs = P("i", None, None, None)

        sharding_decorator = partial(
            shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )
        jacobian_contraction = sharding_decorator(jacobian_contraction)

    # This disables the nkjax.sharding_decorator in here, which might appear
    # in the apply function inside.
    with nkjax.sharding._increase_SHARD_MAP_STACK_LEVEL():
        ntk_local = jacobian_contraction(samples, all_samples, parameters_real).real

    # shape [N_mc, N_mc, 2, 2]
    ntk, token = distributed.allgather(ntk_local, token=token)
    # shape [2*N_mc, 2*N_mc] checked with direct calculation of J^T J
    ntk = rearrange(ntk, "i j z w -> (i z) (j w)")

    # Center the NTK by multiplying with a carefully designed matrix
    # shape [N_mc, N_mc] symmetric matrix
    delta = jnp.eye(N_mc) - 1 / N_mc
    # shape [2*N_mc, 2*N_mc]
    # Gets applied to the sub-blocks corresponding to the real part and imaginary part
    delta_conc = jnp.zeros((2 * N_mc, 2 * N_mc)).at[0::2, 0::2].set(delta)
    delta_conc = delta_conc.at[1::2, 1::2].set(delta)
    delta_conc = delta_conc.at[0::2, 1::2].set(0.0)
    delta_conc = delta_conc.at[1::2, 0::2].set(0.0)

    # shape [2*N_mc, 2*N_mc] centering the jacobian
    ntk = (delta_conc @ (ntk @ delta_conc)) / N_mc

    # add projection regularization
    if proj_reg is not None:
        ntk = ntk + proj_reg / N_mc

    # add diag shift
    ntk = ntk + diag_shift * jnp.eye(ntk.shape[0])

    # some solvers return a tuple, some others do not.
    aus_vector = solver_fn(ntk, dv)
    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
    else:
        info = {}

    if info is None:
        info = {}

    if collect_quadratic_model:
        info.update(_compute_quadratic_model_srt_onthefly(ntk, aus_vector, dv))

    # Center the vector, equivalent to centering
    # The Jacobian
    aus_vector = aus_vector / jnp.sqrt(N_mc)
    aus_vector = delta_conc @ aus_vector

    # shape [N_mc,2]
    aus_vector = aus_vector.reshape(-1, 2)
    aus_vector = distributed.shard_replicated(
        aus_vector, axis=0
    )  # shape [N_mc // p.size,2]

    # _, vjp_fun = jax.vjp(f, parameters_real)
    vjp_fun = nkjax.vjp_chunked(
        _apply_fn,
        parameters_real,
        samples,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=1,
    )

    updates = vjp_fun(aus_vector)[0]  # pytree [N_params,]

    # Must pool the updates among MPI ranks BEFORE caching them into
    # old_updates, otherwise the `old_updates` will diverge among ranks
    updates, token = mpi.mpi_allreduce_sum_jax(updates, token=token)

    if momentum is not None:
        updates = tree_map(lambda x, y: x + momentum * y, updates, old_updates)
        old_updates = updates

    return rss(updates), old_updates, info


def _compute_quadratic_model_srt_onthefly(
    T: Array,
    a: Array,
    dv: Array,
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
    K = T @ a
    linear = K.T @ dv
    quadratic = K.T @ K
    return {"linear_term": linear, "quadratic_term": quadratic}
