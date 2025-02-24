from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from flax import core as fcore
from jax.tree_util import tree_map

from netket import jax as nkjax
from netket.vqs import (
    MCState,
    get_local_kernel_arguments,
    get_local_kernel,
)
from netket.utils import mpi
from netket.stats import statistics, mean
from netket.vqs.mc.common import force_to_grad

from typing import Callable
from netket.utils.types import PyTree, Array
from flax.core.scope import CollectionFilter  # noqa: F401

from .operator import InfidelityUVOperator
from .exact import _prepare

"""
This file contains the declaration of get_local_kernel and similar functions
Necessary to make infidelity optimisation work with SRt.
"""


@get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(
    vstate: MCState, op: InfidelityUVOperator, chunk_size: Optional[int] = None
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    _, Vψ_vars, Vψ_logdistribution = _prepare(vstate, op.V_state, extra_hash_data="V")
    _, Uϕ_vars, Uϕ_logdistribution = _prepare(
        op.target, op.U_target, extra_hash_data="U"
    )
    Vψ_samples = vstate.samples_distribution(
        Vψ_logdistribution,
        variables=Vψ_vars,
        resample_fraction=op.resample_fraction,
        chain_name="Vpsi" if op.V_state is not None else "default",
    )

    Uϕ_samples = op.target.samples_distribution(
        Uϕ_logdistribution,
        variables=Uϕ_vars,
        resample_fraction=op.resample_fraction,
        chain_name="Uphi" if op.U_target is not None else "default",
    )

    return Vψ_samples, (Vψ_vars, Uϕ_samples, Uϕ_vars)


def local_estimator(
    Vψ_logfun,
    Uϕ_logfun,
    _,
    __,
    Vψ_samples,
    args,
    Vψ_chunk_size: Optional[int] = None,
    Uϕ_chunk_size: Optional[int] = None,
    # unused argument, for compatibility
    chunk_size: int = None,
):
    Vψ_vars, Uϕ_samples, Uϕ_vars = args

    out_shape = Vψ_samples.shape[:-1]

    # equivalent to .reshape(-1, N)
    Vψ_samples = jax.lax.collapse(Vψ_samples, 0, Vψ_samples.ndim - 1)
    Uϕ_samples = jax.lax.collapse(Uϕ_samples, 0, Uϕ_samples.ndim - 1)

    Vψ_logfun_ = nkjax.apply_chunked(
        partial(Vψ_logfun, Vψ_vars), chunk_size=Vψ_chunk_size
    )
    Uϕ_logfun_ = nkjax.apply_chunked(
        partial(Uϕ_logfun, Uϕ_vars), chunk_size=Uϕ_chunk_size
    )

    logVψ_x = Vψ_logfun_(Vψ_samples)
    logUϕ_x = Uϕ_logfun_(Vψ_samples)
    logVψ_y = Vψ_logfun_(Uϕ_samples)
    logUϕ_y = Uϕ_logfun_(Uϕ_samples)

    logH = logUϕ_x - logVψ_x
    logE = logVψ_y - logUϕ_y
    logF = logE + logH

    A = jnp.exp(logF)
    E = jnp.exp(logE)
    H = jnp.exp(logH)

    E = mean(E)
    H = E * H

    H = H.reshape(out_shape)
    A = A.reshape(out_shape)
    return H, A


@get_local_kernel.dispatch
def get_local_kernel(
    vstate: MCState, op: InfidelityUVOperator, chunk_size: Optional[int] = None
):
    Vψ_logfun, _, _ = _prepare(vstate, op.V_state, extra_hash_data="V")
    Uϕ_logfun, _, _ = _prepare(op.target, op.U_target, extra_hash_data="U")

    # Default
    Vψ_chunk_size = None
    Uϕ_chunk_size = None

    # Compute the effective chunk size
    # chunk size of vstate overrides everything.
    # If unset, chunk size of target applies to target
    if op.target.chunk_size is not None:
        Uϕ_chunk_size = op.target.chunk_size // getattr(op.U_target, "max_conn_size", 1)

    if chunk_size is not None:
        Vψ_chunk_size = max(chunk_size // getattr(op.V_state, "max_conn_size", 1), 1)
        Uϕ_chunk_size = max(chunk_size // getattr(op.U_target, "max_conn_size", 1), 1)

    return nkjax.HashablePartial(
        local_estimator,
        Vψ_logfun,
        Uϕ_logfun,
        Vψ_chunk_size=Vψ_chunk_size,
        Uϕ_chunk_size=Uϕ_chunk_size,
    )


def infidelity_hermitian(  # noqa: F811
    vstate: MCState,
    op: InfidelityUVOperator,
    chunk_size: Optional[int] = None,
    *,
    return_grad,
    mutable=None,
):
    σ, args = get_local_kernel_arguments(vstate, op)
    local_estimator_fun = get_local_kernel(vstate, op, chunk_size)

    if not return_grad:
        I_stats = infidelity_expect_hermitian(
            chunk_size,
            local_estimator_fun,
            vstate._apply_fun,
            return_grad,
            None,
            vstate.variables,
            σ,
            args,
            op.cv_coeff,
        )
        return I_stats

    else:
        I_stats, I_grad, new_model_state = infidelity_expect_hermitian(
            chunk_size,
            local_estimator_fun,
            vstate._apply_fun,
            return_grad,
            mutable,
            vstate.variables,
            σ,
            args,
            op.cv_coeff,
        )

        if mutable is not False:
            vstate.model_state = new_model_state

        return I_stats, I_grad


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def infidelity_expect_hermitian(
    chunk_size: int,
    local_value_kernel_chunked: Callable,
    apply_fun: Callable,
    return_grad: bool,
    mutable: CollectionFilter,
    variables: PyTree,
    σ: Array,
    local_value_args: PyTree,
    cv_coeff: Optional[float] = None,
) -> tuple[PyTree, PyTree]:
    model_state, parameters = fcore.pop(variables, "params")

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples = σ.shape[0] * mpi.n_nodes

    H_loc, A = local_value_kernel_chunked(
        apply_fun,
        variables,
        σ,
        local_value_args,
        chunk_size=chunk_size,
    )

    # compute fidelity
    res = A.real
    if cv_coeff is not None:
        res = res + cv_coeff * (jnp.abs(A) ** 2 - 1)

    F_stats = statistics(res.reshape(σ_shape[:-1]).T)

    I_stats = F_stats.replace(mean=jnp.abs(1 - F_stats.mean))

    if not return_grad:
        return I_stats
    else:
        if mutable is False:
            vjp_fun_chunked = nkjax.vjp_chunked(
                lambda w, σ: apply_fun({"params": w, **model_state}, σ),
                parameters,
                σ,
                conjugate=True,
                chunk_size=chunk_size,
                chunk_argnums=1,
                nondiff_argnums=1,
            )
            new_model_state = None
        else:
            raise NotImplementedError

        # Compute the gradient
        H_loc -= mean(H_loc)
        I_grad = vjp_fun_chunked(
            -jnp.sign(I_stats.mean) * (jnp.conjugate(H_loc) / n_samples),
        )[0]

        I_grad = force_to_grad(I_grad, parameters)

        return (
            I_stats,
            tree_map(lambda x: mpi.mpi_sum_jax(x)[0], I_grad),
            new_model_state,
        )
