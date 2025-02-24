from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from netket import jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket.vqs import MCState, expect, expect_and_grad, get_local_kernel_arguments
from netket.utils import mpi

from netket_pro._src.jax import expect_advanced

from .operator import InfidelityOperatorUPsi


@expect.dispatch
def infidelity(
    vstate: MCState, op: InfidelityOperatorUPsi, chunk_size: Optional[int] = None
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    sigma, args = get_local_kernel_arguments(vstate, op._U)
    sigma_t, args_t = get_local_kernel_arguments(op.target, op._U_dagger)

    return infidelity_sampling_MCState(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        sigma,
        args,
        sigma_t,
        args_t,
        op.cv_coeff,
        return_grad=False,
        chunk_size=chunk_size,
    )


@expect_and_grad.dispatch
def infidelity(  # noqa: F811
    vstate: MCState,
    op: InfidelityOperatorUPsi,
    chunk_size: Optional[int] = None,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    sigma, args = get_local_kernel_arguments(vstate, op._U)
    sigma_t, args_t = get_local_kernel_arguments(op.target, op._U_dagger)

    return infidelity_sampling_MCState(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        sigma,
        args,
        sigma_t,
        args_t,
        op.cv_coeff,
        return_grad=True,
        chunk_size=chunk_size,
    )


@partial(jax.jit, static_argnames=("afun", "afun_t", "return_grad", "chunk_size"))
def infidelity_sampling_MCState(
    afun,
    afun_t,
    params,
    params_t,
    model_state,
    model_state_t,
    sigma,
    args,
    sigma_t,
    args_t,
    cv_coeff,
    return_grad,
    chunk_size,
):
    N = sigma.shape[-1]
    n_chains_t = sigma_t.shape[-2]

    σ = sigma.reshape(-1, N)
    σ_t = sigma_t.reshape(-1, N)

    if isinstance(args, DiscreteJaxOperator):
        jax_op = True

        xp, mels = None, None
        xp_t, mels_t = None, None
    else:
        jax_op = False
        xp = args[0].reshape(σ.shape[0], -1, N)
        mels = args[1].reshape(σ.shape[0], -1)
        xp_t = args_t[0].reshape(σ_t.shape[0], -1, N)
        mels_t = args_t[1].reshape(σ_t.shape[0], -1)

    def expect_kernel(params):
        def kernel_fun(params_all, σ, σ_t, xp, xp_t, mels, mels_t):
            if jax_op:
                xp, mels = args.get_conn_padded(σ)
                xp_t, mels_t = args_t.get_conn_padded(σ_t)

            params, params_t = params_all
            W = {"params": params, **model_state}
            W_t = {"params": params_t, **model_state_t}

            logpsi_t_xp = afun_t(W_t, xp)
            logpsi_xp_t = afun(W, xp_t)

            log_val = (
                logsumexp(logpsi_t_xp, axis=-1, b=mels)
                + logsumexp(logpsi_xp_t, axis=-1, b=mels_t)
                - afun(W, σ)
                - afun_t(W_t, σ_t)
            )
            res = jnp.exp(log_val).real
            if cv_coeff is not None:
                res = res + cv_coeff * (jnp.exp(2 * log_val.real) - 1)
            return res

        def log_pdf_joint(params_all, σ, σ_t, xp_ravel, xp_t_ravel, mels, mels_t):
            params, params_t = params_all
            log_pdf_vals = afun({"params": params, **model_state}, σ).real
            log_pdf_t_vals = afun_t({"params": params_t, **model_state_t}, σ_t).real
            return 2 * (log_pdf_vals + log_pdf_t_vals)

        return expect_advanced(
            log_pdf_joint,
            kernel_fun,
            (
                params,
                params_t,
            ),
            σ,
            σ_t,
            xp,
            xp_t,
            mels,
            mels_t,
            n_chains=n_chains_t,
        )

    if not return_grad:
        F, F_stats = expect_kernel(params)
        return F_stats.replace(mean=1 - F)

    F, F_vjp_fun, F_stats = nkjax.vjp(
        expect_kernel, params, has_aux=True, conjugate=True
    )

    F_grad = F_vjp_fun(jnp.ones_like(F))[0]
    F_grad = jax.tree_util.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], F_grad)
    I_grad = jax.tree_util.tree_map(lambda x: -x, F_grad)
    I_stats = F_stats.replace(mean=1 - F)

    return I_stats, I_grad
