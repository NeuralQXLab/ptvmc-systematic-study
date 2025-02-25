from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp

from netket.utils.types import Array, PyTree
from netket.jax import apply_chunked
from netket.stats import mean
from netket.operator import AbstractOperator
from netket.vqs import MCState

from netket_pro.utils.sampling_Ustate import _lazy_apply_UV_to_afun
from advanced_drivers._src.driver.ngd.driver_abstract_ngd import _flatten_samples


def infidelity_UV_kernel_args(
    vstate: MCState,
    tstate: MCState,
    U_target: Optional[AbstractOperator] = None,
    V_state: Optional[AbstractOperator] = None,
    sample_Uphi: Optional[bool] = True,
    resample_fraction: Optional[float] = None,
):
    r"""In this function we sample (if the cache is empty) or access the cache (if the cache is not empty).
    Typically the cache is empty as a result of :code:`vstate.reset()` in the :code:`reset_step` method.

    For the variational state, the function always returns Vpsi and related samples.
    For the target state, the function returns Uphi and related samples if :code:`sample_Uphi=True`,
    otherwise it returns the bare target wave function and its samples.

    Uphi but NOT its samples are always returned as the last two elements of the output tuple as they are
    needed to compute the infidelity.

    If :code:`sample_Uphi=False`, Uphi is NOT sampled.
    """

    Vψ_logfun, Vψ_vars, Vψ_logdistribution = _lazy_apply_UV_to_afun(
        vstate, V_state, extra_hash_data="V"
    )
    Uϕ_logfun, Uϕ_vars, Uϕ_logdistribution = _lazy_apply_UV_to_afun(
        tstate, U_target, extra_hash_data="U"
    )

    Vψ_samples = vstate.samples_distribution(
        Vψ_logdistribution,
        variables=Vψ_vars,
        resample_fraction=resample_fraction,
        chain_name="Vpsi" if V_state is not None else "default",
    )

    afun_t, vars_t, samples_t = None, None, None

    if sample_Uphi:
        afun_t = Uϕ_logfun
        vars_t = Uϕ_vars
        samples_t = tstate.samples_distribution(
            Uϕ_logdistribution,
            variables=Uϕ_vars,
            resample_fraction=resample_fraction,
            chain_name="Uphi" if U_target is not None else "default",
        )
    else:
        afun_t = tstate._apply_fun
        vars_t = tstate.variables
        samples_t = tstate.samples_distribution(
            tstate._model,
            variables=vars_t,
            resample_fraction=resample_fraction,
            chain_name="default",  # should we change this to Uphi
        )

    return (
        Vψ_logfun,
        Vψ_vars,
        _flatten_samples(Vψ_samples),
        Uϕ_logfun,
        Uϕ_vars,
        _flatten_samples(samples_t),
        afun_t,
        vars_t,
    )


@partial(
    jax.jit,
    static_argnames=(
        "afun",
        "afun_t",
        "rw_afun_t",
        "chunk_size_U",
        "chunk_size_V",
        "chunk_size_rw",
        "sample_Uphi",
    ),
)
def smc_kernel(
    afun: Callable,
    vars: PyTree,
    samples: Array,
    afun_t: Callable,
    vars_t: PyTree,
    samples_t: Array,
    rw_afun_t: Callable,
    rw_vars_t: PyTree,
    cv_coeff: Optional[float] = -0.5,
    chunk_size_U: Optional[int] = None,
    chunk_size_V: Optional[int] = None,
    chunk_size_rw: Optional[int] = None,
    sample_Uphi: bool = False,
):
    afun_ = apply_chunked(partial(afun, vars), chunk_size=chunk_size_V)
    afun_t_ = apply_chunked(partial(afun_t, vars_t), chunk_size=chunk_size_U)

    out_shape = samples.shape[:-1]

    # equivalent to .reshape(-1, N)
    samples = _flatten_samples(samples)
    samples_t = _flatten_samples(samples_t)

    logVψ_x = afun_(samples)
    logUϕ_x = afun_t_(samples)
    logVψ_y = afun_(samples_t)
    logUϕ_y = afun_t_(samples_t)

    logRϕψ = logUϕ_x - logVψ_x
    logRψϕ = logVψ_y - logUϕ_y
    logA = logRϕψ + logRψϕ
    A = jnp.exp(logA)

    local_grad = (-1.0 * A).reshape(out_shape)

    if cv_coeff is not None:
        A = A.real + cv_coeff * (jnp.abs(A) ** 2 - 1)

    local_loss = (1 - A).reshape(out_shape)

    if not sample_Uphi:  # sampling from phi, not Uphi, and thus need to reweight
        afun_bare_t_ = apply_chunked(
            partial(rw_afun_t, rw_vars_t), chunk_size=chunk_size_rw
        )
        w_y = jnp.abs(jnp.exp(logUϕ_y - afun_bare_t_(samples_t))) ** 2
        w_y /= mean(w_y)

        local_grad *= w_y
        local_loss *= w_y

    return local_grad, local_loss


@partial(
    jax.jit,
    static_argnames=(
        "afun",
        "afun_t",
        "rw_afun_t",
        "chunk_size_U",
        "chunk_size_V",
        "chunk_size_rw",
        "sample_Uphi",
    ),
)
def cmc_kernel(
    afun: Callable,
    vars: PyTree,
    samples: Array,
    afun_t: Callable,
    vars_t: PyTree,
    samples_t: Array,
    rw_afun_t: Callable,
    rw_vars_t: PyTree,
    cv_coeff: Optional[float] = -0.5,
    chunk_size_U: Optional[int] = None,
    chunk_size_V: Optional[int] = None,
    chunk_size_rw: Optional[int] = None,
    sample_Uphi: bool = False,
):
    afun_ = apply_chunked(partial(afun, vars), chunk_size=chunk_size_V)
    afun_t_ = apply_chunked(partial(afun_t, vars_t), chunk_size=chunk_size_U)

    out_shape = samples.shape[:-1]

    # equivalent to .reshape(-1, N)
    samples = jax.lax.collapse(samples, 0, samples.ndim - 1)
    samples_t = jax.lax.collapse(samples_t, 0, samples_t.ndim - 1)

    logVψ_x = afun_(samples)
    logUϕ_x = afun_t_(samples)
    logVψ_y = afun_(samples_t)
    logUϕ_y = afun_t_(samples_t)

    Rϕψ_x = jnp.exp(logUϕ_x - logVψ_x)
    Rψϕ_y = jnp.exp(logVψ_y - logUϕ_y)

    E = Rψϕ_y
    E2 = jnp.abs(Rψϕ_y) ** 2
    if not sample_Uphi:  # sampling from phi, not Uphi, and thus need to reweight
        afun_bare_t_ = apply_chunked(
            partial(rw_afun_t, rw_vars_t), chunk_size=chunk_size_rw
        )
        w_y = jnp.abs(jnp.exp(logUϕ_y - afun_bare_t_(samples_t))) ** 2
        w_y /= mean(w_y)
        E *= w_y
        E2 *= w_y
    E = mean(E)
    E2 = mean(E2)

    local_grad = (-1.0 * Rϕψ_x * E).reshape(out_shape)

    local_loss = Rϕψ_x * E
    if cv_coeff is not None:
        local_loss = local_loss.real + cv_coeff * (jnp.abs(Rϕψ_x) ** 2 * E2 - 1)
    local_loss = (1 - local_loss).reshape(out_shape)

    return local_grad, local_loss
