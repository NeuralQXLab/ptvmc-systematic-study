import jax.numpy as jnp
import jax
from functools import partial

from flax import core as fcore

from netket import jax as nkjax
from netket.vqs import FullSumState, expect, expect_and_grad
from netket.utils import mpi
from netket.stats import Stats

from netket_pro.utils import make_logpsi_U_afun

from .operator import InfidelityUVOperator


def sparsify(U):
    return U.to_sparse()


def _prepare(vstate, operator, extra_hash_data=None):
    ψ_logfun = vstate._apply_fun
    ψ_vars = vstate.variables

    if operator is None:
        return ψ_logfun, ψ_vars, vstate.model
    else:
        Uψ_logfun, Uψ_vars = make_logpsi_U_afun(ψ_logfun, operator, ψ_vars)

    # fix the has to include forward or backward info
    # we use this to give a different hash to 'forward' and 'backward'
    # distributions, even if they are identical, and only differ in the
    # parameters.
    if extra_hash_data is not None:
        Uψ_logfun.__hash__()
        Uψ_logfun._hash = hash((Uψ_logfun._hash, extra_hash_data))

    return Uψ_logfun, Uψ_vars, Uψ_logfun


@expect.dispatch
def infidelity(vstate: FullSumState, op: InfidelityUVOperator):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")
    if not isinstance(op.target, FullSumState):
        raise TypeError("Can only compute infidelity of exact states.")

    ϕ = op.target.to_array(normalize=False)
    if op.U_target is not None:
        U_num_sp = sparsify(op.U_target)
        Vϕ = U_num_sp @ ϕ
    else:
        Vϕ = ϕ

    Uψ_logfun, Uψ_vars, _ = _prepare(vstate, op.V_state)

    return infidelity_sampling_FullSumState(
        Uψ_logfun,
        Uψ_vars,
        vstate._all_states,
        Vϕ,
        return_grad=False,
    )


@expect_and_grad.dispatch
def infidelity(  # noqa: F811
    vstate: FullSumState,
    op: InfidelityUVOperator,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")
    if not isinstance(op.target, FullSumState):
        raise TypeError("Can only compute infidelity of exact states.")

    ϕ = op.target.to_array(normalize=False)
    if op.U_target is not None:
        U_num_sp = sparsify(op.U_target)
        Vϕ = U_num_sp @ ϕ
    else:
        Vϕ = ϕ

    Uψ_logfun, Uψ_vars, _ = _prepare(vstate, op.V_state)

    return infidelity_sampling_FullSumState(
        Uψ_logfun,
        Uψ_vars,
        vstate._all_states,
        Vϕ,
        return_grad=True,
    )


@partial(jax.jit, static_argnames=("afun", "return_grad"))
def infidelity_sampling_FullSumState(
    afun,
    variables,
    sigma,
    Ustate_t,
    return_grad,
):
    model_state, params = fcore.pop(variables, "params")

    def expect_fun(params):
        state = jnp.exp(afun({"params": params, **model_state}, sigma))
        state = state / jnp.sqrt(jnp.sum(jnp.abs(state) ** 2))
        return jnp.abs(state.T.conj().T @ Ustate_t) ** 2 / jnp.abs(
            Ustate_t.conj().T @ Ustate_t
        )

    if not return_grad:
        F = expect_fun(params)
        return Stats(mean=1 - F, error_of_mean=0.0, variance=0.0)

    F, F_vjp_fun = nkjax.vjp(expect_fun, params, conjugate=True)

    F_grad = F_vjp_fun(jnp.ones_like(F))[0]
    F_grad = jax.tree_util.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], F_grad)
    I_grad = jax.tree_util.tree_map(lambda x: -x, F_grad)
    I_stats = Stats(mean=1 - F, error_of_mean=0.0, variance=0.0)

    return I_stats, I_grad
