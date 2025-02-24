from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import tree

from flax import core as fcore

import netket as nk
from netket.utils import struct

from advanced_drivers._src.callbacks.base import AbstractCallback


@partial(
    jax.jit,
    static_argnames=("afun"),
)
def _compute_discrepancies(
    lr,
    h,
    linear_term,
    quadratic_term,
    afun,
    vars,
    σ,
    updated_vars,
    updated_loss,
):
    reweighting_coefficient = (
        jnp.abs(jnp.exp(afun(updated_vars, σ) - afun(vars, σ))) ** 2
    )
    updated_loss *= reweighting_coefficient

    norm_scaling = 1 / nk.stats.mean(reweighting_coefficient)

    updated_h = (nk.stats.mean(updated_loss) * norm_scaling).real

    M = h - lr * linear_term + lr**2 / 2 * quadratic_term

    ξ = jnp.abs((updated_h - M) / (updated_h + M))
    ρ = (updated_h - h) / (M - h)
    return M, ξ, ρ, updated_h


def assert_learning_rate_reachable(driver):
    if not hasattr(driver._optimizer_state, "hyperparams"):
        raise ValueError(
            r"""
            Optimizer state does not have hyperparams. To change the learning rate the optimizer should be
            defined as `optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=lr)`.
            """
        )


class PI_controller_diagshift(AbstractCallback):
    target: float = struct.field(pytree_node=False, default=0.75, serialize=False)
    safety_fac: float = struct.field(pytree_node=False, default=0.9, serialize=False)
    clip_min: float = struct.field(pytree_node=False, default=0.5, serialize=False)
    clip_max: float = struct.field(pytree_node=False, default=2, serialize=False)
    diag_shift_min: float = struct.field(
        pytree_node=False, default=1e-11, serialize=False
    )
    diag_shift_max: float = struct.field(
        pytree_node=False, default=1e-1, serialize=False
    )
    order: int = struct.field(pytree_node=False, default=2, serialize=False)
    beta_1: float = struct.field(pytree_node=False, default=1.0, serialize=False)
    beta_2: float = struct.field(pytree_node=False, default=0.0, serialize=False)

    _multiplier: float = struct.field(pytree_node=False, default=1, serialize=False)
    _old_mutiplier: float = struct.field(
        pytree_node=False, default=None, serialize=False
    )

    _max_step_attempts: int = struct.field(
        pytree_node=False, default=10, serialize=False
    )
    _reduction_bound_low: float = struct.field(
        pytree_node=False, default=0.1, serialize=False
    )
    _reduction_bound_high: float = struct.field(
        pytree_node=False, default=3, serialize=False
    )

    def __init__(
        self,
        target: float = 0.75,
        safety_fac: float = 0.9,
        clip_min: float = 0.5,
        clip_max: float = 2,
        diag_shift_min: float = 1e-11,
        diag_shift_max: float = 1e-1,
        order: int = 2,
        beta_1: float = 1.0,
        beta_2: float = 0.00,
        max_step_attempts: int = 10,
        reduction_bound_low: float = 0.1,
        reduction_bound_high: float = 3,
    ):
        self.target = target
        self.safety_fac = safety_fac
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.diag_shift_min = diag_shift_min
        self.diag_shift_max = diag_shift_max
        self.order = order
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self._old_mutiplier = 1
        self._multiplier = 1

        self._max_step_attempts = max_step_attempts
        self._reduction_bound_high = reduction_bound_high
        self._reduction_bound_low = reduction_bound_low

    def on_run_start(self, step, driver, callbacks):
        assert_learning_rate_reachable(driver)
        if not hasattr(driver, "collect_quadratic_model"):
            raise TypeError(
                "This callback requires the driver to have the `collect_quadratic_model` flag."
            )
        driver.collect_quadratic_model = True

    def on_compute_update_end(self, step, log_data, driver):
        α = driver._optimizer_state.hyperparams["learning_rate"]
        if callable(α):
            α = α(step)

        h = driver._loss_stats.mean.real
        linear_term = jnp.asarray(driver.info["linear_term"])
        quadratic_term = jnp.asarray(driver.info["quadratic_term"])

        afun, vars, σ, extra_args = driver._get_local_estimators_kernel_args()

        model_state, params = fcore.pop(vars, "params")
        δθ = tree.map(lambda x: -α * x, driver._dp)
        updated_pars = tree.map(jnp.add, params, δθ)
        updated_vars = {"params": updated_pars, **model_state}

        # updated_local_loss = driver.local_estimators(parameters=updated_pars)
        _, updated_loss = driver._kernel(
            afun,
            updated_vars,
            σ,
            *extra_args,
        )

        M, ξ, ρ, updated_h = _compute_discrepancies(
            lr=α,
            h=h,
            linear_term=linear_term,
            quadratic_term=quadratic_term,
            afun=afun,
            vars=vars,
            σ=σ,
            updated_vars=updated_vars,
            updated_loss=updated_loss,
        )

        bare_multiplier = (self.target - ρ) * np.abs(self.target - ρ) ** self.order + 1
        clipped_multilier = jnp.clip(bare_multiplier.real, self.clip_min, self.clip_max)
        self._multiplier = self.safety_fac * clipped_multilier

        self._multiplier = (
            self._multiplier**self.beta_1 / self._old_mutiplier**self.beta_2
        )

        driver.diag_shift = jnp.minimum(
            jnp.maximum(driver.diag_shift * self._multiplier, self.diag_shift_min),
            self.diag_shift_max,
        )

        fails = {
            "h_is_nan": {
                "condition": not np.isfinite(h),
                "message": "Loss at current step is NaN.",
            },
            "updated_h_is_nan": {
                "condition": not np.isfinite(updated_h),
                "message": f"Predicted loss is NaN. This is likely due to the geometric tensor or the gradient. Indeed the linear term is {linear_term} and the quadratic term is {quadratic_term}.",
            },
            "rho_out_of_bounds": {
                "condition": ρ < self._reduction_bound_low
                or ρ > self._reduction_bound_high,
                "message": f"ρ is out of bounds. The current value is {ρ}. The bounds are [{self._reduction_bound_low},{self._reduction_bound_high}].",
            },
        }

        epic_fails = ["h_is_nan", "updated_h_is_nan"]

        for key, err in fails.items():
            if err["condition"]:
                driver._reject_step = True

                if key in epic_fails:
                    driver.reset_step(hard=True)

                    print(err["message"], "\n")

        if driver._step_attempt > self._max_step_attempts:
            driver._stop_run = True

        log_data["adaptive_diagshift"] = {
            "M": M,
            "xi": ξ,
            "rho": ρ,
            "updated_h": updated_h,
            "multiplier": self._multiplier,
            "diag_shift": driver.diag_shift,
            "lr": α,
        }

    def on_step_end(self, step, log_data, driver):
        self._old_mutiplier = self._multiplier

        # TODO: move this to another method, because we want to log this information
        # before on_legacy_log is called.
        if log_data is not None:
            log_data["info"] = driver.info

        dp = driver._dp
        log_data["constrained_norm"] = nk.jax.tree_dot(nk.jax.tree_conj(dp), dp)
        log_data["rejected_steps"] = driver._step_attempt
