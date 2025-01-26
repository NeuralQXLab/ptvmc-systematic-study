import qutip as qt
import numpy as np
import ptvmc

from functools import partial

import advanced_drivers as advd
from advanced_drivers._src.callbacks.autodiagshift import PI_controller_diagshift

from matplotlib import pyplot as plt

import optax

import netket as nk
from netket.optimizer.solver import cholesky
from netket.operator.spin import sigmax, sigmaz

import ptvmc._src.exact.epe_integrator
import ptvmc._src.exact.epe_solvers
from ptvmc._src.callbacks.logapply import DynamicLogApply as LogApply


# 2D Lattice
L = 2
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=False)

# Ising spin hamiltonian
J = 1
hc = 3.044 * J
h = 2 * hc
ha = sum([-J * sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges()])
ha += sum([-h * sigmax(hi, i) for i in g.nodes()])

# LogState Spin Machine
ma = nk.models.LogStateVector(hilbert=hi, param_dtype=complex)
ma = ptvmc.nn.DiagonalWrapper(ma, param_dtype=complex)
# nk.models.RBM(alpha=1, use_visible_bias=True, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.ExactSampler(
    hilbert=hi,
)
# nk.sampler.MetropolisLocal(hi, n_chains=16)

# Variational State
vs = nk.vqs.MCState(sa, ma, n_samples=1024)

mx = sum([sigmax(hi, i) for i in g.nodes()]) / g.n_nodes

# Exact simulation
dt = 0.05
T = 0.3

times_qo = np.linspace(0, T, 100)
qo_sol = qt.sesolve(
    ha.to_qobj(),
    vs.to_qobj(),
    times_qo,
    e_ops=[
        mx.to_qobj(),
    ],
    options={
        "progress_bar": True,
    },
)
times_qo = qo_sol.times
mx_vals_qo = qo_sol.expect[0]

# State-vector PE simulation
f_apply = partial(ptvmc._src.exact.epe_solvers.SPPE3, ha.to_sparse())
_, expect_epe = ptvmc._src.exact.epe_integrator.integrator(
    vs.to_array(),
    [
        mx.to_sparse(),
    ],
    T,
    dt,
    f_apply,
)
mx_vals_epe = expect_epe[:, 0].real
times_epe = np.arange(0, T + dt, dt)

# Advanced compression
compression_alg = ptvmc.compression.InfidelityCompression(
    driver_class=advd.driver.InfidelityOptimizerNG,
    build_parameters={
        "diag_shift": 1e-6,
        "optimizer": optax.inject_hyperparams(optax.sgd)(learning_rate=0.05),
        "linear_solver_fn": cholesky,
        "proj_reg": None,
        "momentum": None,
        "chunk_size_bwd": None,
        "collect_quadratic_model": True,
        "use_ntk": False,
        "rloo": False,
        "cv_coeff": -0.5,
        "resample_fraction": None,
        "estimator": "cmc",
    },
    run_parameters={
        "n_iter": 100,
        "callback": [
            PI_controller_diagshift(
                target=0.75,
                safety_fac=0.9,
                clip_min=0.5,
                clip_max=2,
                diag_shift_min=1e-11,
                diag_shift_max=0.1,
                order=2,
                beta_1=1,
                beta_2=0.1,
            )
        ],
    },
)

# Driver
solver = ptvmc.solver.SPPE3()

integration_params = ptvmc.IntegrationParameters(
    dt=dt,
)
generator = -1j * ha
driver = ptvmc.PTVMCDriver(
    generator,
    0.0,
    solver=solver,
    integration_params=integration_params,
    compression_algorithm=compression_alg,
    variational_state=vs,
)


def plot_fn(logger):
    times_var = logger["times"]["value"]
    mx_vals_var = logger["mx"]["Mean"].real

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    ax.plot(times_var, mx_vals_var, "o-", c="tab:blue", label="Variational")
    ax.plot(times_qo, mx_vals_qo, ls="--", c="tab:orange", label="Exact")
    ax.plot(
        times_epe,
        mx_vals_epe,
        "o",
        markerfacecolor="none",
        markeredgecolor="tab:red",
        label="EPE",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$M_x$")
    ax.legend()
    fig.savefig("Mx_dynamic.pdf")


# Run-time logger
logger = nk.logging.RuntimeLog()

callback = [
    LogApply(logger, plot_fn),
]

(out,) = driver.run(
    T=T,
    out=logger,
    obs={"mx": mx},
    obs_in_fullsum=True,
    callback=callback,
)

times_var = out["times"]["value"]
mx_vals_var = out["mx"]["Mean"].real

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ax.plot(times_var, mx_vals_var, "o-", c="tab:blue", label="Variational")
ax.plot(times_qo, mx_vals_qo, ls="--", c="tab:orange", label="Exact")
ax.plot(
    times_epe,
    mx_vals_epe,
    "o",
    markerfacecolor="none",
    markeredgecolor="tab:red",
    label="EPE",
)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$M_x$")
ax.legend()
fig.savefig("Mx.pdf")
