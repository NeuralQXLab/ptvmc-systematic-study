import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

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
L = 3
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
ma = nk.models.LogStateVector(hilbert=hi)
# nk.models.RBM(alpha=5, use_visible_bias=True, param_dtype=complex)

# Monte Carlo Sampling
sa = nk.sampler.ExactSampler(
    hilbert=hi,
)
# nk.sampler.MetropolisLocal(hi, n_chains=16)

# Variational State
vs = nk.vqs.MCState(sa, ma, n_samples=1024)

mx = sum([sigmax(hi, i) for i in g.nodes()]) / g.n_nodes

# Exact simulation
dt = 0.025
T = 1.

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
times_qo = np.asarray(qo_sol.times)
mx_vals_qo = qo_sol.expect[0]

# State-vector PE simulation
f_apply = partial(ptvmc._src.exact.epe_solvers.PPE4, ha.to_sparse())
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

# This function is called recursively as a callback to update the plot and give real-time information about the simulation
def plot_fn(logger):
    times_var = logger["times"]["value"]
    mx_vals_var = logger["mx"]["Mean"].real

    fig, ax = plt.subplots(2, 1, figsize=(4, 3), gridspec_kw={"height_ratios": [1.5, 1]})
    ax[0].plot(h * times_qo, mx_vals_qo, ls="-", c="darkgrey", label="Exact",)
    ax[0].plot(h * times_epe, mx_vals_epe, "o",  ms=5.5, markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.05, label=solver.__repr__()[:-2],)
    ax[0].plot(h * times_var, mx_vals_var, "o", ms=4, c="steelblue", label="Var. " + solver.__repr__()[:-2],)
    ax[0].set_xticklabels([])
    
    ax[0].set_ylabel(r"$M_x$")
    ax[0].legend(ncol=3, loc=(-0.014,1.02), )
    
    color = ["#5DC350", "#986746"]
    final_infidelities = logger.data['Infidelity']['Mean']['value']['value'].real[:, :, -1]
    for i in range(final_infidelities.shape[1]):
        ax[1].plot(h * times_var, final_infidelities[:, i], "o-", c=color[i],  ms=6, label=f"stage {i}", markeredgecolor="k", markeredgewidth=0.4,)
    ax[1].set_yscale("log")    
    ax[1].set_ylabel(r"$\mathcal{I}$")
    ax[1].set_xlabel(r"$ht$")
    
    ax[0].set_xlim(-0.04*h, h*T*1.05)
    ax[1].set_xlim(-0.04*h, h*T*1.05)
    ax[1].set_ylim(1e-12, 1e-6)
    
    fig.savefig("Mx_dynamic.pdf", bbox_inches="tight")
    plt.close(fig)

# Define compression algorithm
compression_alg = ptvmc.compression.InfidelityCompression(
    driver_class=advd.driver.InfidelityOptimizerNG,
    build_parameters={
        "diag_shift": 1e-7,
        "optimizer": optax.inject_hyperparams(optax.sgd)(learning_rate=0.1),
        "linear_solver_fn": cholesky,
        "proj_reg": None,
        "momentum": None,
        "chunk_size_bwd": None,
        "collect_quadratic_model": True,
        "use_ntk": False,
        "cv_coeff": -0.5,
        "resample_fraction": None,
        "estimator": "cmc",
    },
    run_parameters={
        "n_iter": 50,
        "callback": [
            PI_controller_diagshift(
                target=0.9,
                safety_fac=1.,
                clip_min=0.5,
                clip_max=2,
                diag_shift_min=1e-11,
                diag_shift_max=0.1,
                order=1,
                beta_1=0.9,
                beta_2=0.1,
            )
        ],
    },
)

# Discretization scheme used to approximate the time-evolution operator
solver = ptvmc.solver.PPE4()

# Define the PTVMC driver
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
    save_path="states/",
    save_every=1,
)