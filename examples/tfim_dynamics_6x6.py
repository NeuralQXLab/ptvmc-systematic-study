import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import ptvmc

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


# This function is called recursively as a callback to update the plot and give real-time information about the simulation
def plot_fn(logger):
    times_var = logger["times"]["value"]
    mx_vals_var = logger["mx"]["Mean"].real

    fig, ax = plt.subplots(2, 1, figsize=(4, 3), gridspec_kw={"height_ratios": [1.5, 1]})
    ax[0].plot(times_var, mx_vals_var, "o", ms=4, c="steelblue", label="Var. " + solver.__repr__()[:-2],)
    ax[0].set_xticklabels([])
    
    ax[0].set_ylabel(r"$M_x$")
    ax[0].legend()
    
    final_infidelities = logger.data['Infidelity']['Mean']['value']['value'].real[:, :, -1]
    for i in range(final_infidelities.shape[1]):
        ax[1].plot(times_var, final_infidelities[:, i], "o-",  ms=6, label=f"stage {i}", markeredgecolor="k", markeredgewidth=0.4,)
    ax[1].set_yscale("log")    
    ax[1].set_ylabel(r"$\mathcal{I}$")
    ax[1].set_xlabel(r"$ht$")
    
    ax[0].set_xlim(-0.04, T*1.05)
    ax[1].set_xlim(-0.04, T*1.05)
    
    fig.savefig("Mx_dynamic.pdf", bbox_inches="tight")
    plt.close(fig)

# 2D Lattice
L = 6
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=False)

# Ising spin hamiltonian
J = 1
hc = 3.044 * J
h = hc / 10
ha = sum([-J * sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges()])
ha += sum([-h * sigmax(hi, i) for i in g.nodes()])
mx = sum([sigmax(hi, i) for i in g.nodes()]) / g.n_nodes
ha = ha.to_jax_operator()
mx = mx.to_jax_operator()

dt = 0.025
T = 2.

# Spin Machine
num_layers = 3
heads = 12
d_model = int(heads * 6)
b = 2
expansion_factor = 2
transl_invariant = True

ma = ptvmc.nets.ViT(
    num_layers = num_layers,
    d_model = d_model,
    heads = heads,
    b = b,
    L = L,
    expansion_factor = expansion_factor,
    transl_invariant = transl_invariant,
)
ma = ptvmc.nn.DiagonalWrapper(ma, param_dtype=complex)

# Monte Carlo Sampling
n_samples = int(2**14)
sa = nk.sampler.MetropolisLocal(hi, n_chains=n_samples, sweep_size=hi.size//2)

# Variational State
vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)

ha_gs = sum([- sigmax(hi, i) for i in g.nodes()])
ha_gs = ha_gs.to_jax_operator()

init_logger = nk.logging.RuntimeLog()
init_driver = advd.driver.VMC_NG(
    hamiltonian=ha_gs,
    variational_state = vs,
    optimizer = optax.sgd(0.05),
    diag_shift = 1e-6,
    linear_solver_fn = cholesky,
    use_ntk = True, 
    on_the_fly = True,
)
init_driver.run(n_iter=150, out=init_logger)

# Define compression algorithm
compression_alg = ptvmc.compression.InfidelityCompression(
    driver_class=advd.driver.InfidelityOptimizerNG,
    build_parameters={
        "diag_shift": 1e-5,
        "optimizer": optax.inject_hyperparams(optax.sgd)(learning_rate=0.05),
        "linear_solver_fn": cholesky,
        "proj_reg": None,
        "momentum": None,
        "chunk_size_bwd": None,
        "collect_quadratic_model": True,
        "use_ntk": True,
        "on_the_fly": True,
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
solver = ptvmc.solver.SLPE3()

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
    obs_in_fullsum=False,
    callback=callback,
    save_path="states/",
    save_every=1,
)