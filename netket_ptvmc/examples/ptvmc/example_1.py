import netket as nk
import advanced_drivers as advd
from jax import numpy as jnp
import jax
import numpy as np

import ptvmc

# 1D Lattice
L = 4

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=False)

# Ising spin hamiltonian
ha = nk.operator.IsingJax(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, use_visible_bias=True, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.02)

# Variational State
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

compression_alg = ptvmc.compression.InfidelityCompression(
    advd.driver.InfidelityOptimizerNG,
    build_parameters={"diag_shift": 0.01, "optimizer": op},
    run_parameters={"n_iter": 10},
)

# better
compression_alg = ptvmc.compression.InfidelityNGCompression(
    diag_shift=1e-4,
    auto_diag_shift=True,
    learning_rate=0.03,
    max_iters=100,
    target_infidelity=1e-4,
)
# solver
solver = ptvmc.solver.LPE2()

# integrator
integration_params = ptvmc.IntegrationParameters(
    dt=0.1,
)
from copy import copy

vs_init = copy(vs)
integrator = ptvmc._src.integrator.integrator.Integrator(
    ha, solver, compression_alg, t0=0, y0=vs, parameters=integration_params
)
success, infos = integrator.step()
