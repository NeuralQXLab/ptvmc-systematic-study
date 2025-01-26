import netket as nk
from netket.operator.spin import sigmax, sigmaz

from advanced_drivers.driver import InfidelityOptimizerNG

import qutip as qt
import matplotlib.pyplot as plt

import jax.numpy as jnp
import optax

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
ma = nk.models.LogStateVector(hilbert=hi, param_dtype=float)
sa = nk.sampler.ExactSampler(
    hilbert=hi,
)
n_samples = int(2**10)
vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
ts = nk.vqs.MCState(sa, ma, n_samples=n_samples)

_, psi_gs = gs_qo = ha.to_qobj().groundstate()
ts.parameters = {"logstate": jnp.log(jnp.asarray(psi_gs.full())).ravel()}

n_iter = 300
diag_shift = 1e-7
optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=5e-2)
linear_solver_fn = nk.optimizer.solver.cholesky

logger = nk.logging.RuntimeLog()
driver = InfidelityOptimizerNG(
    target_state=ts,
    optimizer=optimizer,
    linear_solver_fn=linear_solver_fn,
    diag_shift=diag_shift,
    variational_state=vs,
    sample_Uphi=True,
    cv_coeff=-0.5,
)
driver.run(
    n_iter=n_iter,
    out=logger,
)

fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.5))
ax.plot(
    logger["Infidelity"]["Mean"].real,
)
ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("Infidelity")
plt.savefig("infidelity_optimization.pdf", bbox_inches="tight")
plt.show()
