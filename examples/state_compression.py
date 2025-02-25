import netket as nk
import matplotlib.pyplot as plt

import optax
import jax.numpy as jnp

from netket.models import RBM

from advanced_drivers.driver import InfidelityOptimizerNG

# Simple 1D Lattice
L = 4
lattice = nk.graph.Chain(L, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)

# Define the U, V operators of the compression
dt = 0.01
H  = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
U  = (0.5 + 0.5j) * H * dt
V  = None

# Define some target state. We find it by exact computation of the ground state
gs = jnp.array(
    [
        0.07664074, 0.1767767, 0.1767767, 0.13529903, 
        0.1767767, 0.57664074, 0.13529903, 0.1767767, 
        0.1767767, 0.13529903, 0.57664074, 0.1767767,
        0.13529903, 0.1767767, 0.1767767, 0.07664074,
    ]
)

# Sampler
sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16,)
   
# Target model 
target_model = nk.models.LogStateVector(hi, param_dtype=float)
tstate = nk.vqs.MCState(
    sampler=sampler,
    model=target_model,
    n_samples=512,
    n_discard_per_chain=0,
    seed=1,
    sampler_seed=1,
)
tstate.parameters = {"logstate": jnp.log(gs)}

# Variational model
model = RBM(param_dtype=jnp.complex128,)

vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=512,
    n_discard_per_chain=0,
    seed=0,
    sampler_seed=0,
)

# Optimization 
diag_shift = 1e-6
optimizer = optax.sgd(learning_rate=0.05)
solver_fn = nk.optimizer.solver.cholesky

gs = InfidelityOptimizerNG(
        target_state=tstate,
        optimizer=optimizer,
        variational_state=vstate,
        diag_shift=diag_shift,
        linear_solver_fn=solver_fn,
        use_ntk=False,
        on_the_fly=False,
        U=U,
        V=V,
        cv_coeff=-0.5,
    )
logger = nk.logging.RuntimeLog()
gs.run(n_iter=120, out=logger)

# Plot
infidelity = logger["Infidelity"]['Mean']
fig, ax = plt.subplots(1,1, figsize=(4,2))
ax.plot(infidelity, 'o-')
ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Infidelity')
fig.savefig("infidelity.pdf", bbox_inches="tight")