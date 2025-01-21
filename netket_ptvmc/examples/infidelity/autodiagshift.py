import netket as nk
import netket_pro as nkp
import advanced_drivers as advd

from matplotlib import pyplot as plt
import numpy as np

import optax


# Let's generate the ground state of the TFIM model. This
# technique also works with excited states.
g = nk.graph.Square(4)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hi, graph=g, h=1.0)

# Here is the ground state
egs, psi0 = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
psi0 = psi0[:, 0]

sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Let's build the target variational state as a LogStateVector, which
# has 2^N parameters encoding the log-wavefunction
ma_target = nk.models.LogStateVector(hi)
vs_target = nk.vqs.MCState(sa, ma_target)

# Compute the log-wavefunction. As some numbers can be negative, we
# convert the wavefunciton to complex beforehand
log_psi0 = np.log(psi0 + 0j)

# And fix the 'parameters' that represent the target state
vs_target.parameters = {
    "logstate": vs_target.parameters["logstate"].at[:].set(log_psi0)
}

# Now construct the variational state we are going to train
ma = nk.models.RBM()
vs = nk.vqs.MCState(sa, ma)

# or just use the Infidelity optimisation driver
# optimizer = nk.optimizer.Sgd(learning_rate=0.05)
optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=0.05)
autodiagshift = advd.callbacks.PI_controller_diagshift(
    diag_shift_max=0.0001,
)

driver = advd.driver.InfidelityOptimizerNG(
    vs_target,
    optimizer,
    variational_state=vs,
    diag_shift=0.0001,
)

log = nk.logging.RuntimeLog()

driver.run(
    600,
    out=log,
    callback=autodiagshift,
)

# plt.ion()
plt.semilogy(log.data["Infidelity"].iters, log.data["Infidelity"])
plt.show()
