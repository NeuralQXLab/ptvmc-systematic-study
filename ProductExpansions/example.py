import numpy as np
import qutip as qt
import netket as nk
import matplotlib.pyplot as plt

from functools import partial
from netket.operator.spin import sigmax, sigmaz
from algorithms import integrator, LPE3


def _setup(L, J, h):
    g = nk.graph.Square(L, pbc=True)
    N = g.n_nodes
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    H = sum([-J * sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges()])
    H += sum([-h * sigmax(hi, i) for i in g.nodes()])
    H = H.to_sparse().tocsc()

    Mx = sum([sigmax(hi, i) for i in g.nodes()]) / N
    e_ops = [
        Mx,
    ]
    e_ops = [op.to_sparse() for op in e_ops]

    gs = np.ones(hi.n_states, dtype=complex)
    gs /= np.linalg.norm(gs)
    return gs, H, e_ops


def convert_to_qobj(gs, H, e_ops):
    N = int(np.log2(H.shape[0]))
    H_qo = qt.Qobj(H, dims=[[2] * N, [2] * N])
    e_ops_qo = [qt.Qobj(e, dims=H_qo.dims) for e in e_ops]

    gs_qo = qt.Qobj(gs, dims=[[2] * N, [1]])
    return gs_qo, H_qo, e_ops_qo


if __name__ == "__main__":
    L = 3
    tf = 2.0
    dt = 5e-3
    algorithm = LPE3

    gs, H, e_ops = _setup(L, 1, 3.044 / 10)
    algorithm = partial(
        algorithm,
        H=H,
    )

    gs_qo, H_qo, e_ops_qo = convert_to_qobj(gs, H, e_ops)

    t_l = np.arange(0, tf + dt, dt)
    sol = qt.sesolve(
        H_qo,
        gs_qo,
        t_l,
        e_ops=e_ops_qo,
        options={"store_states": False},
    )

    psi_array, times_array, e_ops_array = integrator(gs, e_ops, tf, dt, algorithm)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    Mx = sol.expect[0]
    Mx_PE = e_ops_array[:, 0]

    ax.plot(t_l, Mx, c="navy", lw=2, label="Exact")
    ax.plot(times_array, Mx_PE, "--", lw=2, c="tab:orange", label="PE")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\langle M_x \rangle$")
    ax.legend()
    fig.savefig("Mx.pdf", bbox_inches="tight")
