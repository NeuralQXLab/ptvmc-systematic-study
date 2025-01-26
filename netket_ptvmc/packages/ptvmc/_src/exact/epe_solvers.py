from .epe_integrator import _split_hamiltonian

import numpy as np
import scipy.sparse as sp

from copy import copy


def LPE1(H, dt):
    I = sp.identity(H.shape[0]).tocsc()
    print("I:", type(I))
    print("H:", type(H))
    Λ = I - 1j * H * dt

    def apply(psi):
        return Λ @ psi

    return apply


def LPE2(H, dt):
    B1 = (1 - 1j) / 2
    B2 = (1 + 1j) / 2

    x = -1j * H

    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x

    def apply(psi):
        return H2 @ (H1 @ psi)

    return apply


def LPE3(H, dt):
    B1 = 0.6265382932707997
    B2 = 0.1867308533646001 - 0.4807738845503311 * 1j
    B3 = 0.1867308533646001 + 0.4807738845503311 * 1j

    x = -1j * H

    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x
    H3 = I + B3 * dt * x

    def apply(psi):
        return H3 @ (H2 @ (H1 @ psi))

    return apply


def LPE4(H, dt):
    B1 = 0.0426266565027024 - 0.3946329531721134 * 1j
    B2 = 0.0426266565027024 + 0.3946329531721134 * 1j
    B3 = 0.4573733434972976 + 0.2351004879985427 * 1j
    B4 = 0.4573733434972976 - 0.2351004879985427 * 1j

    x = -1j * H

    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x
    H3 = I + B3 * dt * x
    H4 = I + B4 * dt * x

    def apply(psi):
        y = H4 @ (H3 @ (H2 @ (H1 @ psi)))
        return y

    return apply


def PPE2(H, dt, neumann_order=10):
    B1 = 1 / 2
    B2 = -1 / 2

    x = -1j * H
    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x
    inv2 = -(B2 * dt * x)

    def apply(psi):
        y = H1 @ psi
        add = copy(y)
        for _ in range(neumann_order):
            add = inv2 @ add
            y = y + add
        return y

    return apply


def PPE4(H, dt, neumann_order=10):
    B1 = 1 / 12 * (3 - 1j * np.sqrt(3))
    B2 = -(1j / 12) * (-3j + np.sqrt(3))
    B3 = 1 / 12 * (3 + 1j * np.sqrt(3))
    B4 = 1j / 12 * (3j + np.sqrt(3))

    x = -1j * H

    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x
    H3 = I + B3 * dt * x

    inv2 = -(B2 * dt * x)
    inv4 = -(B4 * dt * x)

    def apply(psi):
        y = H1 @ psi

        add = copy(y)
        for _ in range(neumann_order):
            add = inv2 @ add
            y = y + add

        y = H3 @ y

        add = copy(y)
        for _ in range(neumann_order):
            add = inv4 @ add
            y = y + add

        return y

    return apply


def SLPE1(H, dt):
    A1 = 1
    B1 = 1

    Ho, Hd = _split_hamiltonian(H)
    x = -1j * Ho
    z = -1j * Hd

    D1 = sp.diags(np.exp(z * dt * A1))

    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x

    def apply(psi):
        return H1 @ (D1 @ psi)

    return apply


def SLPE2(H, dt):
    A1 = (1 - 1j) / 2
    A2 = (1 + 1j) / 2
    B1 = (1 - 1j) / 2
    B2 = (1 + 1j) / 2

    Ho, Hd = _split_hamiltonian(H)
    x = -1j * Ho
    z = -1j * Hd

    D1 = sp.diags(np.exp(z * dt * A1))
    D2 = sp.diags(np.exp(z * dt * A2))

    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x

    def apply(psi):
        # TODO: compare scaling with D2 @ (H2 @ (D1 @ (H1 @ psi)))
        return H2 @ (D2 @ (H1 @ (D1 @ psi)))

    return apply


def SLPE3(H, dt):
    A1 = 0.1056624327025936 - 0.3943375672974064 * 1j
    A2 = 0.3943375672974064 + 0.1056624327025936 * 1j
    A3 = 0.3943375672974064 - 0.1056624327025936 * 1j
    A4 = 0.1056624327025936 + 0.3943375672974064 * 1j

    B1 = 0.1056624327025936 - 0.3943375672974064 * 1j
    B2 = 0.3943375672974064 + 0.1056624327025936 * 1j
    B3 = 0.3943375672974064 - 0.1056624327025936 * 1j
    B4 = 0.1056624327025936 + 0.3943375672974064 * 1j

    Ho, Hd = _split_hamiltonian(H)
    x = -1j * Ho
    z = -1j * Hd

    D1 = sp.diags(np.exp(z * dt * A1))
    D2 = sp.diags(np.exp(z * dt * A2))
    D3 = sp.diags(np.exp(z * dt * A3))
    D4 = sp.diags(np.exp(z * dt * A4))

    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x
    H3 = I + B3 * dt * x
    H4 = I + B4 * dt * x

    def apply(psi):
        y = H1 @ (D1 @ psi)
        y = H2 @ (D2 @ y)
        y = H3 @ (D3 @ y)
        y = H4 @ (D4 @ y)
        return y

    return apply


def SPPE2(H, dt, neumann_order=10):
    A1 = 1 / 2
    A2 = 1 / 2
    B1 = 1 / 2
    B2 = -1 / 2

    Ho, Hd = _split_hamiltonian(H)
    x = -1j * Ho
    z = -1j * Hd

    D1 = sp.diags(np.exp(z * dt * A1))
    D2 = sp.diags(np.exp(z * dt * A2))

    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x
    inv2 = -(B2 * dt * x)

    def apply(psi):
        y = H1 @ (D1 @ psi)

        add = copy(y)
        for _ in range(neumann_order):
            add = inv2 @ add
            y = y + add

        y = D2 @ y
        return y

    return apply


def SPPE3(H, dt, neumann_order=10):
    A1 = (3 - 1j * np.sqrt(3)) / 12
    A2 = 1 / 2
    A3 = (3 + 1j * np.sqrt(3)) / 12

    B1 = (3 - 1j * np.sqrt(3)) / 12
    B2 = 1j / 12 * (3j + np.sqrt(3))
    B3 = (3 + 1j * np.sqrt(3)) / 12
    B4 = -1j / 12 * (-3j + np.sqrt(3))

    Ho, Hd = _split_hamiltonian(H)
    x = -1j * Ho
    z = -1j * Hd

    D1 = sp.diags(np.exp(z * dt * A1))
    D2 = sp.diags(np.exp(z * dt * A2))
    D3 = sp.diags(np.exp(z * dt * A3))

    I = sp.identity(H.shape[0]).tocsc()
    H1 = I + B1 * dt * x
    H3 = I + B3 * dt * x

    inv2 = -(B2 * dt * x)
    inv4 = -(B4 * dt * x)

    def apply(psi):
        y = H1 @ (D1 @ psi)

        add = copy(y)
        for _ in range(neumann_order):
            add = inv2 @ add
            y = y + add

        y = H3 @ (D2 @ y)

        add = copy(y)
        for _ in range(neumann_order):
            add = inv4 @ add
            y = y + add

        y = D3 @ y
        return y

    return apply
