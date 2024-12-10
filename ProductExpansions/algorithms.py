import numpy as np
import scipy.sparse as sp

from copy import copy
from tqdm import tqdm
from typing import Callable, List, Tuple

from utils import _split_hamiltonian

def identity(H):
    return sp.identity(H.shape[0]).tocsc() 

def LPE1(dt, H):
    I = identity(H)
    Λ = I - 1j * H * dt
    
    def apply(psi):
        return Λ @ psi
    
    return apply


def LPE2(dt, H):
    B1 = (1 - 1j)/2
    B2 = (1 + 1j)/2
    
    I = identity(H)
    x = - 1j * H
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x
    
    def apply(psi):
        return H2 @ (H1 @ psi)
    
    return apply


def LPE3(dt, H):
    B1 = 0.6265382932707997
    B2 = 0.1867308533646001 - 0.4807738845503311 * 1j
    B3 = 0.1867308533646001 + 0.4807738845503311 * 1j
    
    I = identity(H)
    x = - 1j * H
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x
    H3 = I + B3 * dt * x
    
    def apply(psi):
        return H3 @ (H2 @ (H1 @ psi))
    
    return apply


def LPE4(dt, H):
    B1 = 0.0426266565027024 - 0.3946329531721134 * 1j
    B2 = 0.0426266565027024 + 0.3946329531721134 * 1j
    B3 = 0.4573733434972976 + 0.2351004879985427 * 1j
    B4 = 0.4573733434972976 - 0.2351004879985427 * 1j
    
    I = identity(H)
    x = - 1j * H
    
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x
    H3 = I + B3 * dt * x
    H4 = I + B4 * dt * x
    
    def apply(psi):
        y = H4 @ (H3 @ (H2 @ (H1 @ psi)))
        return y
    
    return apply


def PPE2(dt, H, neumann_order=10):
    B1 = 1/2
    B2 = -1/2
    
    I = identity(H)
    x = - 1j * H

    H1 = I + B1 * dt * x
    inv2 = - (B2 * dt * x)
    
    def apply(psi):
        y = H1 @ psi
        add = copy(y)
        for _ in range(neumann_order):
            add = inv2 @ add
            y = y + add
        return y
    
    return apply


def PPE4(dt, H, neumann_order=10):
    B1 = 1/12 * (3 - 1j * np.sqrt(3))
    B2 = -(1j/12) * (-3j + np.sqrt(3))
    B3 = 1/12 * (3 + 1j * np.sqrt(3))
    B4 = 1j/12 * (3j + np.sqrt(3))
    
    I = identity(H)
    x = - 1j * H

    H1 = I + B1 * dt * x
    H3 = I + B3 * dt * x
    
    inv2 = - (B2 * dt * x)
    inv4 = - (B4 * dt * x)
    
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


def SLPE2(dt, H):
    A1 = (1 - 1j)/2
    A2 = (1 + 1j)/2
    B1 = (1 - 1j)/2
    B2 = (1 + 1j)/2
    
    Ho, Hd = _split_hamiltonian(H)
    I = identity(H)
    x = - 1j * Ho
    z = - 1j * Hd
    
    D1 = sp.diags(np.exp(z * dt * A1))
    D2 = sp.diags(np.exp(z * dt * A2))
    
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x    
    
    def apply(psi):
        return D2 @ (H2 @ (D1 @ (H1 @ psi)))
    
    return apply


def SLPE3(dt, H):
    A1 = 0.1056624327025936 - 0.3943375672974064 * 1j
    A2 = 0.3943375672974064 + 0.1056624327025936 * 1j
    A3 = 0.3943375672974064 - 0.1056624327025936 * 1j
    A4 = 0.1056624327025936 + 0.3943375672974064 * 1j
    
    B1 = 0.1056624327025936 - 0.3943375672974064 * 1j
    B2 = 0.3943375672974064 + 0.1056624327025936 * 1j
    B3 = 0.3943375672974064 - 0.1056624327025936 * 1j
    B4 = 0.1056624327025936 + 0.3943375672974064 * 1j
    
    Ho, Hd = _split_hamiltonian(H)
    I = identity(H)
    x = - 1j * Ho
    z = - 1j * Hd
    
    D1 = sp.diags(np.exp(z * dt * A1))
    D2 = sp.diags(np.exp(z * dt * A2))
    D3 = sp.diags(np.exp(z * dt * A3))
    D4 = sp.diags(np.exp(z * dt * A4))
    
    H1 = I + B1 * dt * x
    H2 = I + B2 * dt * x
    H3 = I + B3 * dt * x
    H4 = I + B4 * dt * x
    
    def apply(psi):        
        y = D1 @ (H1 @ psi)
        y = D2 @ (H2 @ y)
        y = D3 @ (H3 @ y)
        y = D4 @ (H4 @ y)
        return y
    
    return apply


def SPPE2(dt, H, neumann_order=10):
    A1 = 1/2
    A2 = 1/2
    B1 = 1/2
    B2 = -1/2
    
    Ho, Hd = _split_hamiltonian(H)
    I = identity(H)
    x = - 1j * Ho
    z = - 1j * Hd.diagonal()
    
    D1 = sp.diags(np.exp(z * dt * A1))
    D2 = sp.diags(np.exp(z * dt * A2))
    
    H1 = I + B1 * dt * x
    inv2 = - (B2 * dt * x)
    
    def apply(psi):
        y = H1 @ (D1 @ psi)

        add = copy(y)
        for _ in range(neumann_order):
            add = inv2 @ add
            y = y + add
        
        y = D2 @ y
        return y
    
    return apply


def SPPE3(dt, H, neumann_order=10):
    A1 = (3 - 1j * np.sqrt(3))/12
    A2 = 1/2
    A3 = (3 + 1j * np.sqrt(3))/12
    
    B1 = (3 - 1j * np.sqrt(3))/12
    B2 = 1j/12 * (3j + np.sqrt(3))
    B3 = (3 + 1j * np.sqrt(3))/12
    B4 = -1j/12 * (-3j + np.sqrt(3))
    
    Ho, Hd = _split_hamiltonian(H)
    I = identity(H)
    x = - 1j * Ho
    z = - 1j * Hd.diagonal()
    
    D1 = sp.diags(np.exp(z * dt * A1))
    D2 = sp.diags(np.exp(z * dt * A2))
    D3 = sp.diags(np.exp(z * dt * A3))
    
    H1 = I + B1 * dt * x
    H3 = I + B3 * dt * x
    
    inv2 = - (B2 * dt * x)
    inv4 = - (B4 * dt * x)
    
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


def integrator(
    psi0: np.ndarray,
    e_ops: List[np.ndarray],
    tf: float,
    dt: float,
    f_apply: Callable[[float], Callable[[np.ndarray], np.ndarray]],
    save: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrates the state vector `psi` over time using a given evolution function.

    Args:
        psi0 (np.ndarray): Initial state vector.
        e_ops (List[np.ndarray]): List of operators to compute expectation values.
        tf (float): Final time.
        dt (float): Time step for integration.
        f_apply (Callable[[float], Callable[[np.ndarray], np.ndarray]]): A function that
            generates the evolution step function given the time step.
        save (bool, optional): Whether to save intermediate states. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - A 2D array of state vectors over time (if `save` is True).
            - A 2D array of expectation values for each operator over time.
    """
    n_steps = int(np.ceil(tf / dt)) + 1  # +1 to include the final state
    
    psi_dim = len(psi0)
    psi_array = np.zeros((n_steps, psi_dim), dtype=complex) if save else None
    e_ops_array = np.zeros((n_steps, len(e_ops)), dtype=float)
    times_array = np.linspace(0, tf, n_steps)

    psi = np.array(psi0, copy=True)
    step_function = f_apply(dt)

    if save:
        psi_array[0] = psi
    e_ops_array[0] = _compute_expectation_values(psi, e_ops)

    for i in tqdm(range(1, n_steps), desc="Integrating", unit="steps"):
        psi = step_function(psi)
        psi /= np.linalg.norm(psi) 

        if save:
            psi_array[i] = psi
        e_ops_array[i] = _compute_expectation_values(psi, e_ops)

    return psi_array, times_array, e_ops_array



def _compute_expectation_values(psi: np.ndarray, operators: List[np.ndarray]) -> np.ndarray:
    normalized_psi = psi / np.linalg.norm(psi)
    return np.array([np.vdot(normalized_psi, op.dot(normalized_psi)).real for op in operators])
