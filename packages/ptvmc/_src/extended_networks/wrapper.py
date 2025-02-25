import jax
import jax.numpy as jnp
import numpy as np

import flax
import flax.linen as nn

from netket.utils.types import DType

SUPPORTED_OPERATIONS = {"", "Z", "ZZ"}


class DiagonalWrapper(nn.Module):
    r"""
    A wrapper allowing to implement a certain number of operations analytically on the network.
    At the moment we support zz gates.

    Args:
        network: The network containig the correlated state.
        kernel_zz_init: The zz Jastrow network.
        param_dtype: dtype for parameters

    Methods:
        __call__: The forward pass of the network.
        apply_zz: Applies a zz gate to the state by acting directly on the `model_state`.
        apply_z:  Applies a z  gate to the state by acting directly on the `model_state`.
    """

    network: nn.Module
    """The network containig the correlated state."""

    kernel_zz_init: jnp.array = jnp.zeros
    """The zz Jastrow network."""

    kernel_z_init: jnp.array = jnp.zeros
    """The z Jastrow network."""

    param_dtype: DType = jnp.complex128
    """dtype for parameters."""

    supported_operations: frozenset[str] = frozenset({"", "ZZ"})
    """
    Tuple of supported operations. This is used to check if the operations in the decomposition are supported.

    The default is to support ZZ (and the identity) for backward compatibility reasons.
    Should be specified as a set: {"", "Z", "ZZ"} for example.
    """

    def __post_init__(self):
        supported_operations = self.supported_operations
        if supported_operations is None:
            supported_operations = {}
        else:
            supported_operations = set(supported_operations)
        if "" not in supported_operations:
            supported_operations = supported_operations.union({""})
        if any(op not in SUPPORTED_OPERATIONS for op in supported_operations):
            raise ValueError(
                f"Unknown operation : {supported_operations.difference(SUPPORTED_OPERATIONS)}\n"
                "(supported: {SUPPORTED_OPERATIONS})"
            )
        self.supported_operations = frozenset(supported_operations)

        return super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        N = x.shape[-1]

        output = self.network(x)

        if "Z" in self.supported_operations:
            kernel_z = self.variable(
                "modifiers",
                "kernel_z",
                self.kernel_z_init,
                (N,),
                self.param_dtype,
            )
            J_z = construct_linear_kernel(kernel_z, N, self.param_dtype)
            output_z = jnp.einsum("...i,i", x, J_z)
            output = output + output_z

        if "ZZ" in self.supported_operations:
            kernel_zz = self.variable(
                "modifiers",
                "kernel_zz",
                self.kernel_zz_init,
                (N * (N - 1) // 2,),
                self.param_dtype,
            )
            W_zz = construct_jastrow_kernel(kernel_zz, N, self.param_dtype)
            output_zz = jnp.einsum("...i,ij,...j", x, W_zz, x)
            output = output + output_zz

        return output

    def apply_operations(
        self,
        decomposition,
        *,
        scale: float = 1.0,
        variables,
    ):
        if any(key not in self.supported_operations for key in decomposition):
            unsupported = {
                key for key in decomposition if key not in self.supported_operations
            }
            raise ValueError(
                f"Unsupported operation(s): {unsupported}. Supported operations: {self.supported_operations}"
            )

        if "" in decomposition:
            pass
        if "Z" in decomposition:
            indices = decomposition["Z"]["acting_on"]
            weights = decomposition["Z"]["weights"]
            variables = self.apply_z(indices, weights, variables=variables, scale=scale)
        if "ZZ" in decomposition:
            indices = decomposition["ZZ"]["acting_on"]
            weights = decomposition["ZZ"]["weights"]
            variables = self.apply_zz(
                indices, weights, variables=variables, scale=scale
            )
        return variables

    @nn.nowrap
    def apply_z(
        self,
        indices,
        weights,
        *,
        scale: float = 1.0,
        variables,
    ):
        """
        Applies a z gate to the state by acting directly on the `model_state`.

        Args:
            indices: the indices of the z gate.
            weights: the weights of the z gate.
            scale: the scaling factor for the z kernel.
            variables: the dictionary of variables.
        """
        kernel_z = variables["modifiers"]["kernel_z"]
        kernel_z = apply_z(kernel_z, indices, weights, scale=scale)
        new_modifiers = {**variables.get("modifiers", {}), "kernel_z": kernel_z}
        new_variables = flax.core.copy(variables, {"modifiers": new_modifiers})
        return new_variables

    @nn.nowrap
    def apply_zz(
        self,
        indices,
        weights,
        *,
        scale: float = 1.0,
        variables,
    ):
        """
        Applies a zz gate to the state by acting directly on the `model_state`.

        Args:
            indices: the indices of the zz gate.
            weights: the weights of the zz gate.
            scale: the scaling factor for the zz kernel.
            variables: the dictionary of variables.
        """
        kernel_zz = variables["modifiers"]["kernel_zz"]
        kernel_zz = apply_zz(kernel_zz, indices, weights, scale=scale)
        new_modifiers = {**variables.get("modifiers", {}), "kernel_zz": kernel_zz}
        new_variables = flax.core.copy(variables, {"modifiers": new_modifiers})
        return new_variables


def construct_linear_kernel(kernel_z, N, param_dtype):
    if jnp.issubdtype(param_dtype, jnp.complex128):
        J_z_r = (
            jnp.zeros(N, dtype=param_dtype)
            .at[:]
            .set(kernel_z.value.real, unique_indices=True, indices_are_sorted=True)
        )
        J_z_i = (
            jnp.zeros(N, dtype=param_dtype)
            .at[:]
            .set(kernel_z.value.imag, unique_indices=True, indices_are_sorted=True)
        )
        J_z = J_z_r + 1j * J_z_i
    else:
        J_z = (
            jnp.zeros(N, dtype=param_dtype)
            .at[:]
            .set(kernel_z.value, unique_indices=True, indices_are_sorted=True)
        )
    return J_z


def apply_z(kernel_z, indices, weights, scale: float = 1.0):
    """
    Applies a z gate to the state by acting directly on the `model_state`.

    Args:
        kernel_z: the z Jastrow network.
        indices: the indices of the z gate.
        weights: the weights of the z gate.
        scale: the scaling factor for the z kernel.
    """
    if not isinstance(kernel_z, jax.Array):
        kernel_z = jnp.asarray(kernel_z)
    sharding = kernel_z.sharding

    flat_indices = jnp.array([i[0] for i in indices])
    kernel_z = kernel_z.at[flat_indices].add(scale * weights)
    kernel_z = jax.lax.with_sharding_constraint(kernel_z, sharding)
    return kernel_z


def construct_jastrow_kernel(kernel_zz, N, param_dtype):
    il = jnp.tril_indices(N, k=-1)
    if jnp.issubdtype(param_dtype, jnp.complex128):
        W_zz_r = (
            jnp.zeros((N, N), dtype=param_dtype)
            .at[il]
            .set(kernel_zz.value.real, unique_indices=True, indices_are_sorted=True)
        )
        W_zz_i = (
            jnp.zeros((N, N), dtype=param_dtype)
            .at[il]
            .set(kernel_zz.value.imag, unique_indices=True, indices_are_sorted=True)
        )
        W_zz = W_zz_r + 1j * W_zz_i

    else:
        W_zz = (
            jnp.zeros((N, N), dtype=param_dtype)
            .at[il]
            .set(kernel_zz.value, unique_indices=True, indices_are_sorted=True)
        )
    return W_zz


def apply_zz(kernel_zz, indices, weights, scale: float = 1.0):
    """
    Applies a zz gate to the state by acting directly on the `model_state`.

    Args:
        kernel_zz: the zz Jastrow network.
        indices: the indices of the zz gate.
        weights: the weights of the zz gate.
        scale: the scaling factor for the zz kernel.
    """
    i, j = np.array([t for t in indices if len(t) == 2]).T
    lin_indices = lin_to_tril_index(i, j)

    if not isinstance(kernel_zz, jax.Array):
        kernel_zz = jnp.asarray(kernel_zz)
    sharding = kernel_zz.sharding

    kernel_zz = kernel_zz.at[lin_indices].add(scale * weights)
    kernel_zz = jax.lax.with_sharding_constraint(kernel_zz, sharding)
    return kernel_zz


def lin_to_tril_index(i_array, j_array):
    swapped = i_array < j_array
    i_array_swapped = np.where(swapped, j_array, i_array)
    j_array_swapped = np.where(swapped, i_array, j_array)

    return i_array_swapped * (i_array_swapped - 1) // 2 + j_array_swapped
