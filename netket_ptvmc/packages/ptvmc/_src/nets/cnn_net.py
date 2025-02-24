import numpy as np
import jax.numpy as jnp

from netket.utils.types import DType, Array
from netket.graph import Lattice

import jax
import flax.linen as nn

from typing import Sequence

@jax.jit
def logcosh_expanded(z: Array) -> Array:
  return 1/2 * z**2 + (-1/12) * z**4 + (1/45) * z**6 

@jax.jit
def logcosh_expanded_dv(z: Array) -> Array:
  return z + (-1/3) * z**3 + (2/15) * z**5

class CNN(nn.Module):
    lattice: Lattice
    kernel_size: Sequence
    channels: tuple
    param_dtype: DType = complex
    
    def __post_init__(self):
        self.kernel_size = tuple(self.kernel_size)
        self.channels = tuple(self.channels)
        super().__post_init__()
    
    def setup(self):
        if isinstance(self.kernel_size[0], int):
            self.kernels = (self.kernel_size,) * len(self.channels)
        else:
            assert len(self.kernel_size) == len(self.channels)
            self.kernels = self.kernel_size
        
    @nn.compact
    def __call__(self, x):
        lattice_shape = tuple(self.lattice.extent)

        x = x/np.sqrt(2)
        _, ns = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, *lattice_shape, 1)
            
        for i,(c,k) in enumerate(zip(self.channels, self.kernels)):
            x = nn.Conv(
                features=c,
                kernel_size=k,
                padding="CIRCULAR",
                param_dtype = self.param_dtype,
                kernel_init = nn.initializers.xavier_normal(),
                use_bias=True)(x)

            if i:
                x = logcosh_expanded_dv(x)
            else:
                x = logcosh_expanded(x)
        
        x = jnp.sum(x, axis=(1,2))/np.sqrt(ns)
        x = nn.Dense(features=x.shape[-1], param_dtype=self.param_dtype, use_bias=False)(x)
        x = jnp.sum(x, axis=-1)/np.sqrt(x.shape[-1])
        return x
