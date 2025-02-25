# Embedding and Encoder classes for ViT

from functools import partial

import jax
import jax.numpy as jnp

from flax import linen as nn
from jax import random
from jax._src import dtypes
from einops import rearrange
from typing import Callable


def custom_uniform(scale=1e-2, dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return (2.0 * random.uniform(key, shape, dtype) - 1.0) * scale

    return init


def roll(J, shift, axis=-1):
    return jnp.roll(J, shift, axis=axis)


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(mat, i, j):
    """
    Reshape the (...,N) matrix to (...,L,L), where L = sqrt(N),
    such that [m(0),m(1),...,m(L-1),m(L),...,m(2L-1),...,m(N-1)] -> [[m(0),m(1),...,m(L-1)],
                                                                     [m(L),m(L+1),...,m(2L-1)],
                                                                     .......
                                                                     [m(N-L),m(N-L+1),...,m(N-1)]]

    Then roll the matrix by i along rows (axis = -2) and j along columns (axis = -1)
    """
    side = int(mat.shape[-1] ** 0.5)
    mat = mat.reshape(mat.shape[0], side, side)
    mat = jnp.roll(jnp.roll(mat, i, axis=-2), j, axis=-1)
    return mat.reshape(mat.shape[0], -1)


@partial(jax.vmap, in_axes=(None, 0, None))
def _compute_attn(J, x, h):
    x = rearrange(x, " L_eff (h d_eff) ->  L_eff h d_eff", h=h)

    x = rearrange(x, " L_eff h d_eff ->  h L_eff d_eff")
    x = jnp.matmul(J, x)
    x = rearrange(x, " h L_eff d_eff  ->  L_eff h d_eff")

    x = rearrange(x, " L_eff h d_eff ->  L_eff (h d_eff)")

    return x


class FMHA(nn.Module):
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True

    def setup(self):
        self.v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )
        if self.transl_invariant:
            self.J = self.param(
                "J",
                custom_uniform(scale=(3.0 / self.L_eff) ** 0.5),
                (self.h, self.L_eff),
                jnp.float64,
            )

            sq_L_eff = int(self.L_eff**0.5)
            assert sq_L_eff * sq_L_eff == self.L_eff
            
            self.J = roll2d(
                self.J, jnp.arange(sq_L_eff), jnp.arange(sq_L_eff)
            )  # [h, sqrt(L_eff), sqrt(L_eff), L_eff]

            self.J = self.J.reshape(self.h, -1, self.L_eff)  # [h, L_eff, L_eff]
            
        else:
            self.J = self.param(
                "J",
                custom_uniform(scale=(3.0 / self.L_eff) ** 0.5),
                (self.h, self.L_eff, self.L_eff),
                jnp.float64,
            )

        self.W = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x):
        x = self.v(x)

        x = _compute_attn(self.J, x, self.h)

        x = self.W(x)

        return x


class Embed(nn.Module):
    d_model: int
    b: int
    extract_patches: Callable

    def setup(self):
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x):
        x = self.extract_patches(x, self.b)
        x = self.embed(x)
        return x


class EncoderBlock(nn.Module):
    d_model: int
    h: int
    L_eff: int
    expansion_factor: int = 4
    transl_invariant: bool = True

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            h=self.h,
            L_eff=self.L_eff,
            transl_invariant=self.transl_invariant,
        )

        self.layer_norm_1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        self.layer_norm_2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    self.expansion_factor * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    h: int
    L_eff: int
    expansion_factor: int = 4
    transl_invariant: bool = True

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                h=self.h,
                L_eff=self.L_eff,
                expansion_factor=self.expansion_factor,
                transl_invariant=self.transl_invariant,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)

        return x
