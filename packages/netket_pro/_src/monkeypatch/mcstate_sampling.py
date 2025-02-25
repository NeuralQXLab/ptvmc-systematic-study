from typing import Optional
from functools import partial

import warnings
import copy

import jax
import jax.numpy as jnp

from netket.sampler import SamplerState
from netket.vqs.mc.mc_state.state import (
    MCState,
    compute_chain_length,
    check_chunk_size,
)

from netket.vqs.mc.mc_state.state import *  # noqa: F403
from netket.utils import timing
from netket.utils.types import PyTree
import netket.jax as nkjax

from netket_pro._src.monkeypatch.util import add_method, attach_method, attach_property

# Add new fields

"""
The idea of this file is to add a few methods mimicking the standard sampling methods to MCState,
that allow for sampling arbitrary distributions.

Those methods also shadow the standard accessors (.samples, ._samples, .sampler_state) to return
the fields managed by netket pro.
"""


def _get_default_name(distribution):
    if isinstance(distribution, partial):
        return _get_default_name(distribution.func)
    if hasattr(distribution, "__name__"):
        return distribution.__module__ + "." + distribution.__name__
    else:
        typ = type(distribution)
        return typ.__module__ + "." + typ.__name__


model_name = "default"


@add_method(MCState)
def init_sampler_distribution(
    self, distribution=None, *, variables=None, seed=None, chain_name=None
):
    """
    Substitute for MCState.sampler.setter ... in the original code, called when sampling
    a new distribution for the first time.
    """
    if distribution is None:
        distribution = self._model
        if chain_name is None:
            chain_name = model_name
    if chain_name is None:
        chain_name = _get_default_name(distribution)

    if variables is None:
        variables = self.variables
    if seed is None:
        if self._sampler_seed is not None:
            self._sampler_seed, seed = jax.random.split(self._sampler_seed)

    sampler_state = self.sampler.init_state(distribution, variables, seed=seed)
    self.sampler_states[chain_name] = sampler_state
    self._sampler_states_previous[chain_name] = sampler_state
    return sampler_state


@attach_method(MCState)
def __init__(self, *args, **kwargs):
    # Dictionary holding the sampler_states for each distribution (equivalent to .sampler_state)
    self.sampler_states = {}
    # Dictionary holding the previous sampler_states for each distribution (equivalent to ._sampler_state_previous)
    self._sampler_states_previous = {}
    # Dictionary holding the samples for each distribution (equivalent to .samples, not serialized)
    self._samples_distributions = {}
    # Dictionary holding the samples of past distributions for resampling (equivalent to ._samples, not serialized)
    self._samples_distribution_resampling_cache = {}
    if not hasattr(self, "_resample_fraction"):
        self._resample_fraction = None


@attach_method(MCState)
def reset(self):
    """Resets the state of the sampler, in order to generate new samples.

    If resmplae_fraction is set, the old samples are kept in memory.
    To reset the samples used by the resample_fraction, you must
    call :func:`~netket.vqs.MCState.reset_hard`.
    """
    # The code in init should in __init__, because init
    # is not always called, but do what we can do here...
    if not hasattr(self, "_samples_distributions"):
        __init__(self)
    # Do this in the sampler code itself
    # for distribution in self._samples_distribution_resampling_cache:
    #    self._samples_distribution_resampling_cache[
    #        distribution
    #    ] = self._samples_distributions[distribution]
    for distribution in self._samples_distributions:
        self._samples_distributions[distribution] = None


@add_method(MCState)
def reset_hard(self):
    """Removes the samples used for resampling."""
    self.reset()
    self._samples_distribution_resampling_cache.clear()


@attach_property(MCState, name="sampler", mode="set", prepend=True)
def sampler(self, new_sampler):
    for distribution in self._sampler_states_previous:
        self.sampler_states[distribution] = None
        self._sampler_states_previous[distribution] = None
        self._samples_distributions[distribution] = None
        if distribution in self._samples_distribution_resampling_cache:
            self._samples_distribution_resampling_cache[distribution] = None


@partial(jax.jit, static_argnames=("chain_length_to_sample"))
def _concatenate_samples(previous_samples, samples, chain_length_to_sample):
    return jnp.concatenate(
        [previous_samples[:, chain_length_to_sample:, :], samples], axis=1
    )


@add_method(MCState)
def sample_distribution(
    self,
    distribution=None,
    variables: Optional[PyTree] = None,
    seed: Optional[int] = None,
    *,
    chain_length: Optional[int] = None,
    n_samples: Optional[int] = None,
    n_discard_per_chain: Optional[int] = None,
    resample_fraction: Optional[float] = None,
    chain_name: Optional[str] = None,
) -> jnp.ndarray:
    r"""Returns the samples for this model given a distribution.

    This function behaves like :attr:`~netket.vqs.MCState.samples`, also sampling its
    output values, but it allows to specify a distribution different from the wavefunction
    itself.

    This is used mainly for the sampling of the :math:`H\\log\\psi` and a target distribution with
    the infidelity sampler, but can also be used to build importance sampling.

    By default, with no arguments, this behaves as :attr:`~netket.vqs.MCState.samples`.

    .. note::

        The sampler state for those distributions are stored in the MCState in the attribute
        `sampler_states`. The samples are stored in the attribute `_samples_distributions`,
        the resampling cache in `_samples_distribution_resampling_cache`.

    Args:
        distribution: The distribution to sample from. This must be a flax Module or
            a Callable with the standard structure. If None, the model of the state is used.
        variables: The variables to sample from. If None, the variables of the model are used
        seed: The seed for the random number generator to be used if the sampler must be
            initialized. It's ignored otherwise.
        chain_length: The length of the chain to sample. If None, the value set internally
            is used. If n_samples is set, this is computed from n_samples. If resample_fraction
            is set, the actual chain length will be :math:`\text{chain_length} \times \text{resample_fraction}`.
        resample_fraction: The fraction of the chain to resample. If None, the value set
            internally is used. If 0, only 1 sample per chain is resampled. Effectively this
            reduces the number of samples to be generated, while returning always the
            same number of samples, already concatenated with the old ones.

    """
    if distribution is None:
        distribution = self._model
        if chain_name is None:
            chain_name = model_name
    if chain_name is None:
        if distribution is self._model:
            chain_name = model_name
        else:
            chain_name = _get_default_name(distribution)

    if variables is None:
        variables = self.variables
    if resample_fraction is None:
        resample_fraction = self.resample_fraction

    if n_samples is None and chain_length is None:
        chain_length = self.chain_length
    else:
        if chain_length is None:
            chain_length = compute_chain_length(self.sampler.n_chains, n_samples)
        if self.chunk_size is not None:
            check_chunk_size(chain_length * self.sampler.n_chains, self.chunk_size)

    if n_discard_per_chain is None:
        n_discard_per_chain = self.n_discard_per_chain

    if resample_fraction is not None:
        if chain_name not in self._samples_distribution_resampling_cache:
            self._samples_distribution_resampling_cache[chain_name] = None

        previous_samples = self._samples_distribution_resampling_cache.get(
            chain_name, None
        )
        if previous_samples is None:
            chain_length_to_sample = chain_length
        else:
            chain_length_to_sample = max(int(chain_length * resample_fraction), 1)
    else:
        chain_length_to_sample = chain_length
        # If we stop to use resample_fraction, we can remove the cache
        if chain_name in self._samples_distribution_resampling_cache:
            del self._samples_distribution_resampling_cache[chain_name]

    sampler_state = self.sampler_states.get(chain_name, None)

    if sampler_state is None:
        sampler_state = self.init_sampler_distribution(
            distribution,
            variables=variables,
            seed=seed,
            chain_name=chain_name,
        )

    # Store the previous sampler state, for serialization purposes
    self._sampler_states_previous[chain_name] = sampler_state

    sampler_state = self.sampler.reset(distribution, variables, sampler_state)

    with timing.timed_scope(f"MCState.sample_distribution #{hash(distribution)}"):
        if n_discard_per_chain > 0:
            with timing.timed_scope("sampling n_discarded samples") as timer:
                _, sampler_state = self.sampler.sample(
                    distribution,
                    variables,
                    state=sampler_state,
                    chain_length=n_discard_per_chain,
                )
                # This won't actually block unless we are really timing
                timer.block_until_ready(_)

        samples, sampler_state = self.sampler.sample(
            distribution,
            variables,
            state=sampler_state,
            chain_length=chain_length_to_sample,
        )

    if resample_fraction is not None:
        if previous_samples is not None:
            samples = _concatenate_samples(
                previous_samples, samples, chain_length_to_sample
            )
        # Store the samples for resampling only a part next time.
        self._samples_distribution_resampling_cache[chain_name] = samples

    self.sampler_states[chain_name] = sampler_state
    self._samples_distributions[chain_name] = samples

    return samples


@add_method(MCState)
def samples_distribution(
    self,
    distribution=None,
    variables: Optional[PyTree] = None,
    seed: Optional[int] = None,
    *,
    resample_fraction: Optional[float] = None,
    chain_name: Optional[str] = None,
) -> jnp.ndarray:
    r"""Returns the samples for this model given a distribution.

    This function behaves like :attr:`~netket.vqs.MCState.samples`, also sampling its
    output values, but it allows to specify a distribution different from the wavefunction
    itself.

    This is used mainly for the sampling of the :math:`H\log\psi` and a target distribution with
    the infidelity sampler, but can also be used to build importance sampling.

    By default, with no arguments, this behaves as :attr:`~netket.vqs.MCState.samples`.

    .. note::

        The sampler state for those distributions are stored in the MCState in the attribute
        `sampler_states`. The samples are stored in the attribute `_samples_distributions`,
        the resampling cache in `_samples_distribution_resampling_cache`.

    Args:
        distribution: The distribution to sample from. This must be a flax Module or
            a Callable with the standard structure. If None, the model of the state is used.
        variables: The variables to sample from. If None, the variables of the model are used
        seed: The seed for the random number generator to be used if the sampler must be
            initialized. It's ignored otherwise.
        resample_fraction: The fraction of the chain to resample. If None, the value set
            internally is used. If 0, only 1 sample per chain is resampled. Effectively this
            reduces the number of samples to be generated, while returning always the
            same number of samples, already concatenated with the old ones.

    """
    if distribution is None:
        distribution = self._model
        if chain_name is None:
            chain_name = model_name
    if chain_name is None:
        if distribution is self._model:
            chain_name = model_name
        else:
            chain_name = _get_default_name(distribution)

    samples = self._samples_distributions.get(chain_name, None)
    if samples is None:
        self.sample_distribution(
            distribution,
            variables,
            resample_fraction=resample_fraction,
            seed=seed,
            chain_name=chain_name,
        )
        samples = self._samples_distributions[chain_name]
    return samples


@add_method(MCState)
def __copy__(self):
    # default new = copy.copy(self)
    cls = type(self)
    new = cls.__new__(cls)
    for k, v in self.__dict__.items():
        new.__dict__[k] = v

    new.sampler_states = copy.copy(self.sampler_states)
    new._sampler_states_previous = copy.copy(self._sampler_states_previous)
    new._samples_distributions = copy.copy(self._samples_distributions)
    new._samples_distribution_resampling_cache = copy.copy(
        self._samples_distribution_resampling_cache
    )
    return new


@add_method(MCState)
def replace_sampler_seed(self, seed: Optional[int] = None):
    """This function should be used to change the rng state of all samplers contained in a
    Monte Carlo State.

    The use-case for this is when you create a copy of a MCState with copy.copy, but you
    want the copy to generate samples that are not correlated to the ones of the original
    state.

    Beware, that for this to work correctly, you probably need to resample a bunch of times
    in order to decorrelate the chains, because this method only changes the rng seed, but not
    the current configurations in the chain.
    """
    seed = nkjax.mpi_split(nkjax.PRNGKey(seed))

    n_samplers = 1 + len(self.sampler_states.keys())
    seeds = jax.random.split(seed, n_samplers)

    self.sampler_state = self.sampler_state.replace(rng=seeds[0])
    for i, (k, v) in enumerate(self.sampler_states.items()):
        self.sampler_states[k] = v.replace(rng=seeds[i + 1])

    self.reset()


@attach_property(MCState, name="samples", mode="get", prepend=False)
def samples(self) -> jax.Array:
    """Returns the set of cached samples.

    .. note::
        This method has been overriden in netket_pro to use
        the :func:`netket.vqs.MCState.samples_distribution` method.

    The samples returned are guaranteed valid for the current state of
    the variational state. If no cached parameters are available, then
    they are sampled first and then cached.

    To obtain a new set of samples either use
    :meth:`~MCState.reset` or :meth:`~MCState.sample`.
    """
    return self.samples_distribution()


@property
def sampler_state(self) -> Optional[SamplerState]:
    return self.sampler_states.get(model_name, None)


@sampler_state.setter
def sampler_state(self, value):
    self.sampler_states[model_name] = value


add_method(sampler_state, MCState)


@property
def _sampler_state_previous(self):
    return self._sampler_states_previous.get(model_name, None)


@_sampler_state_previous.setter
def _sampler_state_previous(self, value):
    self._sampler_states_previous[model_name] = value


add_method(_sampler_state_previous, MCState)


@property
def _samples(self):
    return self._samples_distributions.get(model_name, None)


@_samples.setter
def _samples(self, value):
    self._samples_distributions[model_name] = value


add_method(_samples, MCState, override=True)


#
@property
def resample_fraction(self) -> Optional[float]:
    """The fraction of the chain to resample at every sampling step.

    This is used to reduce the number of samples to be generated, while returning always the
    same number of samples.
    """
    return getattr(self, "_resample_fraction", None)


@resample_fraction.setter
def resample_fraction(self, value: Optional[float]):
    if value is not None:
        chain_length_to_sample = int(max(self.chain_length * value, 1))
        new_resample_fraction = chain_length_to_sample / self.chain_length
        if new_resample_fraction != value:
            warnings.warn(
                f"""
            Resample fraction was set to {value}, but this does not divide the chain length {self.chain_length} evenly,
            so it was set to {new_resample_fraction} instead, which corresponds to a chain length of {chain_length_to_sample}.
            """
            )
        self._resample_fraction = new_resample_fraction
    else:
        self._resample_fraction = None


add_method(resample_fraction, MCState)
