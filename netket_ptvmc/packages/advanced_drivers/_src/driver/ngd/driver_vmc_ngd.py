from typing import Callable, Optional
from functools import partial
import jax

from netket.optimizer.solver import cholesky
from netket.utils.types import ScalarOrSchedule, Optimizer, Array
from netket.operator import AbstractOperator
from netket.utils import struct
from netket.vqs.mc import MCState, get_local_kernel, get_local_kernel_arguments
from netket import jax as nkjax

from netket_pro._src import distributed as distributed
from advanced_drivers._src.driver.ngd.driver_abstract_ngd import (
    AbstractNGDDriver,
    _flatten_samples,
    KernelArgs,
    KernelFun,
)


class VMC_NG(AbstractNGDDriver):
    r"""
    Energy minimization using Variational Monte Carlo (VMC) and Stochastic Reconfiguration (SR)
    with or without its kernel formulation. The two approaches lead to *exactly* the same parameter
    updates. In the kernel SR framework, the updates of the parameters can be written as:

    .. math::
        \delta \theta = \tau X(X^TX + \lambda \mathbb{I}_{2M})^{-1} f,

    where :math:`X \in R^{P \times 2M}` is the concatenation of the real and imaginary part
    of the centered Jacobian, with P the number of parameters and M the number of samples.
    The vector f is the concatenation of the real and imaginary part of the centered local
    energy. Note that, to compute the updates, it is sufficient to invert an :math:`M\times M` matrix
    instead of a :math:`P\times P` one. As a consequence, this formulation is useful
    in the typical deep learning regime where :math:`P \gg M`.

    See `R.Rende, L.L.Viteritti, L.Bardone, F.Becca and S.Goldt <https://arxiv.org/abs/2310.05715>`_
    for a detailed description of the derivation. A similar result can be obtained by minimizing the
    Fubini-Study distance with a specific constrain, see `A.Chen and M.Heyl <https://arxiv.org/abs/2302.01941>`_
    for details.

    When `momentum` is used, this driver implements the SPRING optimizer in
    `G.Goldshlager, N.Abrahamsen and L.Lin <https://arxiv.org/abs/2401.10190>`_
    to accumulate previous updates for better approximation of the exact SR with
    no significant performance penalty.
    """

    _ham: AbstractOperator = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
        diag_shift: ScalarOrSchedule,
        proj_reg: Optional[ScalarOrSchedule] = None,
        momentum: Optional[ScalarOrSchedule] = None,
        linear_solver_fn: Callable[[Array, Array], Array] = cholesky,
        evaluation_mode: Optional[str] = None,
        variational_state: MCState = None,
        chunk_size_bwd: Optional[int] = None,
        collect_quadratic_model: bool = False,
        use_ntk: bool = False,
    ):
        r"""
        Initialize the driver.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the bare energy gradient.
            diag_shift: The diagonal shift of the curvature matrix.
            proj_reg: Weight before the matrix `1/N_samples \\bm{1} \\bm{1}^T` used to regularize the linear solver in SPRING.
            momentum: Momentum used to accumulate updates in SPRING.
            linear_solver_fn: Callable to solve the linear problem associated to the updates of the parameters.
            evaluation_mode: The mode used to compute the jacobian or vjp of the variational state.
                Can be `'real'` or `'complex'` (defaults to the dtype of the output of the model) if the jacobian is to be computed in full.
                Can be `'onthefly'` if the jacobian is to be computed on the fly. This last option is only available in the NTK formulation (`use_ntk=True`).
            variational_state: The :class:`netket.vqs.MCState` to be optimised. Other variational states are not supported.
            chunk_size_bwd: The chunk size to use for the backward pass (jacobian or vjp evaluation).
            collect_quadratic_model: Whether to collect the quadratic model. The quantities collected are the linear and quadratic term in the approximation of the loss function. They are stored in the info dictionary of the driver.
            use_ntk: Whether to use the NTK for the computation of the updates.

        Returns:
            The new parameters, the old updates, and the info dictionary.
        """
        self._ham = hamiltonian.collect()  # type: AbstractOperator

        super().__init__(
            optimizer=optimizer,
            diag_shift=diag_shift,
            proj_reg=proj_reg,
            momentum=momentum,
            linear_solver_fn=linear_solver_fn,
            evaluation_mode=evaluation_mode,
            variational_state=variational_state,
            chunk_size_bwd=chunk_size_bwd,
            collect_quadratic_model=collect_quadratic_model,
            use_ntk=use_ntk,
            minimized_quantity_name="Energy",
        )

    def _get_local_estimators_kernel_args(self) -> KernelArgs:
        vstate = self.state
        samples, extra_args = get_local_kernel_arguments(vstate, self._ham)
        return (
            vstate._apply_fun,
            vstate.variables,
            _flatten_samples(samples),
            (extra_args,),
        )

    @property
    def _kernel(self) -> KernelFun:
        chunk_size = self.state.chunk_size

        if chunk_size is None:
            kernel = get_local_kernel(self.state, self._ham)
        else:
            kernel = nkjax.HashablePartial(
                get_local_kernel(self.state, self._ham, chunk_size),
                chunk_size=chunk_size,
            )
        return nkjax.HashablePartial(_vmc_local_kernel, kernel)


# The original kernel from netket follows a different signature, so we must wrap
# the kernel to make it compatible with the new signature.
@partial(jax.jit, static_argnames=("kernel", "logpsi"))
def _vmc_local_kernel(
    kernel,
    logpsi,
    vars,
    samples,
    operator,
):
    local_energies = kernel(logpsi, vars, samples, operator)
    return local_energies, local_energies
