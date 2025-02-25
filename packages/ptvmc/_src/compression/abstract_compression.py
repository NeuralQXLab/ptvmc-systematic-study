import abc


from netket.vqs import VariationalState
from netket.operator import AbstractOperator

from advanced_drivers._src.driver.abstract_variational_driver import (
    AbstractVariationalDriver,
)

CompressionState = AbstractVariationalDriver
CompressionResultT = tuple[VariationalState, CompressionState, dict]


class AbstractStateCompression(abc.ABC):
    r"""
    Abstract class for state compression algorithms.

    This is used by Schroedinger's equation solver to perform an implicit step, i.e. to compute
    the state :math:`\ket{\psi(t+\Delta t)}` from the state :math:`\ket{\psi(t)}` as

    .. math::

        \ket{\psi(t+\Delta t)} = \hat{V}^{-1}\hat{U}\ket{\psi(t)}

    where :math:`\hat{U}` and :math:`\hat{V}` are given by the particular discretization
    scheme used by the solver.

    This abstract class defines the interface that a compression algorithm must implement to be
    used by the solver, and largely contains the two functions

    - :meth:`init_state` to initialize the compression driver.
    - :meth:`execute` to execute the compression.

    A third function :meth:`init_and_execute` is provided as a convenience method to perform both
    initialization and execution in a single call.
    """

    @abc.abstractmethod
    def init_state(
        self,
        vstate: VariationalState,
        tstate: VariationalState,
        U: AbstractOperator,
        V: AbstractOperator,
    ) -> CompressionState:
        r"""
        Initialize the compression algorithm, returning the compression driver and other
        intermediate data.

        Args:
            vstate: The variational state that will encode the compressed state. A copy of this
                state will be passed to the compression driver, and this will not be modified.
            tstate: The state at previous time step to evolve with the operators :math:`\hat{U}`
                and :math:`\hat{V}`. This state will not be modified.
            U: The operator :math:`\hat{U}`.
            V: The operator :math:`\hat{V}`.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def execute(self, driver: CompressionState) -> CompressionResultT:
        r"""
        Execute the compression.

        Args:
            driver: The compression driver initialized by :meth:`init_state`.

        Returns:
            A tuple containing the compressed state, the compression driver, and a dictionary
            holding the data logged during the compression.
        """
        raise NotImplementedError()

    def init_and_execute(
        self,
        vstate: VariationalState,
        tstate: VariationalState,
        U: AbstractOperator,
        V: AbstractOperator,
    ) -> CompressionResultT:
        r"""
        Initializes and executes the compression algorithm for a single step.

        This is equivalent to calling

        .. code-block:: python

            compression_state = self.init_state(vstate, tstate, U, V)
            out = self.execute(compression_state)

        Args:
            vstate: The variational state that will encode the compressed state. A copy of this
                state will be passed to the compression driver, and this will not be modified.
            tstate: The state at previous time step to evolve with the operators :math:`\hat{U}`
                and :math:`\hat{V}`. This state will not be modified.
            U: The operator :math:`\hat{U}`.
            V: The operator :math:`\hat{V}`.

        Returns:
            A tuple containing the compressed state, the compression driver, and a dictionary
            holding the data logged during the compression.

        """

        compression_state = self.init_state(vstate, tstate, U, V)
        out = self.execute(compression_state)
        return out
