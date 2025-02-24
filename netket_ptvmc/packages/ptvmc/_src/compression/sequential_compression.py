from copy import copy

from netket.utils.types import PyTree
from netket.vqs import VariationalState
from netket.operator import AbstractOperator

from ptvmc._src.compression.abstract_compression import (
    AbstractStateCompression,
    CompressionState,
    CompressionResultT,
)


class SequentialCompression(AbstractStateCompression):
    r"""A compression defined as a sequence of compressions.

    It can be used, for example, to perform an ADAM pre-training before doing NGD, in an
    infidelity optimization step.
    """

    def __init__(
        self,
        compressions: list[AbstractStateCompression],
    ):
        r"""
        Chains a list of compression algorithms together.

        Args:
            compressions: A list of compression objects.
        """

        self.compression_list = compressions

        self._build_parameters = [c.build_parameters for c in self.compression_list]
        self._run_parameters = [c.run_parameters for c in self.compression_list]

    @property
    def n_stages(self) -> int:
        """Get the number of stages."""
        return len(self.compression_list)

    @property
    def build_parameters(self) -> PyTree:
        """Get the compression parameters."""
        return self._build_parameters

    @property
    def run_parameters(self) -> PyTree:
        """Get the compression parameters."""
        return self._run_parameters

    def _pipeline(
        self,
        vstate: VariationalState,
        tstate: VariationalState,
        U: AbstractOperator,
        V: AbstractOperator,
    ):
        compression_result = copy(vstate)

        for compression in self.compression_list:
            compression_result, compression_state, info = compression.init_and_execute(
                compression_result, tstate, U, V
            )
            yield compression_result, compression_state, info

    def init_state(
        self,
        vstate: VariationalState,
        tstate: VariationalState,
        U: AbstractOperator,
        V: AbstractOperator,
    ) -> CompressionState:
        raise NotImplementedError("Only init_and_execute is implemented")

    def execute(self, driver: CompressionState) -> CompressionResultT:
        raise NotImplementedError("Only init_and_execute is implemented")

    def init_and_execute(
        self,
        vstate: VariationalState,
        tstate: VariationalState,
        U: AbstractOperator,
        V: AbstractOperator,
    ) -> CompressionResultT:

        info = {}
        for i, result in enumerate(self._pipeline(vstate, tstate, U, V)):
            compression_result, compression_state, compression_info = result

            # possibly do something
            info.update({f"compression_{i}": compression_info})

        return compression_result, compression_state, info
