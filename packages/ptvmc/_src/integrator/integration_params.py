from typing import Optional

from netket.utils import struct

LimitsDType = tuple[float | None, float | None]
"""Type of the dt limits field, having independently optional upper and lower bounds."""


@struct.dataclass
class IntegrationParameters(struct.Pytree):
    r"""
    Dataclass containing the parameters for the dynamical integration.

    Attributes:
        dt: float: The initial time-step size of the integrator.
        dt_limits: Optional[LimitsDType]: The extremal accepted values for the time-step size `dt`.
    """

    dt: float
    """The initial time-step size of the integrator."""

    # atol: float
    # """The tolerance for the absolute error on the solution."""
    # rtol: float
    # """The tolerance for the relative error on the solution."""

    dt_limits: Optional[LimitsDType] = struct.field(pytree_node=False)
    """The extremal accepted values for the time-step size `dt`."""

    def __init__(
        self,
        dt: float,
        # atol: float = 0.0,
        # rtol: float = 1e-7,
        dt_limits: Optional[LimitsDType] = None,
    ):
        r"""
        Initialize the parameters of the Integrator.

        Args:
            dt: The initial time-step size of the integrator.
            dt_limits: The extremal accepted values for the time-step size `dt`.
                defaults to :code:`(None, 10 * dt)`.
        """
        self.dt = dt

        # self.atol = atol
        # self.rtol = rtol
        if dt_limits is None:
            dt_limits = (None, 10 * dt)
        self.dt_limits = dt_limits
