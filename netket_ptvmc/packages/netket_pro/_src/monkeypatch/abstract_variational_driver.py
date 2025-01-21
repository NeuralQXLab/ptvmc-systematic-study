from netket.driver.abstract_variational_driver import AbstractVariationalDriver
from netket.utils import module_version

from netket_pro._src.monkeypatch.util import replace_method

"""
The idea of this file is to monkeypatch the `iter` method of the `AbstractVariationalDriver` class
"""

if module_version("netket") < (3, 14, 0):
    # This change has made it up into netket.

    @replace_method(AbstractVariationalDriver)
    def iter(self, n_steps: int, step: int = 1):
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                self._forward_and_backward()
                if i == 0:
                    yield self.step_count

                self._step_count += 1
                self.update_parameters(self._dp)
