The different files contain the stability-diagram data for the cmc, smc, and cv-smc estimators. The data can be loaded using the following code:
```python
import numpy as np
data = np.load('stability_diagram.npz')
```
The data contains the following keys:
- `diag_shifts`: The array of diagonal shifts used in the simulation
- `lrs`: The array of learning rates used in the simulation
- `infidelity`: The infidelity reached for each combination of diagonal shift and learning rate
- `rejection_ratio`: The rejection ratio for each combination of diagonal shift and learning rate