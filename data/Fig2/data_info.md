`fidelity_snr.npz` can be loaded using the following code:
```python
import numpy as np
data = np.load('fidelity_snr.npz')
```
The data contains the following keys:
- `full_sum_infidelity`: Full sum infidelity
- `smc_mean`: average infidelity with smc estimator
- `cmc_mean`: average infidelity with cmc estimator
- `cv_cmc_mean`: average infidelity with cv-cmc estimator
- `cv_sms_mean`: average infidelity with cv-smc estimator
- `smc_var`: variance of infidelity with smc estimator
- `cmc_var`: variance of infidelity with cmc estimator
- `cv_cmc_var`: variance of infidelity with cv-cmc estimator
- `cv_sms_var`: variance of infidelity with cv-smc estimator
- `snr_smc`: signal-to-noise ratio for smc estimator
- `snr_cmc`: signal-to-noise ratio for cmc estimator
- `snr_cv_cmc`: signal-to-noise ratio for cv-cmc estimator
- `snr_cv_smc`: signal-to-noise ratio for cv-smc estimator