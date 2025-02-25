The different files contain the time-step data and x-magnetization on a 10x10 TFIM. 
The file `cnn_10x10.npz` contains the SLPE3 p-tVMC simulation with the CNN.
The file `vit_10x10.npz` contains the SLPE3 p-tVMC simulation with the ViT.
The file `tvmc_10x10.npz` contains the largest t-VMC simulations form the paper: https://arxiv.org/abs/1912.08828
The file `ipeps_10x10.npz` contains the iPEPS simulations from the paper: https://arxiv.org/abs/1811.05497

```python
import numpy as np
data = np.load('cnn_10x10.npz')
```
The data contains the following keys:
- `times`: The values of the time
- `mx`: The magnetization in the x-direction