The different files contain the neural network parameters for the states at different time for the TFIM simulations of a quech dynamics with h=hc/10 on a 10x10 lattice. The integration algorithm is SLPE3. 

The data can be simply loaded as 
```python
import netket as nk
import netket_pro as nkp
from ptvmc.nets import ViT

vs = nk.vqs.MCState.load('vit_10x10.npz')
```

