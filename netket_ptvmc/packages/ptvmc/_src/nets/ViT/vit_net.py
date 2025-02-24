from .body import Embed, Encoder
from .heads import OutputHead
from flax import linen as nn
from typing import Callable

def extract_patches2d(x, b):
    """
    Extract bxb patches from the (nbatches, nsites) input x.
    Returns x reshaped to (nbatches,npatches,patch_size), where patch_size = b**2
    """
    batch = x.shape[0]
    L_eff = int((x.shape[1] // b**2) ** 0.5)
    x = x.reshape(batch, L_eff, b, L_eff, b)  # [L_eff, b, L_eff, b]
    x = x.transpose(0, 1, 3, 2, 4)  # [L_eff, L_eff, b, b]

    x = x.reshape(batch, L_eff, L_eff, -1)  # [L_eff, L_eff, b*b]
    x = x.reshape(batch, L_eff * L_eff, -1)  # [L_eff*L_eff, b*b]
    return x

class ViT(nn.Module):
    num_layers: int
    d_model: int
    heads: int    
    b: int
    L: int
    expansion_factor: int = 4
    transl_invariant: bool = True

    def setup(self):
        
        self.L_eff = self.L**2//self.b**2
        
        self.patches_and_embed = Embed(
            self.d_model, self.b, extract_patches=extract_patches2d
        )

        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            h=self.heads,
            L_eff=self.L_eff,
            expansion_factor=self.expansion_factor,
            transl_invariant=self.transl_invariant,
        )

        self.output = OutputHead(self.d_model)

    def __call__(self, x):
        x = self.patches_and_embed(x)

        x = self.encoder(x)

        z = self.output(x)

        return z
