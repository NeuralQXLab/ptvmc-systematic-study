import jax
import jax.numpy as jnp


@jax.jit
def cast_grad_type(Ō_grad, parameters):
    """Convert the forces vector F_k = cov(O_k, E_loc) to the observable gradient.

    In case of a complex target (which we assume to correspond to a holomorphic
    parametrization), this is the identity. For real-valued parameters, the gradient
    is 2 Re[F].
    """
    Ō_grad = jax.tree_util.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )
    return Ō_grad
