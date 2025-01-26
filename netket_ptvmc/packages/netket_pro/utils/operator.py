from netket.operator import DiscreteJaxOperator, ContinuousOperator


def ensure_jax_operator(op):
    if not isinstance(op, (DiscreteJaxOperator, ContinuousOperator)):
        if hasattr(op, "ensure_jax_operator"):
            op = op.ensure_jax_operator()
        else:
            raise TypeError("Only jax operators supported.")
    return op
