from netket.operator import DiscreteOperator, DiscreteJaxOperator


def to_jax_operator(op: DiscreteOperator) -> DiscreteJaxOperator:
    if not isinstance(op, DiscreteJaxOperator):
        if hasattr(op, "to_jax_operator"):
            op = op.to_jax_operator()
        else:
            raise TypeError("Only jax operators supported.")
    return op
