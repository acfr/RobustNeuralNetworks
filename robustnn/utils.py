import jax.numpy as jnp

from typing import Callable, Any
from flax.typing import Array

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.sum(x**2, **kwargs) + eps)


def identity_init():
    """Initialize a weight as the identity matrix.
    
    Assumes that shape is a tuple (n,n), only uses first element.
    """
    def init_func(key, shape, dtype) -> Array:
        return jnp.identity(shape[0], dtype)
    return init_func