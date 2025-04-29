import jax 
import jax.numpy as jnp
from flax import linen as nn 
from typing import Any, Sequence, Callable
from flax.typing import Array, PrecisionLike


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def cayley(W: Array)-> Array:
    """Perform Cayley transform on a stacked matrix `W = [U; V]`
    with `U.shape == (n, n)` and `V.shape == (m, n)`.

    Args:
        W (Array): Input matrix to transform

    Returns:
        Array: Orthogonal matrix.
    """
    # W in shape n x 2n (m=2n)
    # W = [G H]
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    G, H = W[:n, :], W[n:, :]

    # Z = GT-G + HTH -------- Eq6
    Z = (G - G.T) + (H.T @ H)
    I = jnp.eye(n)
    Zi = jnp.linalg.inv(I+Z)

    # (I+Z)(I-z)-1    -2V(I-Z)-1
    return jnp.concatenate([Zi @ (I-Z), -2 * H @ Zi], axis=0)