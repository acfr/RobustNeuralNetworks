# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

import jax
import jax.numpy as jnp

from typing import Callable, Any, Tuple, Union
from flax.typing import Array, PrecisionLike
import optax

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute the l2 norm of a vector/matrix with JAX, using `optax.safe_norm`.

    Safe for backpropagation, unlike `jnp.linalg.norm`. Returns `max(|x|, eps)`, so 
    it is exact for `|x| > eps` and its gradient is zero below. Keyword arguments 
    (e.g. `axis`, `keepdims`) are passed to `optax.safe_norm`.
    """
    return optax.safe_norm(x, eps, **kwargs)

def _cayley_setup(W: Array) -> Tuple[Array, Array, Array, Array]:
    """Shared setup for the Cayley transform of a stacked matrix `W = [U; V]`.

    Args:
        W (Array): Input matrix to transform. Assumes `W.shape[0] >= W.shape[1]`.

    Returns:
        Tuple[Array, Array, Array, Array]: `(V, Z, I, Z + I)`, where `Z` is the
            matrix from which both Cayley blocks are constructed.
    """
    n = W.shape[1]
    U, V = W[:n, :], W[n:, :]
    Z = (U - U.T) + (V.T @ V)
    I = jnp.eye(n)
    return V, Z, I, Z + I

def cayley(W: Array, return_split:bool=False) -> Array | Tuple[Array, Array]:
    """Perform Cayley transform on a stacked matrix `W = [U; V]`
    with `U.shape == (n, n)` and `V.shape == (m, n)`.

    Args:
        W (Array): Input matrix to transform
        return_split (bool, optional): whether to split the output
            into the two Cayley matrices. Defaults to False.

    Returns:
        Array | Tuple[Array, Array]: Orthogonal matrix (or decomposed matrics).
    """
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    V, Z, I, ZI = _cayley_setup(W)
    
    # Note that B * A^-1 = solve(A.T, B.T).T
    A_T = jnp.linalg.solve(ZI, I-Z)
    B_T = -2 * jnp.linalg.solve(ZI.T, V.T).T
    
    if return_split:
        return A_T, B_T
    return jnp.concatenate([A_T, B_T])

def cayley_b(W: Array) -> Array:
    """Compute only the `B` block of the Cayley transform of `W = [U; V]`.

    Equivalent to `cayley(W, return_split=True)[1]`, but skips the linear solve
    for the `A` block. Use this wherever only `B` is needed (eg: the output
    layer of an LBDN, which has no activation to orthogonalise).

    Args:
        W (Array): Input matrix to transform. Assumes `W.shape[0] >= W.shape[1]`.

    Returns:
        Array: the `B` block of the Cayley transform (transposed, as in `cayley`).
    """
    V, _, _, ZI = _cayley_setup(W)
    return -2 * jnp.linalg.solve(ZI.T, V.T).T


def dot_lax(input1, input2, precision: PrecisionLike = None):
    """
    Wrapper around lax.dot_general(). Use this instead of `@` for
    more accurate array-matrix multiplication (higher default precision?)
    """
    return jax.lax.dot_general(
        input1,
        input2,
        (((input1.ndim - 1,), (1,)), ((), ())),
        precision=precision,
    )
    

def identity_init():
    """Initialize a weight as the identity matrix.
    
    Assumes that shape is a tuple (n,n), only uses first element.
    """
    def init_func(key, shape, dtype) -> Array:
        return jnp.identity(shape[0], dtype)
    return init_func


def count_num_params(d):
    """
    Recursively counts the total number of elements in all jax.numpy arrays
    contained in a dictionary (which may contain nested dictionaries).
    
    Parameters:
    d (dict): Dictionary containing jax.numpy arrays and possibly nested dictionaries.
    
    Returns:
    int: Total number of elements in all jax.numpy arrays.
    """
    total_elements = 0
    for value in d.values():
        if isinstance(value, jnp.ndarray):
            total_elements += value.size
        elif isinstance(value, dict):
            total_elements += count_num_params(value)
    
    return total_elements
