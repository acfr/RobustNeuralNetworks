import jax
import jax.numpy as jnp

from typing import Callable, Any, Tuple
from flax.typing import Array, PrecisionLike

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.sum(x**2, **kwargs) + eps)


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
    
    U, V = W[:n, :], W[n:, :]
    Z = (U - U.T) + (V.T @ V)
    I = jnp.eye(n)
    ZI = Z + I
    
    # Note that B * A^-1 = solve(A.T, B.T).T
    A_T = jnp.linalg.solve(ZI, I-Z)
    B_T = -2 * jnp.linalg.solve(ZI.T, V.T).T
    
    if return_split:
        return A_T, B_T
    return jnp.concatenate([A_T, B_T])


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


def solve_discrete_lyapunov_direct(a, q):
    """
    JAX implementation of `scipy.linalg.solve_discrete_lyapunov`.
    Only solves via the direct method.
    """
    a = jnp.asarray(a)
    q = jnp.asarray(q)

    lhs = jnp.kron(a, a.conj())
    lhs = jnp.eye(lhs.shape[0]) - lhs
    x = jnp.linalg.solve(lhs, q.flatten())

    return jnp.reshape(x, q.shape)


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


###################################################
import flax.linen as nn
from typing import Sequence
'''
Define the split solver - make it flexible so that other solver can be used
'''
# C(z) in eq 14
# gamma / u * S * ST
def mln_bwd_z2v(gam, mu, S, z):
    return gam/mu * (z @ S) @ S.T 

def mln_RA(gam, mu, S, V, alpha_, bz, zh, uh, units):
    # C(z)
    zv =  mln_bwd_z2v(gam, mu, S, zh)
    # eq 31
    # v=bz - gamma / u * S * ST
    vh = bz - zv

    au, av = 1/(1+alpha_), alpha_/(1+alpha_)
    # eq 31 a/(1+a)v + 1/(1+a)u
    b = av * vh + au * uh
    z = []
    idx = 0
    for k, nz in enumerate( units):
        if k == 0:
            zk = b[..., idx:idx+nz]
        else:
            # a/(1+a) V z + a/(1+a)v + 1/(1+a)u
            zk = av * zk @ V[k-1].T + b[..., idx:idx+nz]
        z.append(zk)
        idx += nz 
    return jnp.concatenate(z, axis=-1)


# todo
# The following functions are used for DavisYinSplit
from robustnn.plnet.monlipnet import ExplicitMonLipParams
def DavisYinSplit(uk, bz, e: ExplicitMonLipParams, 
        inverse_activation_fn: Callable = nn.relu, 
        Lambda: float = 1.0) -> Tuple[Array, Array]:
    """
    Davis-Yin split solver for MonLip networks.
    Args:
        uk (Array): Current value of u.
        bz (Array): Current value of b.
        e (ExplicitMonLipParams): ExplicitMonLipParams object containing the network parameters.
        inverse_activation_fn (Callable, optional): Inverse activation function. Defaults to nn.relu.
        Lambda (float, optional): Step size for the update. Defaults to 1.0.
    Returns:
        Update once (uk+1, zk+1) as mentioned in eq 14.
    """
    # z = prox(u) = arg min 1/2|x-z|^2+af(z)
    # the following is only correct when relu is used - check appendix B for other activation functions
    zh = inverse_activation_fn(uk)
    # u=2z-u
    uh = 2*zh - uk 
    # eq 31
    # a/(1+a) V z + a/(1+a) (bz - gamma / u * S * ST zh) + 1/(1+a) uh
    zk = mln_RA(e.gam, e.mu, e.S, e.V, e.alpha, bz, zh, uh, e.units)
    # u=u+z-z
    uk += Lambda * (zk - zh) 

    return zk, uk
