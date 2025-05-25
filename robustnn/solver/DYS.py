###################################################
'''
Define the three-operator split solver - DYS
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (May 2024)
'''
import flax.linen as nn
from typing import Sequence
from typing import Callable, Any, Tuple
from flax.typing import Array, PrecisionLike
import jax.numpy as jnp
import jax

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