# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Fixed-point / operator-splitting solvers used across `robustnn`.

This module collects the iterative solvers used to evaluate implicit
(equilibrium) layers, so other modules can share a single implementation:

- `DavisYinSplit`: three-operator (Davis-Yin) split solver for the
  monotone-Lipschitz network (`MonLipNet`) equilibrium.
- `ForwardBackwardSplit`, `PeacemanRachfordSplit`,
  `DouglasRachfordSplit`: operator-splitting solvers for the REN equilibrium
  layer `w = activation(D11 @ w + b)` with a full (non-triangular) `D11`. The
  Douglas-Rachford solver is the default used by the (inverse) REN.

All equilibrium-layer solvers return only the forward fixed point; gradients are
attached separately (e.g. via the implicit function theorem in `ren_base_jax`).

Authors:
- Davis-Yin split adapted from "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz
  Networks" [https://arxiv.org/html/2402.01344v2] by Dechuan Liu (May 2024).
- Equilibrium-layer splitting solvers ported from the BiLipREN code by
  Yurui Zhang.
'''
import flax.linen as nn
from typing import Sequence
from typing import Callable, Any, Tuple
from flax.typing import Array, PrecisionLike
import jax.numpy as jnp
import jax

from robustnn.utils import l2_norm

# Default maximum number of iterations for the equilibrium-layer solvers.
_DEFAULT_SOLVER_ITERS = 200

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


# The following functions are used for DavisYinSplit
def DavisYinSplit(uk, bz, e, 
        inverse_activation_fn: Callable = nn.relu, 
        Lambda: float = 1.0, alpha: float = 1.0) -> Tuple[Array, Array]:
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
    zk = mln_RA(e.gam, e.mu, e.S, e.V, alpha, bz, zh, uh, e.units)
    # u=u+z-z
    uk += Lambda * (zk - zh) 

    return zk, uk


######### Equilibrium-layer solvers: w = activation(D11 @ w + b) #########
# These solve the (full, non-triangular) REN equilibrium layer by operator
# splitting. They only compute the forward fixed point; gradients are attached
# separately (e.g. via the implicit function theorem in `ren_base_jax`).


def ForwardBackwardSplit(activation, D11, b, tol=1e-9, alpha=0.8,
                         max_iter=_DEFAULT_SOLVER_ITERS):
    """Solve `w = activation(D11 @ w + b)` by forward-backward splitting.

    Args:
        activation: monotone activation with slope restricted to `[0, 1]`.
        D11 (Array): equilibrium-layer weight matrix.
        b (Array): equilibrium-layer bias.
        tol (float): convergence tolerance on the residual.
        alpha (float): step size in `(0, 1]`.
        max_iter (int): maximum number of iterations.

    Returns:
        Array: the equilibrium point `w`.
    """
    w_eq = jnp.zeros_like(b)
    I = jnp.eye(D11.shape[0], dtype=b.dtype)
    M = ((1.0 - alpha) * I + alpha * D11).T

    def body_fun(carry):
        w_eq, _, k = carry
        w_eq_new = activation(w_eq @ M + alpha * b)
        error = l2_norm(w_eq - w_eq_new)
        return (w_eq_new, error, k + 1)

    def cond_fun(carry):
        _, error, k = carry
        return jnp.logical_and(error >= tol, k < max_iter)

    init_carry = (w_eq, jnp.inf, 0)
    w_eq_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return w_eq_final


def PeacemanRachfordSplit(activation, D11, b, tol=1e-9, alpha=0.8,
                          max_iter=_DEFAULT_SOLVER_ITERS):
    """Solve `w = activation(D11 @ w + b)` by Peaceman-Rachford splitting.

    Args:
        activation: monotone activation with slope restricted to `[0, 1]`.
        D11 (Array): equilibrium-layer weight matrix.
        b (Array): equilibrium-layer bias.
        tol (float): convergence tolerance on the residual.
        alpha (float): step size.
        max_iter (int): maximum number of iterations.

    Returns:
        Array: the equilibrium point `w`.
    """
    w_eq = jnp.zeros_like(b)
    uk = jnp.zeros_like(b)
    I = jnp.eye(D11.shape[0], dtype=b.dtype)

    def body_fun(carry):
        w_eq, uk, _, k = carry
        uh = 2 * w_eq - uk
        zh = jnp.linalg.solve(I + alpha * (I - D11), (uh + alpha * b).T)
        uk_new = 2 * zh.T - uh
        w_eq_new = activation(uk_new)
        error = l2_norm(w_eq - w_eq_new)
        return (w_eq_new, uk_new, error, k + 1)

    def cond_fun(carry):
        _, _, error, k = carry
        return jnp.logical_and(error >= tol, k < max_iter)

    init_carry = (w_eq, uk, jnp.inf, 0)
    w_eq_final, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return w_eq_final


def DouglasRachfordSplit(activation, D11, b, tol=1e-9, alpha=0.6,
                         max_iter=_DEFAULT_SOLVER_ITERS):
    """Solve `w = activation(D11 @ w + b)` by Douglas-Rachford splitting.

    This is the default solver used by the (inverse) REN, whose `D11` is
    generally not lower-triangular.

    Args:
        activation: monotone activation with slope restricted to `[0, 1]`.
        D11 (Array): equilibrium-layer weight matrix.
        b (Array): equilibrium-layer bias.
        tol (float): convergence tolerance on the residual.
        alpha (float): step size.
        max_iter (int): maximum number of iterations.

    Returns:
        Array: the equilibrium point `w`.
    """
    w_eq = jnp.zeros_like(b)
    uk = jnp.zeros_like(b)
    I = jnp.eye(D11.shape[0], dtype=b.dtype)

    def body_fun(carry):
        w_eq, uk, _, k = carry
        uh = 2 * w_eq - uk
        zh = jnp.linalg.solve(I + alpha * (I - D11), (uh + alpha * b).T)
        uk_new = uk - w_eq + zh.T
        w_eq_new = activation(uk_new)
        error = l2_norm(w_eq - w_eq_new)
        return (w_eq_new, uk_new, error, k + 1)

    def cond_fun(carry):
        _, _, error, k = carry
        return jnp.logical_and(error >= tol, k < max_iter)

    init_carry = (w_eq, uk, jnp.inf, 0)
    w_eq_final, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return w_eq_final
