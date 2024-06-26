import jax
import jax.numpy as jnp

from functools import partial


@partial(jax.jit, static_argnums=(0,))
def solve_tril_layer(activation, D11, b):
    """
    Solve `w = activation(D11 @ w + b)` for lower-triangular D11.
    
    Only valid for the forward pass (not backprop with auto-diff).
    """
    w_eq = jnp.zeros_like(b)
    D11_T = D11.T
    for i in range(D11.shape[0]):
        Di_T = D11_T[:i, i]
        wi = w_eq[..., :i]
        bi = b[..., i]
        Di_wi = wi @ Di_T
        w_eq = w_eq.at[..., i].set(activation(Di_wi + bi))
    return w_eq


def tril_equlibrium_layer(activation, D11, b):
    """
    Solve `w = activation(D11 @ w + b)` for lower-triangular D11.
    
    Activation must be monotone with slope restricted to `[0,1]`.
    """
    # Forward pass
    w_eq = jax.lax.stop_gradient(solve_tril_layer(activation, D11, b))
    
    # Re-evaluate the equilibrium layer so autodiff can track grads
    # through these two operations, then customise for grad of w_eq
    v = w_eq @ D11.T + b
    w_eq = activation(v)
    return tril_layer_do_grad(activation, D11, v, w_eq)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def tril_layer_do_grad(activation, D11, v, w_eq):
    return w_eq


def tril_layer_do_grad_fwd(activation, D11, v, w_eq):
    I = jnp.identity(v.shape[-1])
    return w_eq, (D11, v, I)


def tril_layer_do_grad_bwd(activation, res, y_bar):
    """
    Compute backwards pass with implicit function theorem.
    
    See Equation 13 of Revay et al. (2023).
    """
    D11, v, I = res
    
    # Ignore grads for D11, v
    D11_bar = jnp.zeros_like(D11)
    v_bar = jnp.zeros_like(v)
    
    # Get Jacobian of activation(v) evaluated at v
    # Scalar activation ==> diagonal Jacobian, so get
    # diagonal elements for each batch. j_diag has
    # dimensions (batches, nv)
    _, vjp_act_v = jax.vjp(activation, v)
    j_diag, = vjp_act_v(jnp.ones_like(v))
    
    # Compute gradient with implicit function theorem (per batch)
    w_eq_bar = jnp.zeros_like(v)
    for i in range(w_eq_bar.shape[0]):
        ji = j_diag[i, ...]
        y_bar_i = y_bar[i, ...]
        w_grad = jnp.linalg.solve(I - (ji * D11.T), y_bar_i.T).T
        w_eq_bar = w_eq_bar.at[i, ...].set(w_grad)
    
    return (D11_bar, v_bar, w_eq_bar)


tril_layer_do_grad.defvjp(tril_layer_do_grad_fwd, tril_layer_do_grad_bwd)
