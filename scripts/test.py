import jax
import jax.numpy as jnp

from functools import partial
from jax import custom_vjp
from jax import grad, vjp

import flax.linen as nn

from time import time, time_ns

# from robustnn.ren_jax import solve_tril_layer


### Vector-Jacobian product (reverse mode)

# f :: a -> b
@custom_vjp
def f(x):
  return jnp.sin(x)

# f_fwd :: a -> (b, c)
def f_fwd(x):
  return f(x), (jnp.cos(x),)

# f_bwd :: (c, CT b) -> CT a
def f_bwd(res, y_bar):
  cos_x, = res
  return (cos_x * y_bar,)

f.defvjp(f_fwd, f_bwd)


##################### Custom for REN #####################

def activation(x):
    return jnp.tanh(x)
  
# TODO: Add activation function as an input to all of this!
#       I think sticking with functional programming here
#       actually is fine.

# @jax.jit
def solve_tril_layer(D11, b):
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

def tril_equlibrium_layer(D11, b):
    """
    Solve `w = activation(D11 @ w + b)` for lower-triangular D11.
    
    Activation must be monotone with slope restricted to `[0,1]`.
    """
    
    # Solve the equilibrium layer, forward pass only
    w_eq = jax.lax.stop_gradient(
        solve_tril_layer(D11, b)
    )
    
    # Re-evaluate the equilibrium layer so autodiff can track grads
    # through these two operations, then customise for grad of w_eq
    v = w_eq @ D11.T + b
    w_eq = activation(v)
    # return tril_layer_do_grad(D11, v, w_eq)
    return tril_layer_do_grad_fwd(D11, v, w_eq)

def tril_layer_do_grad(D11, v, w_eq):
    """Dummy forward pass for equilibrium layers."""
    return w_eq
  
def tril_layer_do_grad_fwd(D11, v, w_eq):
    I = jnp.identity(v.shape[-1])
    return w_eq, (D11, v, I)

def tril_layer_do_grad_bwd(res, y_bar):
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
    _, vjp_act_v = vjp(activation, v)
    j_diag, = vjp_act_v(jnp.ones_like(v))
    
    # Compute gradient with implicit function theorem (per batch)
    w_eq_bar = jnp.zeros_like(v)
    for i in range(w_eq_bar.shape[0]):
        ji = j_diag[i, ...]
        y_bar_i = y_bar[i, ...]
        temp = jnp.linalg.solve(I - (ji * D11.T), y_bar_i.T)
        w_eq_bar = w_eq_bar.at[i, ...].set(temp.T)
    
    return (D11_bar, v_bar, w_eq_bar)
    


##################### Test it all out #####################

nv, batches = 16, 128
rng = jax.random.PRNGKey(0)
rng1, rng2 = jax.random.split(rng)
# b = jax.random.normal(rng2, (batches, nv))
# D = jax.random.normal(rng1, (nv, nv))
# D = jnp.tril(D, k=-1)

D = jnp.array(
    [[0.0 , 0.        , 0.        , 0.        , 0.        ],
     [0.27276897, 0.0 , 0.        , 0.        , 0.        ],
     [0.8973534 , 0.45088673, 0.0, 0.        , 0.         ],
     [0.94310784, 0.02125645, 0.44761765, 0.0, 0.         ],
     [0.24344909, 0.17582   , 0.18456626, 0.40024185, 0.0 ]]
)
b = jnp.array(
    [[0.5338    , 0.9719182 , 0.61623883, 0.868845  , 0.6309322 ],
    [0.20438278, 0.7415488 , 0.15026295, 0.21696508, 0.32493377],
    [0.7355863 , 0.79253435, 0.3715024 , 0.1306243 , 0.04838264]]
)
# b = jnp.array([[0.5338    , -0.9719182 , 0.61623883, -0.868845  , 0.6309322 ]])

# Run it once
w_eq = solve_tril_layer(D, b)
# print(w_eq)

# # Time and test
# # Need to jit the bloody test function so everything stays on the GPU...
# @jax.jit
# def test_func():
#     old = solve_tril_layer(D, b)
#     for _ in range(N):
#         out = solve_tril_layer(D, b)
#         results = out - old
#         old = out
#     return results

# N=10000
# test_func()
# start = time_ns()
# out = test_func()
# dt = ((time_ns() - start) / N)
# print("Time taken (nanosec): ", dt)


# Test gradients
out, res = tril_equlibrium_layer(D, b)
out = tril_layer_do_grad_bwd(res, jnp.ones_like(w_eq))
print(out[2])