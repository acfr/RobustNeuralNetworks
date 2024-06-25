import jax
import jax.numpy as jnp

from functools import partial
from jax import custom_vjp
from jax import grad

from time import time, time_ns


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


##################### Let's write this up #####################

def activation(x):
    return jnp.tanh(x)

@jax.jit
def solve_tril_layer(D11, b):
    """
    Solve `w = activation(D11 @ w + b)` for lower-triangular D11.
    
    Activation must be monotone with slope restricted to `[0,1]`.
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
    

nv, batches = 16, 128
rng = jax.random.PRNGKey(0)
rng1, rng2 = jax.random.split(rng)
b = jax.random.normal(rng2, (batches, nv))
D = jax.random.normal(rng1, (nv, nv))
D = jnp.tril(D, k=-1)

# Run it once
w_eq = solve_tril_layer(D, b)
# print(w_eq)

# Time and test
# Need to jit the bloody test function so everything stays on the GPU...
@jax.jit
def test_func():
    old = solve_tril_layer(D, b)
    for _ in range(N):
        out = solve_tril_layer(D, b)
        results = out - old
        old = out
    return results

N=10000
test_func()
start = time_ns()
out = test_func()
dt = ((time_ns() - start) / N)
print("Time taken (nanosec): ", dt)


# D = jnp.array(
#     [[0.0 , 0.        , 0.        , 0.        , 0.        ],
#      [0.27276897, 0.0 , 0.        , 0.        , 0.        ],
#      [0.8973534 , 0.45088673, 0.0, 0.        , 0.         ],
#      [0.94310784, 0.02125645, 0.44761765, 0.0, 0.         ],
#      [0.24344909, 0.17582   , 0.18456626, 0.40024185, 0.0 ]]
# )
# b = jnp.array(
#     [[0.5338    , 0.9719182 , 0.61623883, 0.868845  , 0.6309322 ],
#     [0.20438278, 0.7415488 , 0.15026295, 0.21696508, 0.32493377],
#     [0.7355863 , 0.79253435, 0.3715024 , 0.1306243 , 0.04838264]]
# )