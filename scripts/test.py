import jax
import jax.numpy as jnp

from functools import partial
from jax import custom_vjp
from jax import grad

from time import time, time_ns

from robustnn.ren_jax import solve_tril_layer


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
