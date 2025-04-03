import jax
import jax.numpy as jnp
import flax.linen as nn

from robustnn import ren
from robustnn import r2dn

# Need this to avoid matrix multiplication discrepancy?
jax.config.update("jax_default_matmul_precision", "highest")

# Random seeds
rng = jax.random.key(0)
rng, keyA, keyB, keyC, keyD, key1, key2 = jax.random.split(rng, 7)

# Initialise a random linear system
nu, nx, nv, ny = 5, 3, 4, 2
nh = (2,) * 2
A = jax.random.normal(keyA, (nx, nx)) / (5*nx)
B = jax.random.normal(keyB, (nx, nu))
C = jax.random.normal(keyC, (ny, nx))
D = jax.random.normal(keyD, (ny, nu))

# Define a REN from this
model = ren.ContractingREN(
    nu, nx, nv, ny, activation=nn.tanh, init_as_linear=(A,B,C,D)
)
# model = r2dn.ContractingR2DN(
#     nu, nx, nv, ny, nh, activation=nn.tanh, init_as_linear=(A,B,C,D)
# )
model.explicit_pre_init()

# Dummy inputs and states
batches = 1
states = model.initialize_carry(key1, (batches, nu)) + 1
inputs = jnp.ones((batches, nu))
params = model.init(key2, states, inputs)

# Check the result via explicit model
explicit = model.direct_to_explicit(params)
x1, y1 = model.explicit_call(params, states, inputs, explicit)

# Check the result via forward mode
jit_call = jax.jit(model.apply)
x2, y2 = jit_call(params, states, inputs)

# Check it matches the linear system
x3 = states @ A.T + inputs @ B.T
y3 = states @ C.T + inputs @ D.T

print("REN1 state: ", x2)
print("REN2 state: ", x1)
print("SS state:   ", x3, "\n")
print("REN1 out:   ", y2)
print("REN2 out:   ", y1)
print("SS out:     ", y3)
