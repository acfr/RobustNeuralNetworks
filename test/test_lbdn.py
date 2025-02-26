import jax
import jax.numpy as jnp
import flax.linen as nn

from robustnn.lbdn import LBDN

rng = jax.random.key(0)
rng, key = jax.random.split(rng, 2)

# Model size and Lipschitz bound
nu, ny = 5, 2
layers = (8, 16)
gamma = jnp.float32(10)

# Create LBDN model
model = LBDN(nu, layers, ny, gamma=gamma, activation=nn.tanh)

# Dummy inputs
batches = 4
inputs = jnp.ones((batches, nu))
params = model.init(key, inputs)

# Forward mode
jit_call = jax.jit(model.apply)
out = jit_call(params, inputs)
print(out)

# Test taking a gradient
def loss(inputs):
    out = model.apply(params, inputs)
    return jnp.sum(out**2)

grad_func = jax.jit(jax.grad(loss))
gs = grad_func(inputs)

print(loss(inputs))
print("Output grad: ", gs[0])
