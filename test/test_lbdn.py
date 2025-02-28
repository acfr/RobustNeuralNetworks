import jax
import jax.numpy as jnp
import flax.linen as nn

from robustnn.lbdn import LBDN

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")

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
# jit_call = jax.jit(model.apply)
@jax.jit
def jit_call(params, inputs):
    explicit = model.direct_to_explicit(params)
    return model.explicit_call(params, inputs, explicit)

out = jit_call(params, inputs)
print(out)

# Test taking a gradient
def loss(inputs):
    out = jit_call(params, inputs)
    return jnp.sum(out**2)

grad_func = jax.jit(jax.grad(loss))
gs = grad_func(inputs)

print(loss(inputs))
print("Output grad: ", gs[0])
