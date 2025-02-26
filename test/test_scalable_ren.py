import jax
import jax.numpy as jnp
import flax.linen as nn

from robustnn import scalable_ren as sren

# Random seeds
rng = jax.random.key(0)
rng, keyX, keyY, keyS, key1, key2 = jax.random.split(rng, 6)

# Initialise a scalable REN
nu = 5
nx = 3
nv = 6
ny = 2
nh = (8, 16)
model = sren.ScalableREN(
    nu, nx, nv, ny, nh,
    activation=nn.tanh, 
)

# Dummy inputs and states
batches = 4
states = model.initialize_carry(key1, (batches, nu)) + 1
inputs = jnp.ones((batches, nu))
params = model.init(key2, states, inputs)

# Forward mode
jit_call = jax.jit(model.apply)
new_state, out = jit_call(params, states, inputs)
print(new_state)
print(out)

# Test taking a gradient
def loss(states, inputs):
    nstate, out = model.apply(params, states, inputs)
    return jnp.sum(nstate**2) + jnp.sum(out**2)

grad_func = jax.jit(jax.grad(loss, argnums=(0,1)))
gs = grad_func(states, inputs)

print(loss(states, inputs))
print("States grad: ", gs[0])
print("Output grad: ", gs[1])

# Check conversion to explicit params
# print(model.params_to_explicit(params))