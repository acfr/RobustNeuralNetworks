import jax
import jax.numpy as jnp
import flax.linen as nn

from robustnn.ren_jax.ren_models import LipschitzREN

# TODO: Test for nx = 0, and/or nv = 0 in forward/reverse mode

rng = jax.random.key(0)
key1, key2 = jax.random.split(rng)

nu, nx, nv, ny = 1, 3, 4, 2
ren = LipschitzREN(nu, nx, nv, ny, gamma=10.0, activation=nn.tanh, abar=0.5)

batches = 4
states = ren.initialize_carry(key1, (batches, nu)) + 1
inputs = jnp.ones((batches, nu))
params = ren.init(key2, states, inputs)

new_state, out = ren.apply(params, states, inputs)
print(new_state)
print(out)

# Test taking a gradient
def loss(states, inputs):
    nstate, out = ren.apply(params, states, inputs)
    return jnp.sum(nstate**2) + jnp.sum(out**2)

grad_func = jax.jit(jax.grad(loss, argnums=(0,1)))
gs = grad_func(states, inputs)

print(loss(states, inputs))
print("States grad: ", gs[0])
print("Output grad: ", gs[1])
