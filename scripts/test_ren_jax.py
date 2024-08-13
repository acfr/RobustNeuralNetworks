import jax
import jax.numpy as jnp
import flax.linen as nn

import robustnn.ren_jax.ren_models as ren

rng = jax.random.key(0)
rng, keyX, keyY, keyS, key1, key2 = jax.random.split(rng, 6)

nu, nx, nv, ny = 5, 3, 4, 2
X = jax.random.normal(keyX, (ny, ny))
Y = jax.random.normal(keyY, (nu, nu))
S = jax.random.normal(keyS, (nu, ny))

Q = -X.T @ X
R = S @ jnp.linalg.solve(Q, S.T) + Y.T @ Y

# model = ren.LipschitzREN(nu, nx, nv, ny, gamma=1.0, activation=nn.tanh)
model = ren.GeneralREN(nu, nx, nv, ny, Q=Q, S=S, R=R, activation=nn.tanh)
model.check_valid_qsr()

batches = 4
states = model.initialize_carry(key1, (batches, nu)) + 1
inputs = jnp.ones((batches, nu))
params = model.init(key2, states, inputs)

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
