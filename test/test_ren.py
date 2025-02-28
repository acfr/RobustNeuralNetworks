import jax
import jax.numpy as jnp
import flax.linen as nn

from robustnn import ren

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")

# Random seeds
rng = jax.random.key(0)
rng, keyX, keyY, keyS, key1, key2 = jax.random.split(rng, 6)

# Initialise a QSR-constrained REN
nu, nx, nv, ny = 5, 3, 4, 2
X = jax.random.normal(keyX, (ny, ny))
Y = jax.random.normal(keyY, (nu, nu))
S = jax.random.normal(keyS, (nu, ny))

Q = -X.T @ X
R = S @ jnp.linalg.solve(Q, S.T) + Y.T @ Y

model = ren.GeneralREN(nu, nx, nv, ny, Q=Q, S=S, R=R, 
                       activation=nn.tanh, init_method="long_memory")
model.check_valid_qsr()

# Dummy inputs and states
batches = 4
states = model.initialize_carry(key1, (batches, nu)) + 1
inputs = jnp.ones((batches, nu))
params = model.init(key2, states, inputs)

# Forward mode
# jit_call = jax.jit(model.apply)
@jax.jit
def jit_call(params, states, inputs):
    explicit = model.direct_to_explicit(params)
    return model.explicit_call(params, states, inputs, explicit)

new_state, out = jit_call(params, states, inputs)
print(new_state)
print(out)

# Test taking a gradient
def loss(states, inputs):
    nstate, out = jit_call(params, states, inputs)
    return jnp.sum(nstate**2) + jnp.sum(out**2)

grad_func = jax.jit(jax.grad(loss, argnums=(0,1)))
gs = grad_func(states, inputs)

print(loss(states, inputs))
print("States grad: ", gs[0])
print("Output grad: ", gs[1])
