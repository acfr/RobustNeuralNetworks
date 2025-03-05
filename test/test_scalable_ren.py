import jax
import jax.numpy as jnp

from datetime import datetime

from robustnn import scalable_ren as sren
from robustnn.utils import count_num_params

from utils import compute_p_sren


# Problem size
nu = 10
ny = 20
batches = 50
horizon = 512

# Initialise a scalable REN
nx = 50             # Number of states
nv = 100            # Number of equilibirum layer states
nh = (128,)*6       # Number of hidden layers in the LBDN
model = sren.ScalableREN(nu, nx, nv, ny, nh)
    
# Random seeds
rng = jax.random.key(0)
rng, key1, key2, key3, key4 = jax.random.split(rng, 5)

# Dummy inputs and states
states = model.initialize_carry(key1, (batches, None))
states = jax.random.normal(key2, states.shape)
inputs = jax.random.normal(key3, (horizon, batches, nu))

# Initialise the model and check how many params
params = model.init(key4, states, inputs[0])
print("Number of params: ", count_num_params(params))

# Dummy loss function that calls the REN on a sequence of data
@jax.jit
def loss(params, x0, u):
    x1, y = model.simulate_sequence(params, x0, u)
    return jnp.sum(x1**2) + jnp.sum(y**2)

# Roughly time how long it takes to run forward eval. Run it once for JIT first
first_losses = loss(params, states, inputs)

print("Start forward: ", datetime.now())
losses = loss(params, states, inputs)
print("End forward:   ", datetime.now())

# Do the same for the backwards pass
grad_func = jax.jit(jax.grad(loss))
first_grads = grad_func(params, states, inputs)

print("Start backward: ", datetime.now())
grads = grad_func(params, states, inputs)
print("End backward:   ", datetime.now())


############################################################

# Test contraction
def mat_norm2(x, A):
    return jnp.sum((x @ A.T) * x, axis=-1)

# Random seeds
rng = jax.random.key(0)
rng, key1, key2, key3, key4, key5 = jax.random.split(rng, 6)

# Dummy inputs and states
states = model.initialize_carry(key1, (batches, nu))
x0 = 10*jax.random.normal(key2, states.shape)
x1 = jax.random.normal(key3, states.shape)
inputs = jax.random.normal(key4, (batches, nu))

params = model.init(key5, states, inputs)

# Simulate one step
x0n, _ = model.apply(params, x0, inputs)
x1n, _ = model.apply(params, x1, inputs)

# Test for contraction
P = compute_p_sren(model, params)
lhs = mat_norm2(x0n - x1n, P) - mat_norm2(x0 - x1, P)
rhs = 0

assert jnp.all(lhs <= rhs)
