import jax
import jax.numpy as jnp

from datetime import datetime

from robustnn import ren
from robustnn import scalable_ren as sren
from robustnn.utils import count_num_params

from utils import compute_p_contractingren, compute_p_sren


# Problem size
nu = 10
ny = 20
batches = 50
horizon = 512

# Initialise a standard contracting REN
nx_ren = 50         # Number of states
nv_ren = 400        # Number of equilibrium layer states
model_ren = ren.ContractingREN(nu, nx_ren, nv_ren, ny)

# Initialise a scalable REN
nx_sren = 50        # Number of states
nv_sren = 100       # Number of equilibirum layer states
nh = (128,)*6       # Number of hidden layers in the LBDN
model_sren = sren.ScalableREN(nu, nx_sren, nv_sren, ny, nh)

# Test function
def test_ren(model, seed=0):
    
    # Random seeds
    rng = jax.random.key(seed)
    rng, key1, key2, key3, key4 = jax.random.split(rng, 5)
    
    # Dummy inputs and states
    states = model.initialize_carry(key1, (batches, nu))
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
    
    return (losses, grads), (first_losses, first_grads)

# print("Testing REN:")
# test_ren(model_ren)

print("Testing Scalable REN:")
test_ren(model_sren)


############################################################


# Test contraction
def mat_norm2(x, A):
    return jnp.sum((x @ A.T) * x, axis=-1)

def test_contraction(model, p_func, seed=0):
    
    # Random seeds
    rng = jax.random.key(seed)
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
    P = p_func(model, params)
    lhs = mat_norm2(x0n - x1n, P) - mat_norm2(x0 - x1, P)
    rhs = 0
    
    return jnp.all(lhs <= rhs)

print("Quick contraction check: ")
# print("REN:  ", test_contraction(model_ren, compute_p_contractingren))
print("SREN: ", test_contraction(model_sren, compute_p_sren))
