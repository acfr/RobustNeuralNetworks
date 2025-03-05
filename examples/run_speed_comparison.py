import jax
import jax.numpy as jnp
import timeit

from robustnn import ren
from robustnn import scalable_ren as sren
from robustnn.utils import count_num_params

import utils.speed as utils
    
filename = "timing_results"

# Choose a fixed problem size
nu = 10     # Number of inputs
ny = 10     # Number of outputs
nx = 64     # Number of states

# Nominal model sizes
nv_ren = 128
nv_sren = nv_ren // 2
n_layers = 4

# Nominal data sizes
batches = 64
horizon = 256

# Combinations to run through
batches_ = [2**n for n in range(12)]
horizons_ = [2**n for n in range(11)]
neurons_ = [2**n for n in range(2, 10)]

print("Batches to test:  ", batches_)
print("Horizons to test: ", horizons_)
print("Neurons to test:  ", neurons_, "\n")


def build_models(nv_ren, nv_sren, nh_sren):
    """Build the REN and Scalable REN models."""
    model_ren = ren.ContractingREN(nu, nx, nv_ren, ny)
    model_sren = sren.ScalableREN(nu, nx, nv_sren, ny, nh_sren)
    return model_ren, model_sren


def initialise_model(model, batches, horizon, seed=0):
    """Initialise params, states, and define input sequence."""
    # Sort out RNG keys
    rng = jax.random.key(seed)
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    
    # Create dummy input data
    states = model.initialize_carry(key1, (batches, nx))
    states = jax.random.normal(key1, states.shape)
    inputs = jax.random.normal(key2, (horizon, batches, nu))
    
    # Initialise the model and check parameter count
    params = model.init(key3, states, inputs[0])
    return params, states, inputs


def time_forwards(model, params, states, inputs, n_repeats):
    """Time the forwards pass of a model."""
    # Define a simple forwards pass for timing
    @jax.jit
    def forward(params, x0, u):
        x1, _ = model.simulate_sequence(params, x0, u)
        return x1
    
    # Time compilation
    start = timeit.default_timer()
    forward(params, states, inputs).block_until_ready()
    compile_time = timeit.default_timer() - start
    
    # Time evaluation
    eval_time = timeit.timeit(
        lambda: forward(params, states, inputs).block_until_ready(),
        number=n_repeats
    )
    return compile_time, eval_time / n_repeats


def time_backwards(model, params, states, inputs, n_repeats):
    """Time the backwards pass of a model (computing grads)."""
    # Dummy loss function to backpropagate through
    @jax.jit
    def loss(params, x0, u):
        x1, y = model.simulate_sequence(params, x0, u)
        return jnp.sum(x1**2) + jnp.sum(y[-1]**2)
    
    grad_func = jax.jit(jax.grad(loss))
    
    def grad_test(params, x0, u):
        grads = grad_func(params, x0, u)
        jax.tree.map(lambda x: x.block_until_ready, grads)
        return grads
    
    # Time compilation
    start = timeit.default_timer()
    grad_test(params, states, inputs)
    compile_time = timeit.default_timer() - start
    
    # Time evaluation
    eval_time = timeit.timeit(
        lambda: grad_test(params, states, inputs),
        number=n_repeats
    )
    return compile_time, eval_time / n_repeats
    
    
def time_model(model, batches, horizon, n_repeats, do_backwards=True):
    """Time forwards and backwards passes, print and store results."""
    # Initialise the model params and count them
    params, states, inputs = initialise_model(model, batches, horizon)
    num_params = count_num_params(params)
    
    # Time the forwards pass
    cf_time, rf_time = time_forwards(model, params, states, inputs, n_repeats)
    print(f"Forwards compile time: {cf_time:.6f} seconds")
    print(f"Forwards eval time   : {rf_time:.6f} seconds")
    
    # Time the backwards pass
    cb_time, rb_time = None, None
    if do_backwards:
        cb_time, rb_time = time_backwards(model, params, states, inputs, n_repeats)
        print(f"Backwards compile time: {cb_time:.6f} seconds")
        print(f"Backwards eval time   : {rb_time:.6f} seconds")
    
    return {
        "nv": model.features,
        "batches": batches,
        "horizon": horizon,
        "num_params": num_params,
        "forwards_compile": cf_time,
        "forwards_eval": rf_time,
        "backwards_compile": cb_time,
        "backwards_eval": rb_time,
    }


def run_timing(nv_ren, batches, horizon, n_repeats=1000):
    """Run the timing for both REN and scalable REN."""
    # Choose size of scalable-REN to match num params
    nv_sren = nv_ren // 2
    nh = utils.choose_lbdn_width(nu, nx, ny, nv_ren, nv_sren, n_layers)
    
    # Build models
    nh_sren = (nh,) * n_layers
    m_ren, m_sren = build_models(nv_ren, nv_sren, nh_sren)
    
    # Time the forwards and backwards passes
    print("### REN: ###")
    results_ren = time_model(m_ren, batches, horizon, n_repeats)
    print("### Scalable REN: ###")
    results_sren = time_model(m_sren, batches, horizon, n_repeats)
    
    # Add hidden layer info from the scalable REN to look at later
    results_sren["nh"] = m_sren.hidden

    return results_ren, results_sren


# Save results each time in case the GPU overloads
results = {}

# Time sequence length
ren_results, sren_results = [], []
for h in horizons_:
    print(f"horizon = {h}")
    r1, r2 = run_timing(nv_ren, batches, h)
    print()
    ren_results.append(r1)
    sren_results.append(r2)
    
results["horizon_ren"] = utils.list_to_dicts(ren_results)
results["horizon_sren"] = utils.list_to_dicts(sren_results)
utils.save_results(filename, results)

# Time batch size
ren_results, sren_results = [], []
for b in batches_:
    print(f"batches = {b}")
    r1, r2 = run_timing(nv_ren, b, horizon)
    print()
    ren_results.append(r1)
    sren_results.append(r2)
    
results["batches_ren"] = utils.list_to_dicts(ren_results)
results["batches_sren"] = utils.list_to_dicts(sren_results)
utils.save_results(filename, results)

# Time model size
ren_results, sren_results = [], []
for nv in neurons_:
    print(f"nv = {nv}")
    r1, r2 = run_timing(nv, batches, horizon)
    print()
    ren_results.append(r1)
    sren_results.append(r2)
    
results["nv_ren"] = utils.list_to_dicts(ren_results)
results["nv_sren"] = utils.list_to_dicts(sren_results)
utils.save_results(filename, results)
