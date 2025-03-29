import jax
import jax.numpy as jnp
import timeit

from robustnn import ren
from robustnn import r2dn
from robustnn.utils import count_num_params

from utils.utils import choose_lbdn_width
from utils import speed
    
filename = "timing_results_v3"

# Choose a fixed problem size
nu = 10     # Number of inputs
ny = 10     # Number of outputs
nx = 64     # Number of states

# Nominal model and data sizes
nv_ren = 128
nv_r2dn = 64
n_layers = 4
nh_r2dn = choose_lbdn_width(nu, nx, ny, nv_ren, nv_r2dn, n_layers)

batches = 64
horizon = 128

# Combinations to run through
batches_ = [2**n for n in range(4, 13)]
horizons_ = [2**n for n in range(11)]
neurons_ = [2**n for n in range(2, 15)]

widths_ = [2**n for n in range(13)]
layers_ = list(range(1, 111, 10))

print("Horizons to test: ", horizons_)
print("Batches to test:  ", batches_)
print("Neurons to test:  ", neurons_, "\n")


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
        return jnp.mean(x1**2) + jnp.mean(y**2)
    
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
    
def time_model(model, batches, horizon, n_repeats, run_timing):
    """Time forwards and backwards passes, print and store results."""
    
    if run_timing:
        # Initialise the model params and count them
        params, states, inputs = initialise_model(model, batches, horizon)
        num_params = count_num_params(params)
        
        # Time the forwards pass
        cf_time, rf_time = time_forwards(model, params, states, inputs, n_repeats)
        print(f"Forwards compile time: {cf_time:.6f} seconds")
        print(f"Forwards eval time   : {rf_time:.6f} seconds")
        
        # Time the backwards pass
        cb_time, rb_time = time_backwards(model, params, states, inputs, n_repeats)
        print(f"Backwards compile time: {cb_time:.6f} seconds")
        print(f"Backwards eval time   : {rb_time:.6f} seconds")
    else:
        num_params, cf_time, rf_time, cb_time, rb_time = None, None, None, None, None
    
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

def run_timing(batches, horizon, nv, nh=None, nlayers = None, n_repeats=1000):
    """Run the timing for either REN or R2DN."""

    if nh is None:
        print("### REN: ###")
        model = ren.ContractingREN(nu, nx, nv, ny)
        num_ps = speed.compute_num_ren_params(model)
        run_timing = False #True if (nv <= 2**10) else False
        
    else:
        print("### R2DN: ###")
        hidden = (nh,) * nlayers
        model = r2dn.ContractingR2DN(nu, nx, nv, ny, hidden)
        num_ps = speed.compute_num_r2dn_params(model)
        run_timing = False #True # Hopefully always feasible for S-REN
        
    results = time_model(model, batches, horizon, n_repeats, run_timing)
    results["num_params"] = num_ps
    if nh is not None: 
        results["nh"] = nh
        results["nlayers"] = nlayers
    
    return results


# Save results each time in case the GPU overloads
results = {}

# Time sequence length
ren_results, r2dn_results = [], []
for h in horizons_:
    print(f"horizon = {h}")
    r1 = run_timing(batches, h, nv_ren)
    r2 = run_timing(batches, h, nv_r2dn, nh_r2dn, n_layers)
    print()
    ren_results.append(r1)
    r2dn_results.append(r2)
    
results["horizon_ren"] = speed.list_to_dicts(ren_results)
results["horizon_r2dn"] = speed.list_to_dicts(r2dn_results)
speed.save_results(filename, results)

# Time batch size
ren_results, r2dn_results = [], []
for b in batches_:
    print(f"batches = {b}")
    r1 = run_timing(b, horizon, nv_ren)
    r2 = run_timing(b, horizon, nv_r2dn, nh_r2dn, n_layers)
    print()
    ren_results.append(r1)
    r2dn_results.append(r2)
    
results["batches_ren"] = speed.list_to_dicts(ren_results)
results["batches_r2dn"] = speed.list_to_dicts(r2dn_results)
speed.save_results(filename, results)

# Time model size
ren_results = []
for nv in neurons_:
    print(f"nv = {nv}")
    r1 = run_timing(batches, horizon, nv)
    ren_results.append(r1)
    print()
    
r2dn_nh_results = []
for nh in widths_:
    print(f"nh = {nh}")
    r2 = run_timing(batches, horizon, nv_r2dn, nh, n_layers)
    r2dn_nh_results.append(r2)
    print()
    
r2dn_l_results = []
for depth in layers_:
    print(f"No. layers = {depth}")
    r3 = run_timing(batches, horizon, nv_r2dn, nh_r2dn, depth)
    r2dn_l_results.append(r3)
    print()

results["nv_ren"] = speed.list_to_dicts(ren_results)
results["nh_r2dn"] = speed.list_to_dicts(r2dn_nh_results)
results["layers_r2dn"] = speed.list_to_dicts(r2dn_l_results)
speed.save_results(filename, results)
