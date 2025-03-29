import jax
import jax.numpy as jnp
import timeit
from pathlib import Path

from robustnn import ren
from robustnn import r2dn
from robustnn.utils import count_num_params

from utils import utils

dirpath = Path(__file__).resolve().parent

# Nominal data sizes
batches = 64
horizon = 128


def initialise_model(model, batches, horizon, seed=0):
    """Initialise params, states, and define input sequence."""
    # Sort out RNG keys
    rng = jax.random.key(seed)
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    
    # Create dummy input data
    states = model.initialize_carry(key1, (batches, model.state_size))
    states = jax.random.normal(key1, states.shape)
    inputs = jax.random.normal(key2, (horizon, batches, model.input_size))
    
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

def time_model(model, batches, horizon, n_repeats):
    """Time forwards and backwards passes, print and store results."""
    
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

def build_model(config):
    """Build neural models."""
    nu, nx, ny = config["nx"], config["nx"], config["nx"]
    if config["network"] == "contracting_ren":
        model = ren.ContractingREN(
            nu, nx, config["nv"], ny, 
            identity_output=True,
            activation=utils.get_activation(config["activation"]),
            init_method=config["init_method"],
            do_polar_param=config["polar"]
        )
    elif config["network"] == "contracting_r2dn":
        model = r2dn.ContractingR2DN(
            nu, nx, config["nv"], ny, config["nh"], 
            identity_output=True,
            activation=utils.get_activation(config["activation"]),
            init_method=config["init_method"],
            do_polar_param=config["polar"]
        )
    return model

def run_timing(filename, batches, horizon, n_repeats=1000):
    """Run the timing for a trained REN or R2DN."""

    config, params, results = utils.load_results(filename)
    model = build_model(config)
        
    time_results = time_model(model, batches, horizon, n_repeats)
    results = results | time_results
    utils.save_results(config, params, results)


# Read all the files and save
fpath = dirpath / f"../results/expressivity/"
files = [f for f in fpath.iterdir()]
for f in files:
    if f.is_file() and not (f.suffix == ".pdf"):
        print(f)
        run_timing(f, batches, horizon)
