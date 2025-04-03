import jax
import jax.numpy as jnp
from flax.linen import initializers as init

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path

from robustnn import ren
from robustnn import r2dn
from robustnn.utils import count_num_params

from utils.plot_utils import startup_plotting
from utils import utils
from utils import sysid
from utils import data_handling as handler

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")

# Training hyperparameters
ren_config = {
    "experiment": "f16",
    "network": "contracting_ren",
    "seq_len": 256,
    "epochs": 70,
    "clip_grad": 10,
    "seed": 0,
    "schedule": {
        "init_value": 1e-3,
        "decay_steps": 20,
        "decay_rate": 0.1,
        "end_value": 1e-6,
    },
    "nx": 75,
    "nv": 150,
    "activation": "relu",
    "init_method": "linear",
    "polar": True,
    
    "linear_init": {
        "kmax": 2,      # Try 4
        "num_eigs": 25  # Try 15
    },
} 

# Should have size: 96995 params (ish)
r2dn_config = deepcopy(ren_config)
r2dn_config["network"] = "contracting_r2dn"

# Reverse-engineer width of hidden layers
r2dn_config["nv"] = ren_config["nv"] // 2
r2dn_config["layers"] = 4
nu, ny = 2, 3
nh = utils.choose_lbdn_width(
    nu, 
    ren_config["nx"], 
    ny, 
    ren_config["nv"], 
    r2dn_config["nv"], 
    r2dn_config["layers"]
)
r2dn_config["nh"] = (nh,) * r2dn_config["layers"]


################################################
# Explicit init
################################################

if ren_config["init_method"] == "linear":
    
    # Setup and size checks
    dt = 1.0 / 400.0 # Data collected at 400 Hz
    kmax = ren_config["linear_init"]["kmax"]
    num_eigs = ren_config["linear_init"]["num_eigs"]
    assert ((kmax+1) * num_eigs == ren_config["nx"])

    # Random seeds
    np.random.seed(ren_config["seed"])
    rng = jax.random.key(ren_config["seed"])
    key1, key2, key3 = jax.random.split(rng, 3)
    
    # Randomly generate a bunch of eigenvalues in the region of interest
    min_freq = 2*np.pi * 2                  # 2 Hz
    max_freq = 2*np.pi * 15                 # 15 Hz
    eigs = np.random.uniform(min_freq, max_freq, num_eigs)
    
    # Generate the initial linear system
    A, _, _, _ = utils.laguerre_composition(eigs, kmax, dt)
    B = init.glorot_normal()(key1, (A.shape[0], nu))
    C = init.glorot_normal()(key2, (ny, A.shape[0]))
    D = jnp.zeros((ny, nu))
    init_lsys = (A, B, C, D)
else:
    init_lsys = ()
    

################################################

def build_ren(config):
    """Build a REN for the PDE observer."""
    if config["network"] == "contracting_ren":
        model = ren.ContractingREN(
            2, 
            config["nx"],
            config["nv"],
            3,
            activation=utils.get_activation(config["activation"]), 
            init_method=config["init_method"],
            do_polar_param=config["polar"],
            init_as_linear=init_lsys,
        )
    elif config["network"] == "contracting_r2dn":
        model = r2dn.ContractingR2DN(
            2,
            config["nx"],
            config["nv"],
            3,
            config["nh"],
            activation=utils.get_activation(config["activation"]),
            init_method=config["init_method"],
            init_as_linear=init_lsys,
        )
    return model
    

def run_sys_id_test(config):
    """Run system identification on F16 dataset.

    Args:
        config (dict): Training/model config options.
    """
    
    # Download the data and load it in
    handler.download_and_extract_f16()
    train, val = handler.load_f16()

    # Initialise a REN
    model = build_ren(config)
    model.explicit_pre_init()

    # Make training/validation data sets
    n_segments = train[0].shape[0] / config["seq_len"]
    u_train = jnp.array_split(train[0], n_segments)
    y_train = jnp.array_split(train[1], n_segments)
    train_data = list(zip(u_train, y_train))
    val_data = val
    
    # Set up the optimizer
    optimizer = sysid.setup_optimizer(config, len(u_train))

    # Run the training loop
    params, train_results = sysid.train(
        train_data, 
        model, 
        optimizer, 
        epochs=config["epochs"], 
        seed=config["seed"]
    )

    # Test on validation data
    results = sysid.validate(model, params, val_data, seed=config["seed"])
    results = results | train_results
    results["num_params"] = count_num_params(params)
    
    # Save results for later evaluation
    utils.save_results(config, params, results)
    return params, results


def train_and_test(config):
    
    # Train the model
    run_sys_id_test(config)

    # Load and test it
    config, _, results = utils.load_results_from_config(config)
    _, fname = utils.generate_fname(config)

    print("Number of params: ", results["num_params"])    
    print("MSE:   ", results["mse"])
    print("NRMSE: ", results["nrmse"])

    # Plot some of the validation results to see if it's working
    batch = 0
    indx = 2
    npoints = 3000
    y_true = results["y"][:npoints, batch, indx]
    y_pred = results["y_pred"][:npoints, batch, indx]

    plt.figure()
    plt.plot(results["train_loss"])
    plt.xlabel("Training epochs")
    plt.ylabel("Training loss")
    plt.ylim(0.5, 11.1)
    plt.yscale('log')
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_loss.pdf")
    plt.close()

    plt.figure()
    plt.plot(y_true - y_pred)
    plt.xlabel("Time steps")
    plt.ylabel("Acceleration")
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_output_dif.pdf")
    plt.close()
    
    # Plot the test loss vs time
    # Plot from second time to ingore compilation time with JIT
    times = results["times"]
    time_seconds = [(t - times[1]).total_seconds() for t in times]
    
    plt.plot(time_seconds[1:], results["train_loss"][1:])
    plt.xlabel("Training time (s)")
    plt.ylabel("Training loss")
    plt.ylim(0.5, 11.1)
    plt.yscale("log")
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_loss_time.pdf")
    plt.close()


# Test it out on nominal config
train_and_test(r2dn_config)
train_and_test(ren_config)
