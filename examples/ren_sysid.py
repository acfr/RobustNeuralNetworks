# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/nic-barbara/RobustNeuralNetworks/blob/main/LICENSE 

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from robustnn import ren
from robustnn.utils import count_num_params

from utils.plot_utils import startup_plotting
from utils import utils
from utils import sysid
from utils import data_handling as handler

startup_plotting()
dirpath = Path(__file__).resolve().parent
jax.config.update("jax_default_matmul_precision", "highest")

# Training hyperparameters
# NOTE: This one is quite slow to train. Use a smaller model 
#       (or bigger GPU) to speed things up.
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
    "init_method": "long_memory",
    "polar": True,
} 


def build_ren(config):
    """Build a REN."""
    return ren.ContractingREN(
        2, 
        config["nx"],
        config["nv"],
        3,
        activation=utils.get_activation(config["activation"]), 
        init_method=config["init_method"],
        do_polar_param=config["polar"]
    )
    

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
train_and_test(ren_config)
