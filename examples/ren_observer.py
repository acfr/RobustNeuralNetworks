# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from robustnn import ren
from robustnn.utils import count_num_params

from utils.plot_utils import startup_plotting
from utils import observer as obsv 
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent
jax.config.update("jax_default_matmul_precision", "highest")

# Training hyperparameters
ren_config = {
    "experiment": "pde",
    "network": "contracting_ren",
    "epochs": 200,                  # (train for longer for better final result)
    "lr": 2e-3,
    "min_lr": 1e-6,
    "lr_patience": 10,
    "batches": 200,
    "time_steps": 100_000,
    
    "nx": 51,
    "nv": 200,
    "activation": "relu",
    "init_method": "long_memory",
    "polar": True,
    
    "seed": 0,
}


def build_ren(input_data, config):
    """Build a REN for the PDE observer."""
    return ren.ContractingREN(
        input_data.shape[-1], 
        config["nx"],
        config["nv"],
        config["nx"],
        activation=utils.get_activation(config["activation"]),
        init_method=config["init_method"],
        do_polar_param=config["polar"],
        identity_output=True
    )


def run_observer_training(config):
    """Run observer design on reaction-diffusion PDE.

    Args:
        config (dict): Training/model config options.
    """
    
    # Get simulated PDE data
    print("Getting observer data...")
    X, U = obsv.get_data(
        time_steps=config["time_steps"], 
        nx=config["nx"],
        seed=config["seed"]
    )
    xt = X[:-1]                 # X at time t
    xn = X[1:]                  # X at time t+1
    y = obsv.measure(X, U)      # Measured end points and middle
    input_data = y[:-1]

    # Split into batches for training
    data = obsv.batch_data(
        xn, 
        xt, 
        input_data, 
        batches=config["batches"], 
        seed=config["seed"]
    )
    print("Done!")

    # Create a REN model for the observer
    model = build_ren(input_data, config)

    # Train a model
    params, results = obsv.train_observer(
        model, 
        data, 
        epochs=config["epochs"], 
        lr=config["lr"],
        min_lr=config["min_lr"],
        lr_patience=config["lr_patience"],
        seed=config["seed"]
    )
    
    # Test it out
    valres = obsv.validate(model, params)
    results = results | valres
    results["num_params"] = count_num_params(params)

    # Save results for later evaluation
    utils.save_results(config, params, results)
    return params, results


def train_and_test(config):
    
    # Train the model
    run_observer_training(config)

    # Load for testing
    config, _, results = utils.load_results_from_config(config)
    _, fname = utils.generate_fname(config)
    x_true = results["true_states"]
    xhat = results["pred_states"]
    
    # For nice colours, change the color map a little bit
    def truncate_colormap(cmap, minval=0.1, maxval=1.0, n=256):
        new_cmap = LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n))
        )
        return new_cmap
    cmap = plt.get_cmap('hot')
    cmap = truncate_colormap(cmap, minval=0.0, maxval=0.85)
    
    # Function for plotting the heat maps
    def plot_heatmap(data, i, ax):
        xlabel = "Time steps" if i >= 3 else ""
        ylabel = "True" if i == 1 else ("Observer" if i == 2 else "Error")
        
        im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yticks([])
        if i < 3:
            ax.set_xticks([])
        return im
    
    # Print number of params
    print("NRMSE: ", results["nrmse"])
    print("Number of params: ", results["num_params"])
    
    # Plot the heat map
    fig, axes = plt.subplots(3, 1, figsize=(6, 4.2))
    im1 = plot_heatmap(x_true.T, 1, axes[0])
    plot_heatmap(xhat.T, 2, axes[1])
    plot_heatmap(jnp.abs(x_true - xhat).T, 3, axes[2])
    fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.1, pad=0.04)
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_heatmap.pdf")
    plt.close(fig)
    
    # Plot estiamated state at a particular spot
    indx = 12
    plt.plot(x_true[:,indx], label="True")
    plt.plot(xhat[:,indx], label="Observer")
    plt.xlabel("Time steps")
    plt.ylabel(f"State at site {indx}")
    plt.legend()
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_trajectory.pdf")
    plt.close()
    
    # Also plot training loss
    plt.plot(results["mean_loss"])
    plt.xlabel("Training epochs")
    plt.ylabel("Training loss")
    plt.yscale('log')
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_loss.pdf")
    plt.close()
    
    # Plot the test loss vs time
    # Plot from second time to ingore compilation time with JIT
    times = results["times"]
    time_seconds = [(t - times[1]).total_seconds() for t in times]
    
    plt.plot(time_seconds[1:], results["mean_loss"][1:])
    plt.xlabel("Training time (s)")
    plt.ylabel("Training loss")
    plt.yscale("log")
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_loss_time.pdf")
    plt.close()


# Test it out on nominal config
train_and_test(ren_config)
