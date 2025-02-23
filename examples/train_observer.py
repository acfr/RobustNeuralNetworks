import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

from robustnn import ren
from utils.plot_utils import startup_plotting
from utils import observer as obsv 
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")

# Training hyperparameters
# TODO: Tune these. 1e-5 is the benchmark tolerance
default_config = {
    "experiment": "pde",
    "epochs": 100,
    "lr": 1e-3,
    "min_lr": 1e-6,
    "batches": 200,
    "time_steps": 100_000,
    
    "nx": 51,
    "nv": 200,
    "activation": "tanh",
    "init_method": "random",
    "polar": False,
    
    "seed": 0,
}


def build_pde_obsv_ren(input_data, config):
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
    model = build_pde_obsv_ren(input_data, config)

    # Train a model
    params, results = obsv.train_observer(
        model, 
        data, 
        epochs=config["epochs"], 
        lr=config["lr"],
        min_lr=config["min_lr"],
        seed=config["seed"]
    )

    # Save results for later evaluation
    utils.save_results(config, params, results)
    return params, results


def train_and_test(config):
    
    # Train the model
    run_observer_training(config)

    # Load for testing
    config, params, results = utils.load_results_from_config(config)
    _, fname = utils.generate_fname(config)

    # Generate test data
    def init_u_func(*args, **kwargs):
        return 0.5*jnp.ones(*args, **kwargs)
        
    x_true, u = obsv.get_data(
        time_steps=2001,
        init_u_func=init_u_func,
        init_x_func=jnp.ones,
        nx=config["nx"],
        seed=config["seed"],
    )
    y = obsv.measure(x_true, u)
    
    # Re-build and initialise the observer
    model = build_pde_obsv_ren(y, config)
    
    # Simulate the observer through time
    key = jax.random.PRNGKey(config["seed"])
    x0 = model.initialize_carry(key, y[0].shape)
    _, xhat = model.simulate_sequence(params, x0, y)
    
    # Function for plotting the heat maps
    def plot_heatmap(data, i, ax):
        xlabel = "Time steps" if i >= 3 else ""
        ylabel = "True" if i == 1 else ("Observer" if i == 2 else "Error")
        
        im = ax.imshow(data, aspect='auto', cmap="inferno", origin='lower')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yticks([])
        if i < 3:
            ax.set_xticks([])
        return im
    
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
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_trajectory.pdf")
    plt.close()
    
    # Also plot training loss
    plt.plot(results["mean_loss"])
    plt.xlabel("Training epochs")
    plt.ylabel("Training loss")
    plt.yscale('log')
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_loss.pdf")
    plt.close()


# Test it out on nominal config
train_and_test(default_config)

# Change initialisation
config = deepcopy(default_config)
config["init_method"] = "cholesky"
train_and_test(config)

# Change activation
config = deepcopy(default_config)
config["activation"] = "relu"
train_and_test(config)

# Change polar param
config = deepcopy(default_config)
config["polar"] = False
train_and_test(config)
