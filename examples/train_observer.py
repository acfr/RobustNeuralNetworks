import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from robustnn import ren
from utils.plot_utils import startup_plotting
from utils import observer as obsv 
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")

# Training hyperparameters
config = {
    "experiment": "pde",
    "epochs": 50,
    "lr": 1e-3,
    "min_lr": 1e-6,
    "batches": 200,
    "time_steps": 100_000,
    
    "nx": 51,
    "nv": 200,
    "activation": "tanh",
    "init_method": "cholesky",
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
        # TODO: Remove output layer for observer example!!
        # It's not actually used and shouldn't be trainable.
    )


def run_observer_training(config):
    """Run observer design on reaction-diffusion PDE.

    Args:
        config (dict): Training/model config options.
    """
    
    # Get simulated PDE data
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
        0.5*jnp.ones(*args, **kwargs)
        
    x_true, u = obsv.get_data(
        time_steps=2000,
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
    xhat = model.simulate_sequence(params, x0, y)
    
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
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    plot_heatmap(x_true, 1, axes[0])
    plot_heatmap(xhat, 2, axes[1])
    plot_heatmap(jnp.abs(x_true - xhat), 3, axes[2])
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_heatmap.pdf")
    plt.close(fig)
    
    # Also plot training loss
    plt.figure(1)
    plt.plot(results["train_loss"])
    plt.xlabel("Training epochs")
    plt.ylabel("Training loss")
    plt.yscale('log')
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_loss.pdf")
    plt.close()


# Test it out on nominal config
train_and_test(config)
