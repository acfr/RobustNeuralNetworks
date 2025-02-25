import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path

from robustnn import ren
from utils.plot_utils import startup_plotting
from utils import youla
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")

# Training hyperparameters
# Slightly better results with (nx, nv) = (50, 500), 
# but not worth computation time
default_config = {
    "experiment": "youla",
    "epochs": 100,
    "lr": 1e-3,
    "min_lr": 1e-6,
    "lr_patience": 10,
    "batches": 64,
    "test_batches": 64,
    "max_steps": 800,   
    "rollout_length": 200,
    
    "nx": 10,
    "nv": 100,
    "activation": "relu",
    # "init_method": "long_memory",
    "init_method": "long_memory_explicit",
    "polar": True,
    
    "seed": 0,
}


def build_ren(config):
    """Build a REN for the Youla-REN policy."""
    return ren.ContractingREN(
        1, 
        config["nx"],
        config["nv"],
        1,
        activation=utils.get_activation(config["activation"]),
        init_method=config["init_method"],
        do_polar_param=config["polar"],
    )


def run_youla_ren_training(config):
    """Run RL with the Youla-REN on simple linear system.

    Args:
        config (dict): Training/model config options.
    """
    
    # Create the REN model and linear system environment
    model = build_ren(config)
    model.explicit_pre_init()
    env = youla.ExampleSystem()
    
    # Train the model
    params, results = youla.train_yoularen(
        env, 
        model,
        epochs          = config["epochs"],
        batches         = config["batches"],
        test_batches    = config["test_batches"],
        rollout_length  = config["rollout_length"],
        max_steps       = config["max_steps"],
        lr              = config["lr"],
        min_lr          = config["min_lr"],
        lr_patience     = config["lr_patience"],
        seed            = config["seed"]
    )
    
    # Save results for later evaluation
    utils.save_results(config, params, results)
    return params, results


def train_and_test(config):
    
    # Train the model
    run_youla_ren_training(config)

    # Load for testing
    config, params, results = utils.load_results_from_config(config)
    _, fname = utils.generate_fname(config)
    
    # Re-build REN and environment
    model = build_ren(config)
    env = youla.ExampleSystem()
    
    # Generate test data
    batches = 1
    rng = jax.random.PRNGKey(config["seed"])
    test_x0 = env.init_state(batches)
    test_q0 = model.initialize_carry(rng, (batches, 1))
    
    # Test disturbances are steps of increasing magnitude
    amplitudes = jnp.linspace(0, 8, num=7)
    test_d = jnp.vstack(
        [a * jnp.vstack([
            jnp.ones((50, batches, 1)), 
            jnp.zeros((50, batches, 1))
        ]) for a in amplitudes]
    )

    # Roll out the test REN
    _, (z, u) = youla.rollout(env, model, params, test_x0, test_q0, test_d)
    z = jnp.squeeze(z)
    u = jnp.squeeze(u)
    d = jnp.squeeze(test_d)
    
    # Roll out a version with zero control inputs to compare
    max_u, env.max_u = env.max_u, 0
    _, (z0, _) = youla.rollout(env, model, params, test_x0, test_q0, test_d)
    z0 = jnp.squeeze(z0)
    
    # Plot output z vs time
    plt.plot(d, label="Disturbance")
    plt.plot(z0, label="Open Loop")
    plt.plot(z, label="Youla")
    plt.xlabel("Time steps")
    plt.ylabel("Output")
    plt.legend()
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_outputs.pdf")
    plt.close()
    
    # Plot control inputs u vs time
    plt.plot(u, label="Youla-REN")
    plt.hlines(-max_u, xmin=0, xmax=len(u)-1, 
               colors="k", linestyle="dashed", label="Constraints")
    plt.xlabel("Time steps")
    plt.ylabel("Control input")
    plt.legend(loc="upper left")
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_ctrl.pdf")
    plt.close()
    
    # Plot the test loss during training
    plt.plot(results["test_loss"])
    plt.xlabel("Training epochs")
    plt.ylabel("Test loss")
    plt.ylim(1, 10)
    plt.yscale("log")
    plt.savefig(dirpath / f"../results/{config['experiment']}/{fname}_loss.pdf")
    plt.close()
    

# Test it out on nominal config
train_and_test(default_config)
