import jax
import numpy as np
import matplotlib.pyplot as plt
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
config = {
    "experiment": "youla",
    "epochs": 100,          # TODO: Tune this
    "lr": 1e-3,
    "min_lr": 1e-6,
    "lr_patience": 10,      # TODO: Tune this
    "batches": 64,          # TODO: Tune this
    "max_steps": 800,   
    "rollout_length": 200,  # TODO: Tune this
    
    "nx": 10, # 50
    "nv": 100, # 500
    "activation": "tanh",
    "init_method": "cholesky",
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
    env = youla.ExampleSystem()
    
    # Train the model
    params, results = youla.train_yoularen(
        env, 
        model,
        epochs          = config["epochs"],
        batches         = config["batches"],
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
    
    # TODO: Evaluate and plot.

# Test it out
train_and_test(config)
