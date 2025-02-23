import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from robustnn import ren
from utils.plot_utils import startup_plotting
from utils import utils
from utils import sysid
from utils import data_handling as handler

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")

# Training hyperparameters
config = {
    "experiment": "f16",
    "seq_len": 1024,            # TODO: Worth tuning this. Smaller better?
    "epochs": 70,
    "clip_grad": 1e-1,
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
    "init_method": "cholesky",
    "polar": True,
} 


def run_sys_id_test(config):
    """Run system identification on F16 dataset.

    Args:
        config (dict): Training/model config options.
    """
    
    # Download the data and load it in
    handler.download_and_extract_f16()
    train, val = handler.load_f16()

    # Initialise a REN
    nu, ny = 2, 3
    model = ren.ContractingREN(
        nu, 
        config["nx"], 
        config["nv"], 
        ny, 
        activation=utils.get_activation(config["activation"]), 
        init_method=config["init_method"],
        do_polar_param=config["polar"]
    )

    # Make training/validation data sets
    n_segments = train[0].shape[0] / config["seq_len"]
    u_train = jnp.array_split(train[0], n_segments)
    y_train = jnp.array_split(train[1], n_segments)
    train_data = list(zip(u_train, y_train))
    val_data = val
    
    # Set up the optimizer
    optimizer = sysid.setup_optimizer(config, len(u_train))

    # Run the training loop
    params, train_loss = sysid.train(
        train_data, 
        model, 
        optimizer, 
        epochs=config["epochs"], 
        seed=config["seed"]
    )

    # Test on validation data
    results = sysid.validate(model, params, val_data, seed=config["seed"])
    results["train_loss"] = train_loss

    # Save results for later evaluation
    utils.save_results(config, params, results)
    return params, results


def train_and_test(config):
    
    # Train the model
    run_sys_id_test(config)

    # Load and test it
    config, _, results = utils.load_results_from_config(config)
    _, fname = utils.generate_fname(config)

    print("MSE:   ", results["mse"])
    print("NRMSE: ", results["nrmse"])

    # Plot some of the validation results to see if it's working
    batch = 0
    indx = 2
    npoints = 3000
    y_true = results["y"][:npoints, batch, indx]
    y_pred = results["y_pred"][:npoints, batch, indx]

    plt.figure(1)
    plt.plot(results["train_loss"])
    plt.xlabel("Training epochs")
    plt.ylabel("Training loss")
    plt.ylim(0.5, 11.1)
    plt.yscale('log')
    plt.savefig(dirpath / f"../results/f16/{fname}_loss.pdf")

    plt.figure(2)
    plt.plot(y_true - y_pred)
    plt.xlabel("Time steps")
    plt.ylabel("Acceleration")
    plt.savefig(dirpath / f"../results/f16/{fname}_output_dif.pdf")
    plt.close()


# Run it all for a few different configs
config["polar"] = True
config["activation"] = "relu"
config["init_method"] = "cholesky"
train_and_test(config)

config["activation"] = "relu"
config["init_method"] = "random"
train_and_test(config)

# Try it all again without the polar parameterisation
config["polar"] = False
config["activation"] = "relu"
config["init_method"] = "cholesky"
train_and_test(config)

config["activation"] = "relu"
config["init_method"] = "random"
train_and_test(config)

