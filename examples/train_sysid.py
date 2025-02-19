import jax.numpy as jnp

from robustnn import ren
from utils import data_handling as handler
from utils import train as utils


# Training hyperparameters
config = {
    "experiment": "f16",
    "seq_len": 1024,
    "epochs": 70,
    "clip_grad": 1e-1,
    "seed": 123,
    "schedule": {
        "init_value": 1e-3,
        "decay_steps": 20,
        "decay_rate": 0.1,
        "end_value": 1e-6,
    },
    "nx": 75,
    "nv": 150,
    "activation": "relu",
    "init_method": "random",
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
        init_method=config["init_method"]
    )

    # Make training/valudation data sets
    n_segments = train[0].shape[0] / config["seq_len"]
    u_train = jnp.array_split(train[0], n_segments)
    y_train = jnp.array_split(train[1], n_segments)
    train_data = list(zip(u_train, y_train))
    val_data = [val]
    
    # Set up the optimizer
    optimizer = utils.setup_optimizer(config, len(u_train))

    # Run the training loop
    params, train_loss = utils.train(
        train_data, 
        model, 
        optimizer, 
        epochs=config["epochs"], 
        seed=config["seed"]
    )

    # Test on validation data
    results = utils.validate(model, params, val_data, seed=config["seed"])
    results["train_loss"] = train_loss

    # Save results for later evaluation
    utils.save_results(config, params, results)
    return params, results


# Train a model
run_sys_id_test(config)
params, results = utils.load_results_from_config(config)
