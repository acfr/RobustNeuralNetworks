#!/usr/bin/env python3
import sys
sys.path.append("..")

import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt
import jax.random as random
from copy import deepcopy
from pathlib import Path

# from robustnn import ren
# # from robustnn import r2dn
from robustnn import blbdn

from robustnn.utils import count_num_params
from examples.utils.two_dof_gen import generate_two_dof_data

from utils.plot_utils import startup_plotting
from utils import observer as obsv
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")
blbdn_config = {
    "input_dim": 2,
    "output_dim": 2,
    "lr_max": 1e-2,
    "epochs": 5,
    "n_batch": 50,
    "name": 'BiLipNet',
    "depth": 2,
    "layer_size": [256]*8,
    "tau": 2,
    "gamma": 1,
    "activation": "relu"
}

rng = random.PRNGKey(42)
rng, rng_data = random.split(rng, 2)

data = generate_two_dof_data(rng_data)

print(f"DATA dim: {data['data_dim']}")

def train_model(model, data):
    model.explicit_pre_init()




def build_blbdn(input_data, config) -> blbdn.BLBDN:
    """Build a Bilip for the C-space."""
    print(config['layer_size'])
    model = blbdn.BLBDN(
        input_size=data['data_dim'],
        output_size=config['output_dim'],
        hidden_sizes=config["layer_size"],
        gamma=config["gamma"],
        activation=utils.get_activation(config["activation"]),
    )
    return model


model = build_blbdn(data, blbdn_config)
ret = train_model(model, data)
