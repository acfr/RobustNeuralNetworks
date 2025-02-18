import jax
import jax.numpy as jnp
import flax.linen as nn

from robustnn import ren
from utils import data_handling as dh

# Download the data and load it in
dh.download_and_extract_f16()
train, val, stats = dh.load_f16()

# TODO: Initialise a REN

# TODO: Make training/valudation data sets

# TODO: Set up optimiser

# TODO: Write a training loop

# TODO: Test on validation data

# TODO: Save results for later evaluation


# # Max's code for the F16 dataset
# nx = 75
# nv = 150

# data_options = (seq_len = 1024, set = "f16")
# train_options = (Epochs = 70, clip_grad = 1E-1, seed = 123,
#                     schedule = (η = 1E-3, decay_rate = 0.1, decay_steps = 20, min_lr = 1E-6))


# model_options = (model = "stable_ffren",
#                     model_args = (nu = 2, nx = nx, nv = nv, ny = 3))
# model_name = string("stable_ffren_", nx, "_", nv)
# run_sys_id_test(model_name, model_options, data_options, train_options)

# model_options = (model = "lstm", model_args = (nu = 2, nv = 170, ny = 3))
# model_name = string("lstm_", model_options.model_args.nv)
# run_sys_id_test(model_name, model_options, data_options, train_options)
