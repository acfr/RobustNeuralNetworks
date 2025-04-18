#!/usr/bin/env python3
import sys
sys.path.append("..")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path

# from robustnn import ren
# # from robustnn import r2dn
from robustnn import blpdn

from robustnn.utils import count_num_params

from utils.plot_utils import startup_plotting
from utils import observer as obsv
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Need this to avoid matrix multiplication discrepancy
jax.config.update("jax_default_matmul_precision", "highest")
