import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from utils.plot_utils import startup_plotting

import utils.speed as utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Load the saved data
filename = "timing_results_v3"
filepath = dirpath / "../results/timing/"
savepath = dirpath / "../paperfigs/timing/"
results = utils.load_results(filepath / f"{filename}.pickle")

def format_plot(xlabel, ylabel, filename_suffix, x):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min(x), max(x))
    plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    plt.grid(True, which='both', linestyle=':', linewidth=0.75)
    plt.savefig(savepath / f"{filename}_{filename_suffix}.pdf", bbox_inches='tight')
    plt.close()
    
# Choose colours and linestyles
color_r = "#009E73"
color_s = "#D55E00"

ls_fwd = "solid"
ls_bck = "dashed"


# ---------------------------------------------------------------
# Model sizes
# ---------------------------------------------------------------

def comput_activations_r2dn(exp_key, key1, key2):
    na_r2dn = np.array([], dtype=np.int64)
    for k in range(len(results[exp_key][key1])):
        na = results[exp_key][key1][k] * results[exp_key][key2][k]
        na_r2dn = np.append(na_r2dn, na)
    return na_r2dn
    
# Compute number of activations
na_ren = results["nv_ren"]["nv"]
na_r2dn_nh = comput_activations_r2dn("nh_r2dn", "nh", "nlayers")
na_r2dn_nl = comput_activations_r2dn("layers_r2dn", "nlayers", "nh")

# Number of model params
size_ren = results["nv_ren"]["num_params"]
size_r2dn_nh = results["nh_r2dn"]["num_params"]
size_r2dn_nl = results["layers_r2dn"]["num_params"]

# Plot number of activations vs. number of model params
plt.figure(figsize=(4.2, 2.5))
plt.plot(na_ren, size_ren, color=color_r, label="REN")
plt.plot(na_r2dn_nh, size_r2dn_nh, color=color_s, linestyle=ls_fwd, label="R2DN (width)")
plt.plot(na_r2dn_nl, size_r2dn_nl, color=color_s, linestyle=ls_bck, label="R2DN (depth)")

format_plot("No. activations", "No. parameters", "activations_params", 
            [min([min(na_ren), min(na_r2dn_nh), min(na_r2dn_nl)]), 
             max([max(na_ren), max(na_r2dn_nh), max(na_r2dn_nl)])]
)
