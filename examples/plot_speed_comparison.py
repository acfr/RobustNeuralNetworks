import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from utils.plot_utils import startup_plotting

import utils.speed as utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Load the saved data
filename = "timing_results_v2"
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

# Plot eval time vs. sequence length
x = results["horizon_ren"]["horizon"]

y_ren_fwd = results["horizon_ren"]["forwards_eval"]
y_ren_bck = results["horizon_ren"]["backwards_eval"]
y_sren_fwd = results["horizon_sren"]["forwards_eval"]
y_sren_bck = results["horizon_sren"]["backwards_eval"]

plt.figure(figsize=(4.2, 2.5))
plt.plot(x, y_ren_fwd, color=color_r, linestyle=ls_fwd, label="REN (Forward)")
plt.plot(x, y_ren_bck, color=color_r, linestyle=ls_bck, label="REN (Backward)")
plt.plot(x, y_sren_fwd, color=color_s, linestyle=ls_fwd, label="Scalable REN (Forward)")
plt.plot(x, y_sren_bck, color=color_s, linestyle=ls_bck, label="Scalable REN (Backward)")

format_plot("Sequence length", "Evaluation time (s)", "sequence", x)

# Plot eval time vs. batch size
x = results["batches_ren"]["batches"]

y_ren_fwd = results["batches_ren"]["forwards_eval"]
y_ren_bck = results["batches_ren"]["backwards_eval"]
y_sren_fwd = results["batches_sren"]["forwards_eval"]
y_sren_bck = results["batches_sren"]["backwards_eval"]

plt.figure(figsize=(4.2, 2.5))
plt.plot(x, y_ren_fwd, color=color_r, linestyle=ls_fwd, label="REN (Forward)")
plt.plot(x, y_ren_bck, color=color_r, linestyle=ls_bck, label="REN (Backward)")
plt.plot(x, y_sren_fwd, color=color_s, linestyle=ls_fwd, label="Scalable REN (Forward)")
plt.plot(x, y_sren_bck, color=color_s, linestyle=ls_bck, label="Scalable REN (Backward)")

format_plot("Batch size", "Evaluation time (s)", "batches", x)

# Plot eval time vs. number of model params
x1 = results["nv_ren"]["num_params"]
x2 = results["nv_sren"]["num_params"]

y_ren_fwd = results["nv_ren"]["forwards_eval"]
y_ren_bck = results["nv_ren"]["backwards_eval"]
y_sren_fwd = results["nv_sren"]["forwards_eval"]
y_sren_bck = results["nv_sren"]["backwards_eval"]

plt.figure(figsize=(4.2, 2.5))
plt.plot(x1, y_ren_fwd, color=color_r, linestyle=ls_fwd, label="REN (Forward)")
plt.plot(x1, y_ren_bck, color=color_r, linestyle=ls_bck, label="REN (Backward)")
plt.plot(x2, y_sren_fwd, color=color_s, linestyle=ls_fwd, label="Scalable REN (Forward)")
plt.plot(x2, y_sren_bck, color=color_s, linestyle=ls_bck, label="Scalable REN (Backward)")

# Only plot the parts of the array that are filled in
indx = y_ren_fwd == None
indx = [not val for val in indx]
xmin = min([min(x1), min(x2)])
xmax = max([max(x1[indx]), max(x2[indx])])
format_plot("No. model parameters", "Evaluation time (s)", 
            "modelsize", x=[xmin, xmax])


# ---------------------------------------------------------------
# Lots of plotting code below here to make visualisation nice!
# ---------------------------------------------------------------

# Print hidden layer sizes vs. nv just for my interest
nv_ren = results["nv_ren"]["nv"]
nv_sren = results["nv_sren"]["nv"]
nh_sren = results["nv_sren"]["nh"]
for k in range(len(nv_ren)):
    print("REN nv: ", nv_ren[k], "\tS-REN nv: ", nv_sren[k], "\tS-REN nh: ", nh_sren[k])

# Compute total number of activations
nact_ren = np.zeros(len(nv_ren))
nact_sren = np.zeros(len(nv_ren))
for k in range(len(nv_ren)):
    nv = nv_sren[k]
    nh = nh_sren[k]
    nact_sren[k] = np.prod(nh)
    nact_ren[k] = nv_ren[k]
    
# Compute slope of curves
slope1 = np.log(x1[-1] / x1[-2]) / np.log(nact_ren[-1] / nact_ren[-2])
slope2 = np.log(x2[-1] / x2[-2]) / np.log(nact_sren[-1] / nact_sren[-2])
    
# Plot number of activations vs. number of model params
plt.figure(figsize=(4.2, 2.5))
plt.plot(nact_ren, x1, color=color_r, label="REN")
plt.plot(nact_sren, x2, color=color_s, label="Scalable REN")

plt.annotate(
    f"slope = {slope1:.2f}",
    xy=(8e3, 1e7), 
    xycoords='data',
    fontsize=12,
)
plt.annotate(
    f"slope = {slope2:.2f}",
    xy=(3e9, 2.1e5), 
    xycoords='data',
    fontsize=12,
)

format_plot("No. activations", "No. parameters", "activations_params", 
            [min(nact_ren), max(nact_sren)])

# Plot number of activations vs. computation time
x1 = nact_ren
x2 = nact_sren

plt.figure(figsize=(4.2, 2.5))
plt.plot(x1, y_ren_fwd, color=color_r, linestyle=ls_fwd, label="REN (Forward)")
plt.plot(x1, y_ren_bck, color=color_r, linestyle=ls_bck, label="REN (Backward)")
plt.plot(x2, y_sren_fwd, color=color_s, linestyle=ls_fwd, label="Scalable REN (Forward)")
plt.plot(x2, y_sren_bck, color=color_s, linestyle=ls_bck, label="Scalable REN (Backward)")

format_plot("No. activations", "Evaluation time (s)", "activations_time",
            [min(nact_ren[indx]), max(nact_sren[indx])])
