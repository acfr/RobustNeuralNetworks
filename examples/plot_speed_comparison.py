import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from utils.plot_utils import startup_plotting

import utils.speed as utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Load the saved data
filename = "timing_results_v1"
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

format_plot("No. model parameters", "Evaluation time (s)", 
            "modelsize", x=[min([min(x1), min(x2)]), max([max(x1), max(x2)])])


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
    
# We'll put the curves on different x-limits. We need to shift
# the axes appropriately. In log scale, subtraction/addition are
# division/multiplication
xshift = min(nact_sren) / min(nact_ren)
x1min = min(nact_ren)
x1max = max(nact_sren) / xshift
x2min = min(nact_ren) * xshift
x2max = max(nact_sren)

# Plot number of activations vs. number of model params
plt.figure(figsize=(4.2, 2.5))
ax1 = plt.gca()
ax1.plot(nact_ren, x1, color=color_r, label="REN")
ax1.set_xlabel("No. activations (REN)")
ax1.set_ylabel("Number of parameters")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(x1min, x1max)

ax2 = ax1.twiny()
ax2.plot(nact_sren, x2, color=color_s, label="Scalable REN")
ax2.set_xlabel("No. activations (SREN)")
ax2.set_xscale("log")
ax2.set_xlim(x2min, x2max)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

plt.sca(ax1)
plt.legend(lines1 + lines2, labels1 + labels2, 
           ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22))
plt.grid(True, which='both', linestyle=':', linewidth=0.75)
plt.savefig(savepath / f"{filename}_activations_msize.pdf", bbox_inches='tight')
plt.close()


# Plot number of activations vs. computation time
x1 = nact_ren
x2 = nact_sren

plt.figure(figsize=(4.2, 2.5))
ax1 = plt.gca()
ax1.plot(x1, y_ren_fwd, color=color_r, linestyle=ls_fwd, label="REN (Forward)")
ax1.plot(x1, y_ren_bck, color=color_r, linestyle=ls_bck, label="REN (Backward)")
ax1.set_xlabel("No. activations (REN)")
ax1.set_ylabel("Evaluation time (s)")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(x1min, x1max)

ax2 = ax1.twiny()
ax2.plot(x2, y_sren_fwd, color=color_s, linestyle=ls_fwd, label="Scalable REN (Forward)")
ax2.plot(x2, y_sren_bck, color=color_s, linestyle=ls_bck, label="Scalable REN (Backward)")
ax2.set_xlabel("No. activations (SREN)")
ax2.set_xscale("log")
ax2.set_xlim(x2min, x2max)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

plt.sca(ax1)
plt.legend(lines1 + lines2, labels1 + labels2, 
           ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22))
plt.grid(True, which='both', linestyle=':', linewidth=0.75)
plt.savefig(savepath / f"{filename}_activations_time.pdf", bbox_inches='tight')
plt.close()
