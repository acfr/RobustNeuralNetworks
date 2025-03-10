import matplotlib.pyplot as plt

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
    plt.xscale("log", base=2 if "size" not in filename_suffix else 10)
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

# Plot eval time vs. sequence length (forward)
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

# Plot eval time vs. batch size (forward)
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

# Plot eval time vs. number of model params (forward)
x = results["nv_ren"]["num_params"]

y_ren_fwd = results["nv_ren"]["forwards_eval"]
y_ren_bck = results["nv_ren"]["backwards_eval"]
y_sren_fwd = results["nv_sren"]["forwards_eval"]
y_sren_bck = results["nv_sren"]["backwards_eval"]

plt.figure(figsize=(4.2, 2.5))
plt.plot(x, y_ren_fwd, color=color_r, linestyle=ls_fwd, label="REN (Forward)")
plt.plot(x, y_ren_bck, color=color_r, linestyle=ls_bck, label="REN (Backward)")
plt.plot(x, y_sren_fwd, color=color_s, linestyle=ls_fwd, label="Scalable REN (Forward)")
plt.plot(x, y_sren_bck, color=color_s, linestyle=ls_bck, label="Scalable REN (Backward)")

format_plot("Number of model parameters", "Evaluation time (s)", "modelsize", x)

# Also print hidden layer sizes vs. nv just for my interest
nv_ren = results["nv_ren"]["nv"]
nv_sren = results["nv_sren"]["nv"]
nh_sren = results["nv_sren"]["nh"]

for k in range(len(nv_ren)):
    print("REN nv: ", nv_ren[k], "\tS-REN nv: ", nv_sren[k], "\tS-REN nh: ", nh_sren[k])
