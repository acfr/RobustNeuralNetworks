import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Sequence

from utils.plot_utils import startup_plotting
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent


def get_results(data):
    return {
        "nrmse": 100 * np.array([d["results"]["val_nrmse"] for d in data]),
        "mse": np.array([d["results"]["val_mse"] for d in data]),
        "size": np.array([d["results"]["num_params"] for d in data]),
        "time_fwd": np.array([d["results"]["forwards_eval"] for d in data]),
        "time_bck": np.array([d["results"]["backwards_eval"] for d in data]),
    }

def read_results(
    key: str, 
    opts: Sequence[str],
    fixed: dict = {},
):
    """
    Aggregate results for different model/hyperparam combinations.
    """
    # Read in the pickle files for this experiment
    data = []
    fpath = dirpath / f"../results/expressivity/"
    files = [f for f in fpath.iterdir() if f.is_file() and not (f.suffix == ".pdf")]
    for f in files:
        d = utils.load_results(f)
        data.append({"config": d[0], "results": d[2]})
    
    # Separate the data into multiple groups for comparison
    # This allows us to pick and choose combinations of hyperparams
    # to keep fixed or vary, depending on what we want to compare
    results, group_data = {}, {}
    n_groups = len(opts)
    for k in range(n_groups):
        _group_data = [
            d for d in data if (d["config"][key] == opts[k] and all(
                [d["config"][fk] == fixed[fk] for fk in fixed])
            )
        ]
        group_data[opts[k]] = _group_data
        results[opts[k]] = get_results(_group_data)
        
    return results, group_data

def plot_results():
    
    # Get data aggregated for each model, and make sure
    # they're all using the same init_method (i.e., don't load other files)
    model_results, _ = read_results(
        key="network", 
        opts=["contracting_ren", "scalable_ren"],
    )
    
    # Choose colours
    color_r = "#009E73"
    color_s = "#D55E00"

    # Plot accuracy vs number of params
    x1 = model_results["contracting_ren"]["size"]
    x2 = model_results["scalable_ren"]["size"]
    y1 = model_results["contracting_ren"]["nrmse"]
    y2 = model_results["scalable_ren"]["nrmse"]

    plt.figure(figsize=(4.5, 3.5))
    plt.scatter(x1, y1, marker="x", color=color_r, label="REN")
    plt.scatter(x2, y2, marker="+", color=color_s, label="Scalable REN")
    
    plt.xlabel("Model size")
    plt.ylabel("NRMSE (\%)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min([min(x1), min(x2)]), max([max(x1), max(x2)]))
    plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.25))
    plt.grid(True, which='both', linestyle=':', linewidth=0.75)
    plt.tight_layout()
    plt.savefig(dirpath / f"../paperfigs/timing/expressivity.pdf")
    plt.close()
    
    # ----------- Plot accuracy vs number of params -----------
    
    # Get data
    x1 = 1 / model_results["contracting_ren"]["nrmse"]
    x2 = 1 / model_results["scalable_ren"]["nrmse"]
    y1_fwd = model_results["contracting_ren"]["time_fwd"]
    y2_fwd = model_results["scalable_ren"]["time_fwd"]
    # y1_bck = model_results["contracting_ren"]["time_bck"]
    # y2_bck = model_results["scalable_ren"]["time_bck"]
    
    # Lines of best fit
    p1_fwd = np.polyfit(np.log(x1), np.log(y1_fwd), 1)
    p2_fwd = np.polyfit(np.log(x2), np.log(y2_fwd), 1)
    # p1_bck = np.polyfit(np.log(x1), np.log(y1_bck), 1)
    # p2_bck = np.polyfit(np.log(x2), np.log(y2_bck), 1)
    x = np.linspace(min([min(x1), min(x2)]), max([max(x1), max(x2)]), 100)
    Y1_fwd = np.exp(np.polyval(p1_fwd, np.log(x)))
    Y2_fwd = np.exp(np.polyval(p2_fwd, np.log(x)))
    # Y1_bck = np.exp(np.polyval(p1_bck, np.log(x)))
    # Y2_bck = np.exp(np.polyval(p2_bck, np.log(x)))

    # Plotting
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(x, Y1_fwd, linestyle="dotted", color=color_r)
    plt.plot(x, Y2_fwd, linestyle="dotted", color=color_s)
    plt.scatter(x1, y1_fwd, marker="x", color=color_r, label="REN")
    plt.scatter(x2, y2_fwd, marker="+", color=color_s, label="R2DN")

    # plt.plot(x, Y1_bck, linestyle="dashed", color=color_r)
    # plt.plot(x, Y2_bck, linestyle="dashed", color=color_s)
    # plt.scatter(x1, y1_bck, marker="X", color=color_r, label="REN (Backward)")
    # plt.scatter(x2, y2_bck, marker="P", color=color_s, label="R2DN (Backward)")
    
    plt.annotate(
    f"slope = {p1_fwd[0]:.2f}",
    xy=(1, 2.5e-2), 
    xycoords='data',
    fontsize=12,
    )
    plt.annotate(
        f"slope = {p2_fwd[0]:.2f}",
        xy=(1, 3.3e-3), 
        xycoords='data',
        fontsize=12,
    )
    
    plt.xlabel("Model expressivity (1 / NRMSE)")
    plt.ylabel("Evaluation time (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.75)
    plt.tight_layout()
    plt.savefig(dirpath / f"../paperfigs/timing/expressivity_time.pdf")
    plt.close()
    
    
plot_results()