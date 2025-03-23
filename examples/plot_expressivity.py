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
        "nrmse": np.array([d["results"]["train_loss"][-1] for d in data]),
        "mse": np.array([d["results"]["val_mse"] for d in data]),
        "size": np.array([d["results"]["num_params"] for d in data]),
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
    plt.scatter(x1, y1, marker="X", color=color_r, label="REN")
    plt.scatter(x2, y2, marker="+", color=color_s, label="Scalable REN")
    
    plt.xlabel("Model size")
    plt.ylabel("NRMSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min([min(x1), min(x2)]), max([max(x1), max(x2)]))
    plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.25))
    plt.grid(True, which='both', linestyle=':', linewidth=0.75)
    plt.tight_layout()
    plt.savefig(dirpath / f"../paperfigs/timing/expressivity.pdf")
    plt.close()
    
    
plot_results()