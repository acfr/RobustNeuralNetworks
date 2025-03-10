import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Sequence

from utils.plot_utils import startup_plotting
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

# For each experiment:
#
#   Next steps will be to add the following:
#       - Plot aggregated mean loss vs. time
#         (will require interpolation)
# Let's do this.

def get_loss_key(experiment):
    if experiment == "youla":
        loss_key = "test_loss"
    elif experiment == "pde":
        loss_key = "mean_loss"
    elif experiment == "f16":
        loss_key = "train_loss"
    return loss_key


def get_reward_data(data: list, experiment: str):
    losses = np.array([d["results"][get_loss_key(experiment)] for d in data])
    times = np.array([d["results"]["times"] for d in data])
    
    return {
        "losses": losses.mean(axis=0),
        "stdev": losses.std(axis=0),
        "max": losses.max(axis=0),
        "min": losses.min(axis=0),
        "times": times # TODO: interpolate times and aggregate
    }
    

def aggregate_results(experiment: str, key: str, opts: Sequence[str]):
    
    # Read in the pickle files for this experiment
    data = []
    fpath = dirpath / f"../results/{experiment}/"
    files = [f for f in fpath.iterdir() if f.is_file() and not (f.suffix == ".pdf")]
    for f in files:
        d = utils.load_results(f)
        data.append({"config": d[0], "results": d[2]})
    
    # Separate the data into multiple groups for comparison
    results = {}
    n_groups = len(opts)
    for k in range(n_groups):
        group_data = [d for d in data if (d["config"][key] == opts[k])]
        results[opts[k]] = get_reward_data(group_data, experiment)
        
    # Return data for plotting
    return results

    
def plot_results(experiment, ylabel ):
    
    model_results = aggregate_results(
        experiment, key="network", opts=["contracting_ren", "scalable_ren"]
    )

    # Choose colours
    color_r = "#009E73"
    color_s = "#D55E00"

    # Make model plots
    y1 = model_results["contracting_ren"]["losses"]
    y2 = model_results["scalable_ren"]["losses"]
    y1min = model_results["contracting_ren"]["max"]
    y1max = model_results["contracting_ren"]["min"]
    y2min = model_results["scalable_ren"]["max"]
    y2max = model_results["scalable_ren"]["min"]
    x = np.arange(len(y1))

    plt.figure(figsize=(4.5, 3))
    plt.plot(x, y1, color=color_r, label="REN")
    plt.plot(x, y2, color=color_s, label="Scalable REN")
    plt.fill_between(x, y1min, y1max, alpha=0.2, color=color_r)
    plt.fill_between(x, y2min, y2max, alpha=0.2, color=color_s)

    plt.xlabel("Training epochs")
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.xlim(min(x), max(x))
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.75)
    plt.tight_layout()
    plt.savefig(dirpath / f"../paperfigs/performance/{experiment}_loss.pdf")
    plt.close()


plot_results("f16", "Training loss")
plot_results("pde", "Training loss")
plot_results("youla", "Test loss")