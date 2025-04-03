import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Sequence

from utils.plot_utils import startup_plotting
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent


def get_loss_key(experiment):
    if experiment == "youla":
        loss_key = "test_loss"
    elif experiment == "pde":
        loss_key = "mean_loss"
    elif experiment == "f16":
        loss_key = "train_loss"
    return loss_key


def get_reward_data(data: list, experiment: str):
    
    # Get basic loss data
    losses = np.array([d["results"][get_loss_key(experiment)] for d in data])
    times = np.array([d["results"]["times"] for d in data])
    
    # Store time delta
    # NOTE: This includes the JITted time (first delta)
    times = (times[:, :] - times[:, 0:1])
    times = np.vectorize(lambda td: td.total_seconds())(times)
    
    # Interpolate over time
    npoints = len(losses[0])
    time = np.linspace(0, times.max(), npoints)
    time_losses = np.array([
        np.interp(time, times[k], losses[k])
        for k in range(times.shape[0])
    ])
    
    # Return aggregated results
    return {
        "losses": losses.mean(axis=0),
        "stdev": losses.std(axis=0),
        "max": losses.max(axis=0),
        "min": losses.min(axis=0),
        "time": time,
        "time_losses": time_losses.mean(axis=0),
        "time_stdev": time_losses.std(axis=0),
        "time_max": time_losses.max(axis=0),
        "time_min": time_losses.min(axis=0),
    }
    

def aggregate_results(
    experiment: str, 
    key: str, 
    opts: Sequence[str],
    fixed: dict = {},
):
    """
    Aggregate results for different model/hyperparam combinations.
    """
    # Read in the pickle files for this experiment
    data = []
    fpath = dirpath / f"../results/{experiment}/"
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
        results[opts[k]] = get_reward_data(_group_data, experiment)
        
    return results, group_data


def format_plot(xlabel, ylabel, filename_suffix, x1, x2, yscale):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.xlim(min([min(x1), min(x2)]), max([max(x1), max(x2)]))
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.75)
    plt.tight_layout()
    plt.savefig(dirpath / f"../paperfigs/performance/{filename_suffix}.pdf")
    plt.close()
    

def plot_results(experiment, ylabel, yscale="log"):
    
    # Get data aggregated for each model, and make sure
    # they're all using the same init_method (i.e., don't load other files)
    model_results, _ = aggregate_results(
        experiment, 
        key="network", 
        opts=["contracting_ren", "contracting_r2dn"],
        fixed={"init_method": "long_memory"}
    )

    # Choose colours
    color_r = "#009E73"
    color_s = "#D55E00"

    # Make loss plots for different models
    y1 = model_results["contracting_ren"]["losses"]
    y2 = model_results["contracting_r2dn"]["losses"]
    y1min = model_results["contracting_ren"]["max"]
    y1max = model_results["contracting_ren"]["min"]
    y2min = model_results["contracting_r2dn"]["max"]
    y2max = model_results["contracting_r2dn"]["min"]
    x = np.arange(len(y1))

    plt.figure(figsize=(4, 2.7))
    plt.plot(x, y1, color=color_r, label="REN")
    plt.plot(x, y2, color=color_s, label="R2DN")
    plt.fill_between(x, y1min, y1max, alpha=0.2, color=color_r)
    plt.fill_between(x, y2min, y2max, alpha=0.2, color=color_s)
    
    format_plot("Training epochs", ylabel, f"{experiment}_loss", x, x, yscale)
    
    # Now do loss vs. time plots
    y1 = model_results["contracting_ren"]["time_losses"]
    y2 = model_results["contracting_r2dn"]["time_losses"]
    y1min = model_results["contracting_ren"]["time_max"]
    y1max = model_results["contracting_ren"]["time_min"]
    y2min = model_results["contracting_r2dn"]["time_max"]
    y2max = model_results["contracting_r2dn"]["time_min"]
    x1 = model_results["contracting_ren"]["time"]
    x2 = model_results["contracting_r2dn"]["time"]
    
    plt.figure(figsize=(4, 2.7))
    plt.plot(x1, y1, color=color_r, label="REN")
    plt.plot(x2, y2, color=color_s, label="R2DN")
    plt.fill_between(x1, y1min, y1max, alpha=0.2, color=color_r)
    plt.fill_between(x2, y2min, y2max, alpha=0.2, color=color_s)
    
    # Plot a marker at the end to signify training is done
    plt.scatter(x1[-1:], y1[-1:], c=color_r, s=20, marker="o")
    plt.scatter(x2[-1:], y2[-1:], c=color_s, s=20, marker="o")
    
    # ax = plt.gca()
    # ax.figure.set_size_inches(3.5, 2.5)

    format_plot("Elapsed training time (s)", ylabel, f"{experiment}_timeloss", x1, x2, yscale)


plot_results("f16", "Training loss")
plot_results("pde", "Training loss")
plot_results("youla", "Test loss", "linear")


# ---------------------------------------------------------------
# Compute test metrics
# ---------------------------------------------------------------

def print_test_results(experiment):
    
    # Get data
    models = ["contracting_ren", "contracting_r2dn"]
    _, data = aggregate_results(
        experiment, 
        key="network", 
        opts=models,
        fixed={"init_method": "long_memory"}
    )
    
    # Loop and print
    for model in models:
        N = len(data[model])
        
        # Test error
        if experiment in ["pde", "f16"]:
            test_data = np.array([
                data[model][k]["results"]["nrmse"] 
                for k in range(N)
            ])
        elif experiment == "youla":
            test_data = np.array([
                data[model][k]["results"]["test_loss"][-1] 
                for k in range(N)
            ])
        test_data *= 100
        
        # Timing data (ignore first time which includes JIT compilation)
        time_data = np.zeros(N)
        for k in range(N):
            t = np.array(data[model][k]["results"]["times"])[1:]
            t = np.vectorize(lambda td: td.total_seconds())(t - t[0])
            time_data[k] = np.mean(np.diff(t))
            
        print(f"{model}:\t {time_data.mean():.3g}\t {time_data.std():.2g}\t "+
              f"{test_data.mean():.3g}\t {test_data.std():.2g}")

experiment = ["f16", "pde", "youla"]
for e in experiment:
    print_test_results(e)
