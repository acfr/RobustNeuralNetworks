import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from utils.plot_utils import startup_plotting
from utils import utils

startup_plotting()
dirpath = Path(__file__).resolve().parent


def get_raw_results(data):
    nrmse = 100 * np.array([d["results"]["val_nrmse"] for d in data])
    return {
        "nrmse": nrmse,
        "express": 1/nrmse,
        "mse": np.array([d["results"]["val_mse"] for d in data]),
        "size": np.array([d["results"]["num_params"] for d in data]),
        "time_fwd": np.array([d["results"]["forwards_eval"] for d in data]),
        "time_bck": np.array([d["results"]["backwards_eval"] for d in data]),
    }
    
def aggregate_results(data):
    
    # Important data
    nrmse = 100 * np.array([d["results"]["val_nrmse"] for d in data])
    mse = np.array([d["results"]["val_mse"] for d in data])
    size = np.array([d["results"]["num_params"] for d in data])
    time_fwd = np.array([d["results"]["forwards_eval"] for d in data])
    time_bck = np.array([d["results"]["backwards_eval"] for d in data])
    
    # Calculate expressivity
    expressivity = 1 / nrmse
    
    # Aggregate and store
    return {
        "express_mean": np.mean(expressivity),
        "express_std": np.std(expressivity),
        "nrmse_mean": np.mean(nrmse),
        "nrmse_std": np.std(nrmse),
        "mse_mean": np.mean(mse),
        "mse_std": np.std(mse),
        "size_mean": np.mean(size),
        "size_std": np.std(size),
        "time_fwd_mean": np.mean(time_fwd),
        "time_fwd_std": np.std(time_fwd),
        "time_bck_mean": np.mean(time_bck),
        "time_bck_std": np.std(time_bck),
    }
    
def get_sizes(config):
    if config["network"] == "contracting_ren":
        return config["nv"]
    if config["network"] == "scalable_ren":
        return config["nh"][0]

def read_results():
    """
    Aggregate results for different model/hyperparam combinations.
    """
    
    opts = ["contracting_ren", "scalable_ren"]
    
    # Read in all the pickle files
    data = []
    fpath = dirpath / f"../results/expressivity/"
    # fpath = dirpath / f"../results/expressivity-paper/expressivity-final/"
    files = [f for f in fpath.iterdir() if f.is_file() and not (f.suffix == ".pdf")]
    for f in files:
        d = utils.load_results(f)
        data.append({"config": d[0], "results": d[2]})
        
    # Get mean/std results for each network structure and model size (resp.)
    results = {}
    for network in opts:
        network_data = [d for d in data if d["config"]["network"] == network]
        net_results = []
        sizes = np.unique([get_sizes(d["config"]) for d in network_data])
        for n in sizes:
            net_size_data = [d for d in network_data if get_sizes(d["config"]) == n]
            net_results.append(aggregate_results(net_size_data))
        results[network] = utils.list_to_dicts(net_results)
        
    # Also get all results for each model just to inspect the raw data
    raw_results = {}
    for network in opts:
        network_data = [d for d in data if d["config"]["network"] == network]
        raw_results[network] = get_raw_results(network_data)
        
    return results , raw_results

def plot_results():
    
    # Get data aggregated for each model, and make sure
    # they're all using the same init_method (i.e., don't load other files)
    results, raw_results = read_results()

    # Choose colours
    color_r = "#009E73"
    color_s = "#D55E00"

    # Plot accuracy vs number of params
    x1 = raw_results["contracting_ren"]["size"]
    x2 = raw_results["scalable_ren"]["size"]
    y1 = raw_results["contracting_ren"]["nrmse"]
    y2 = raw_results["scalable_ren"]["nrmse"]

    plt.figure(figsize=(4.5, 3.5))
    plt.scatter(x1, y1, marker="x", color=color_r, label="REN")
    plt.scatter(x2, y2, marker="+", color=color_s, label="R2DN")
    
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
    
    # Raw data
    x1_raw = raw_results["contracting_ren"]["express"]
    x2_raw = raw_results["scalable_ren"]["express"]
    y1_raw = raw_results["contracting_ren"]["time_fwd"]
    y2_raw = raw_results["scalable_ren"]["time_fwd"]
    
    # Aggregated data
    x1 = results["contracting_ren"]["express_mean"]
    x2 = results["scalable_ren"]["express_mean"]
    y1_fwd = results["contracting_ren"]["time_fwd_mean"]
    y2_fwd = results["scalable_ren"]["time_fwd_mean"]
    
    # Errors
    x1_std = results["contracting_ren"]["express_std"]
    x2_std = results["scalable_ren"]["express_std"]
    y1_fwd_std = results["contracting_ren"]["time_fwd_std"]
    y2_fwd_std = results["scalable_ren"]["time_fwd_std"]
    
    # Log transform
    x1_log = np.log(x1_raw)
    x2_log = np.log(x2_raw)
    y1_log = np.log(y1_raw)
    y2_log = np.log(y2_raw)
    
    # Lines of best fit (fit to raw data)
    x1_fit = np.linspace(0.8*min(x1), 1.2*max(x1), 100)
    x2_fit = np.linspace(0.8*min(x2), 1.2*max(x2), 100)
    p1_fwd = np.polyfit(x1_log, y1_log, 1)
    p2_fwd = np.polyfit(x2_log, y2_log, 1)
    Y1_fwd = np.exp(np.polyval(p1_fwd, np.log(x1_fit)))
    Y2_fwd = np.exp(np.polyval(p2_fwd, np.log(x2_fit)))
    
    # Slope errors
    e1 = np.sum((y1_log - np.polyval(p1_fwd, x1_log))**2)
    e2 = np.sum((y2_log - np.polyval(p2_fwd, x2_log))**2)
    
    s1 = np.sqrt(e1 / (np.sum((x1_log - x1_log.mean())**2) * (len(x1_log) - 2)))
    s2 = np.sqrt(e2 / (np.sum((x2_log - x2_log.mean())**2) * (len(x2_log) - 2)))

    # Plotting
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(x1_fit, Y1_fwd, linestyle="dotted", color=color_r)
    plt.plot(x2_fit, Y2_fwd, linestyle="dotted", color=color_s)
    plt.errorbar(
        x1, y1_fwd, xerr=x1_std, yerr=y1_fwd_std,  marker="D", 
        color=color_r, label="REN", linestyle="none", elinewidth=0.8
    )
    plt.errorbar(
        x2, y2_fwd, xerr=x2_std, yerr=y2_fwd_std,  marker="^", 
        color=color_s, label="R2DN", linestyle="none", elinewidth=0.8
    )
    
    # Annotate slopes
    plt.annotate(
    f"slope = {p1_fwd[0]:.2f} $\pm$ {s1:.2f}",
    xy=(0.015, 2e-2), 
    # xy=(0.018, 5e-2), 
    xycoords='data',
    fontsize=12,
    )
    plt.annotate(
        f"slope = {p2_fwd[0]:.3f} $\pm$ {s2:.3f}",
        xy=(0.4, 8.5e-3), 
        # xy=(0.4, 2.2e-2), 
        xycoords='data',
        fontsize=12,
    )
    
    # Plot formatting
    plt.xlabel("Model expressivity (NRMSE$^{-1}$)")
    plt.ylabel("Inference time (s)")
    # plt.ylabel("Backpropagagion time (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(4.5e-3, 5e-2)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.75)
    plt.tight_layout()
    plt.savefig(dirpath / f"../paperfigs/timing/expressivity_time.pdf")
    plt.close()
    
    
plot_results()