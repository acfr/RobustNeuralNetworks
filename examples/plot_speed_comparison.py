import matplotlib.pyplot as plt

from pathlib import Path
from utils.plot_utils import startup_plotting

import utils.speed as utils

startup_plotting()
dirpath = Path(__file__).resolve().parent

# Load the saved data
filename = "timing_results"
filepath = dirpath / f"../results/timing/"
results = utils.load_results(filepath / f"{filename}.pickle")

# Plot eval time vs. sequence length (forward)
x = results["horizon_ren"]["horizon"]
y_ren = results["horizon_ren"]["forwards_eval"]
y_sren = results["horizon_sren"]["forwards_eval"]

plt.plot(x, y_ren, label="REN")
plt.plot(x, y_sren, label="Scalable REN")
plt.xlabel("Sequence length")
plt.ylabel("Evaluation time (s)")
plt.title("Forward Pass")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig(filepath / f"{filename}_sequence_forward.pdf")
plt.close()

# ...and backwards
y_ren = results["horizon_ren"]["backwards_eval"]
y_sren = results["horizon_sren"]["backwards_eval"]

plt.plot(x, y_ren, label="REN")
plt.plot(x, y_sren, label="Scalable REN")
plt.xlabel("Sequence length")
plt.ylabel("Evaluation time (s)")
plt.title("Backwards Pass")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig(filepath / f"{filename}_sequence_backwards.pdf")
plt.close()

# Plot eval time vs. batch size (forward)
x = results["batches_ren"]["batches"]
y_ren = results["batches_ren"]["forwards_eval"]
y_sren = results["batches_sren"]["forwards_eval"]

plt.plot(x, y_ren, label="REN")
plt.plot(x, y_sren, label="Scalable REN")
plt.xlabel("Batch size")
plt.ylabel("Evaluation time (s)")
plt.title("Forward Pass")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig(filepath / f"{filename}_batches_forward.pdf")
plt.close()

# ...and backwards
y_ren = results["batches_ren"]["backwards_eval"]
y_sren = results["batches_sren"]["backwards_eval"]

plt.plot(x, y_ren, label="REN")
plt.plot(x, y_sren, label="Scalable REN")
plt.xlabel("Batch size")
plt.ylabel("Evaluation time (s)")
plt.title("Backwards Pass")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig(filepath / f"{filename}_batches_backwards.pdf")
plt.close()

# Plot eval time vs. number of model params (forward)
# x = results["nv_ren"]["nv"]
x = results["nv_ren"]["num_params"]
y_ren = results["nv_ren"]["forwards_eval"]
y_sren = results["nv_sren"]["forwards_eval"]

plt.plot(x, y_ren, label="REN")
plt.plot(x, y_sren, label="Scalable REN")
plt.xlabel("Model size")
plt.ylabel("Evaluation time (s)")
plt.title("Forward Pass")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig(filepath / f"{filename}_nv_forward.pdf")
plt.close()

# ...and backwards
y_ren = results["nv_ren"]["backwards_eval"]
y_sren = results["nv_sren"]["backwards_eval"]

plt.plot(x, y_ren, label="REN")
plt.plot(x, y_sren, label="Scalable REN")
plt.xlabel("Model size")
plt.ylabel("Evaluation time (s)")
plt.title("Backwards Pass")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig(filepath / f"{filename}_nv_backwards.pdf")
plt.close()

# Also print hidden layer sizes vs. nv just for my interest
nv_ren = results["nv_ren"]["nv"]
nv_sren = results["nv_sren"]["nv"]
nh_sren = results["nv_sren"]["nh"]

for k in range(len(nv_ren)):
    print("REN nv: ", nv_ren[k], "\tS-REN nv: ", nv_sren[k], "\tS-REN nh: ", nh_sren[k])
