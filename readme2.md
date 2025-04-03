# Robust Recurrent Deep Networks (R2DN)

This repository contains the code associated with the paper *Scalable Parameterization of Contracting and Lipschitz Recurrent Deep Networks*.

Included are JAX implementations of each of the following robust neural models:

- The Sandwich layer and corresponding Lipschitz Bounded Deep Network from [Wang & Manchester (ICML 2023)]().
- Contracting, Lipschitz, and (Q, S, R)-ribust Recurrent Equilibrium Networks (RENs) from [Revay, Wang, & Manchester (TAC 2023)]().
- Contracting Robust Recurrent Deep Networks (R2DNs) from our current work.

Robust neural models are included in the `robustnn/` directory. Scripts used to generate the results in the paper are in the `examples/` directory.

## Installation

Create a [virtual python environment](https://docs.python.org/3/library/venv.html). Activate the virtual environment and install all dependencies in `requirements.txt` with

    python -m pip install -r requirements.txt
    pip install -e .

The second line installs the local package `robustnn` itself. The `requirements.txt` file was generated with [`pipreqs`](https://github.com/bndr/pipreqs). If you want to run JAX on an NVIDIA GPU, you'll also need to do the following:

    pip install -U "jax[cuda12_pip]"

## Reproducing the Results

Simply run the `run.sh` script in the root directory of this repository to run all the experiments, process the results, and reproduce the figures from the paper.