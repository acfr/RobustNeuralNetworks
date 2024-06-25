# Lipschitz Bounded Networks

This repository containts a collection of Lipschitz bounded networks to use for our research at the ACFR. All networks are implemented in Python/JAX.

The intention is to have a common library of network implementations so that we can avoid repeated code. This way, we can all contribute to making the networks more efficient and sufficiently modular for use in all of our research projects.

This repository is a work-in-progress.

## Installation

First, [create a virtual environment](https://docs.python.org/3/library/venv.html). Activate the virtual environment and install all dependencies in `requirements.txt` with

    python -m pip install -r requirements.txt
    pip install -e .

The second line installs the local package `lbnn` itself. For Mac, you may need to use `python3` instead of `python` here. The `requirements.txt` file was generated with [`pipreqs`](https://github.com/bndr/pipreqs). If you want to run JAX on an NVIDIA GPU, you'll also need to do the following:

    pip install -U "jax[cuda12_pip]"

