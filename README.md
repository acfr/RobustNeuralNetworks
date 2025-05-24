# Robust Neural Networks

This repository will contain a collection of our robust neural networks, all implemented in Python/JAX. 

The intention is for this to be the one and only set of implementations for (eg) REN, LBDN, BiLipNet, PLNet, etc. to make sure we all use up-to-date versions of the code. We will also publish this repository and use it to distribute to the community alongside various research papers.

This repository and README are a work-in-progress.

[NOTE]: This branch includes experimental code to initialise RENs and R2DNs from explicit models. It's a WIP.

## Installation

First, [create a virtual environment](https://docs.python.org/3/library/venv.html). Activate the virtual environment and install all dependencies in `requirements.txt` with

    python -m pip install -r requirements.txt
    pip install -e .

The second line installs the local package `lbnn` itself. The `requirements.txt` file was generated with [`pipreqs`](https://github.com/bndr/pipreqs). If you want to run JAX on an NVIDIA GPU, you'll also need to do the following:

    pip install -U "jax[cuda12_pip]"

