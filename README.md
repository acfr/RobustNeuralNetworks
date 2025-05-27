# Robust Neural Networks

This repository contains a collection or robust neural network architectures developed at the Australian Centre For Robotics (ACFR). All networks are implemented in Python/JAX.

Implemented networks architectures include:

- Lipschitz-bounded Sandwich MLPs from [Wang & Manchester (ICML 2023)](https://proceedings.mlr.press/v202/wang23v.html).

- Recurrent Equilibrium Network (REN) from [Revay, Wang, & Manchester (TAC 2023)](https://ieeexplore.ieee.org/document/10179161).

- **[WIP]** Monotone, Bi-Lipschitz (BiLipNet), and Polyak-Lojasiewicz networks (PLNet) from [Wang, Dvijotham, & Manchester (ICML 2024)](https://proceedings.mlr.press/v235/wang24p.html).

- Robust Recurrent Deep Network (R2DN) from [Barbara, Wang, & Manchester (arXiv 2025)](https://arxiv.org/abs/2504.01250).

This repository (and README) are a work-in-progress. More network architectures will be added as we go along. 

## Installation

To install the required dependencies, open a terminal in the root directory of this repository and enter the following commands.

```bash
    ./install.sh
```

This will create a Python virtual environment at ./venv and install all dependencies.

### A Note on Dependencies

All code was tested and developed in Ubuntu 22.04 with CUDA 12.4 and Python 3.10.12. 

Requirements were generated with [`pipreqs`](https://github.com/bndr/pipreqs). The ```install.sh``` will check for whether CUDA is available for your machine, and install the corresponding jax package. 
