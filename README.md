# Robust Neural Networks

This repository contains a collection or robust neural network architectures developed at the Australian Centre For Robotics (ACFR). All networks are implemented in Python/JAX.

Implemented network architectures include:

- Lipschitz-bounded Sandwich MLPs from [Wang & Manchester (ICML 2023)](https://proceedings.mlr.press/v202/wang23v.html). Tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acfr/RobustNeuralNetworks/blob/main/examples/sandwich_mnist.ipynb)

- Recurrent Equilibrium Network (REN) from [Revay, Wang, & Manchester (TAC 2023)](https://ieeexplore.ieee.org/document/10179161).

- **[WIP]** Monotone, Bi-Lipschitz (BiLipNet), and Polyak-Lojasiewicz networks (PLNet) from [Wang, Dvijotham, & Manchester (ICML 2024)](https://proceedings.mlr.press/v235/wang24p.html).

- Robust Recurrent Deep Network (R2DN) from [Barbara, Wang, & Manchester (arXiv 2025)](https://arxiv.org/abs/2504.01250).

This repository is a work-in-progress. More network architectures, tutorials, and documentation will be added as we go along. 

## Installation for Development

All dependencies are managed with [uv](https://docs.astral.sh/uv/). To install uv, run the following (Mac/Linux, see the [docs](https://docs.astral.sh/uv/getting-started/installation/) for Windows).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, to install the package and all of its dependencies, open a terminal in the root directory of this repository and enter the following command.

```bash
./install.sh
```

This will create a Python virtual environment at `./.venv`, install the pinned Python version, and install all dependencies. The script checks whether CUDA is available on your machine and installs the corresponding jax package. To check that everything works, run the test suite:

```bash
./run_tests.sh
```

### A Note on Dependencies

Dependencies are declared in `pyproject.toml` and locked to exact versions in `uv.lock`, which is committed to the repository so that everyone builds the same environment. They are split into two sets:

- The **core dependencies** are those required by the `robustnn` package itself (`jax`, `flax`, `numpy`, and `torch`). These are the only packages a user installing `robustnn` will get.
- The **`examples` extra** adds everything needed to run the demos in `examples/` and the scripts in `test/` (`matplotlib`, `optax`, `pandas`, `scipy`, `tensorflow`, etc.). The `install.sh` script installs this extra by default for development.

There is also a `cuda12` extra, which `install.sh` selects automatically when CUDA is detected. If you would rather manage the environment yourself, the equivalent commands are:

```bash
uv sync --extra examples                # CPU
uv sync --extra examples --extra cuda12 # CUDA 12
```

All code was tested and developed in Ubuntu 22.04 with CUDA 12.4 and Python 3.12.

## Running an Example

Once you have installed the package as above, use `uv run` to run any of the scripts in the `examples/` folder. For example, from the root directory of the project, run:

```bash
uv run python examples/sandwich_mnist.py
```

There is no need to activate the virtual environment first; `uv run` does it for you. If you would prefer to activate it manually, use `source .venv/bin/activate`.

## Contact

Please contact Nicholas Barbara (nicholas.barbara@sydney.edu.au) with any questions.
