# Robust Neural Networks

This repository contains a collection or robust neural network architectures developed at the Australian Centre For Robotics (ACFR). All networks are implemented in Python/JAX.

Implemented network architectures currently include:

- Lipschitz-bounded Sandwich MLPs from [Wang & Manchester (ICML 2023)](https://proceedings.mlr.press/v202/wang23v.html). Tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acfr/RobustNeuralNetworks/blob/main/examples/sandwich_mnist.ipynb)

- Recurrent Equilibrium Network (REN) from [Revay, Wang, & Manchester (TAC 2023)](https://ieeexplore.ieee.org/document/10179161).

- **[WIP]** Monotone, Bi-Lipschitz (BiLipNet), and Polyak-Lojasiewicz networks (PLNet) from [Wang, Dvijotham, & Manchester (ICML 2024)](https://proceedings.mlr.press/v235/wang24p.html).

- Robust Recurrent Deep Network (R2DN) from [Barbara, Wang, & Manchester (CDC 2026)](https://arxiv.org/abs/2504.01250).

This repository and README are a work-in-progress. More network architectures, tutorials, and documentation will be added as we go along. 

## Installation for Development

All dependencies are managed with [uv](https://docs.astral.sh/uv/). To install uv, run the following (Mac/Linux, see the [docs](https://docs.astral.sh/uv/getting-started/installation/) for Windows).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, to install the package and all of its dependencies, open a terminal in the root directory of this repository and enter the following command.

```bash
./install.sh
```

This will create a Python virtual environment at `./.venv`, install the pinned Python version, and install all dependencies. The script checks whether CUDA is available on your machine and installs the corresponding JAX/PyTorch version accordingly. To check that everything works, run the test suite:

```bash
./run_tests.sh
```

### A Note on Dependencies

Dependencies are declared in `pyproject.toml` and are split into two sets:

- **Core dependencies:** required by the `robustnn` package itself.
- **Extra dependencies for `examples`:** all additional dependencies needed to run the demos in `examples/` and the scripts in `test/`. 

By default, the `install.sh` script installs boht the core and extra dependencies, and checks for CUDA to install the correct versions of `jax` and `torch`. If you would rather just install the minimal `robustnn` package itself, do not run the `install.sh` script, and instead run one of the following.

```bash
uv sync --extra cpu     # CPU only
uv sync --extra cuda13  # GPU with CUDA 13
```

## Running an Example

Once you have installed the package as above, use `uv run` to run any of the scripts in the `examples/` folder. For example, from the root directory of the project, run:

```bash
uv run python examples/sandwich_mnist.py
```

## Contact

Please raise an [issue](https://github.com/acfr/RobustNeuralNetworks/issues) if you have any questions, and we'll do our best to get back to you as soon as we can!
