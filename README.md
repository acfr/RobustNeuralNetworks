# Robust Neural Networks

This repository contains a collection of robust neural network architectures developed at the Australian Centre For Robotics (ACFR). All networks are implemented in Python with PyTorch and/or JAX.

Implemented network architectures currently include:

- Lipschitz-bounded Sandwich MLPs from [Wang & Manchester (ICML 2023)](https://proceedings.mlr.press/v202/wang23v.html). Tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acfr/RobustNeuralNetworks/blob/main/examples/sandwich_mnist.ipynb)

- Recurrent Equilibrium Network (REN) from [Revay, Wang, & Manchester (TAC 2023)](https://ieeexplore.ieee.org/document/10179161).

- Monotone, Bi-Lipschitz (BiLipNet), and Polyak-Lojasiewicz networks (PLNet) from [Wang, Dvijotham, & Manchester (ICML 2024)](https://proceedings.mlr.press/v235/wang24p.html).

- Robust Recurrent Deep Network (R2DN) from [Barbara, Wang, & Manchester (CDC 2026)](https://arxiv.org/abs/2504.01250).

This repository and README are a work-in-progress. More network architectures, tutorials, and documentation will be added as we go along. 

## Installation for Development

All dependencies are managed with [uv](https://docs.astral.sh/uv/). To install uv, run the following (Mac/Linux, use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) for Windows).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, to install the package and all of its dependencies, open a terminal in the root directory of this repository and enter the following command.

```bash
./install.sh
```

This will create a Python virtual environment at `./.venv`, install the pinned Python version, and install all dependencies. The script checks whether CUDA is available on your machine and installs the corresponding JAX/PyTorch version accordingly. To check your installation, run the test suite:

```bash
./run_tests.sh
```

### Testing

Tests are written with [pytest](https://pytest.org) and live in `tests/`. Each network has a single test file (e.g. `test_plnet.py`, `test_bilipschitz_ren.py`) whose tests run against both the JAX/Flax and PyTorch implementations through the `backend` fixture (see `tests/backends.py`). Tests for networks or features that only exist in JAX use the `jax_backend` fixture, and the PyTorch side is skipped with a `TODO(torch)` note in the file. `./run_tests.sh` is a thin wrapper around `uv run pytest`, so any pytest arguments work:

```bash
./run_tests.sh                  # everything
./run_tests.sh -m torch         # PyTorch tests only (or -m jax)
./run_tests.sh -n auto          # run in parallel
./run_tests.sh -k plnet         # select by name
```

Besides property tests (output shapes, inverse round-trips, Lipschitz bounds, gradients, error handling), every network is checked against **golden input/output files** in `tests/golden/<network>.json`. Each case stores the constructor arguments, the (unconstrained) parameters, some inputs, and the expected outputs. The JAX and PyTorch tests both load the same file, so if both pass, the two implementations agree with each other. Networks that only exist in JAX (e.g. `LBDN`, the contracting/Lipschitz/general RENs, the linear RENs and the R2DN) are checked against their golden files too, which guards against regressions when refactoring.

Golden files are generated from the JAX implementation with `uv run python tests/generate_golden.py [network ...]`. Only regenerate them deliberately (a new case, or an intended change to a network) and review the diff.

The same suite runs on every push to `main` and every pull request through GitHub Actions (`.github/workflows/tests.yml`, CPU only).

### A Note on Dependencies

Dependencies are declared in `pyproject.toml` and are split into two sets:

- **Core dependencies:** required by the `robustnn` package itself.
- **Extra dependencies for `examples`:** all additional dependencies needed to run the demos in `examples/` and the scripts in `test/`. 

By default, the `install.sh` script installs both the core and extra dependencies, and checks for CUDA to install the correct versions of `jax` and `torch`. If you would rather just install the `robustnn` package itself, do not run the `install.sh` script, and instead run one of the following.

```bash
uv sync --extra cpu     # CPU only
uv sync --extra cuda13  # GPU with CUDA 13

## Running an Example

Once you have installed the package as above, use `uv run` to run any of the scripts in the `examples/` folder. For example, from the root directory of the project, run:

```bash
uv run python examples/sandwich_mnist.py
```

## Contact

Please raise an [issue](https://github.com/acfr/RobustNeuralNetworks/issues) if you have any questions, and we'll do our best to get back to you as soon as we can!
