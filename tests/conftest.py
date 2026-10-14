# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Shared pytest fixtures.

Most tests take the `backend` fixture and run once per backend (JAX and PyTorch).
Tests that only make sense for one backend take `jax_backend` / `torch_backend`
instead. Both kinds are marked `jax` / `torch` so `-m jax` and `-m torch` work.
"""

import importlib.util

import numpy as np
import pytest

from backends import get_backend


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(0)


@pytest.fixture(
    params=[pytest.param("jax", marks=pytest.mark.jax),
            pytest.param("torch", marks=pytest.mark.torch)]
)
def backend(request):
    return get_backend(request.param)


@pytest.fixture
def jax_backend():
    return get_backend("jax")


@pytest.fixture
def torch_backend():
    return get_backend("torch")


def pytest_collection_modifyitems(items):
    for item in items:
        if "jax_backend" in item.fixturenames:
            item.add_marker(pytest.mark.jax)
        if "torch_backend" in item.fixturenames:
            item.add_marker(pytest.mark.torch)
