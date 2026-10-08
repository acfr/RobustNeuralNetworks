# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Shared pytest fixtures. Backend-specific fixtures live in the
`jax_tests/` and `torch_tests/` conftest files."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_numpy():
    np.random.seed(0)
