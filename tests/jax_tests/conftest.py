# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax


def pytest_collection_modifyitems(items):
    for item in items:
        if "jax_tests" in str(item.fspath):
            item.add_marker(pytest.mark.jax)


@pytest.fixture
def key():
    return jax.random.key(0)
