# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

import pytest

pytest.importorskip("torch")

import torch


def pytest_collection_modifyitems(items):
    for item in items:
        if "torch_tests" in str(item.fspath):
            item.add_marker(pytest.mark.torch)


@pytest.fixture(autouse=True)
def _seed_torch():
    torch.manual_seed(0)
