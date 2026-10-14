# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the PyTorch `Unitary` layer."""

import numpy as np
import pytest
import torch

import golden
from robustnn.orthogonal_torch import Unitary
from torch_utils import check_golden, grads_finite


# -------------------------------- Unitary ---------------------------------
@pytest.mark.parametrize("batches", [1, 7])
@pytest.mark.parametrize("size", [1, 2, 5])
def test_unitary(batches, size):
    layer = Unitary(size, size, bias=True)
    x = torch.randn(batches, size)
    y = layer(x)
    assert tuple(y.shape) == (batches, size)

    e = layer.direct_to_explicit()
    np.testing.assert_allclose(e.Q @ e.Q.T, np.eye(size), atol=1e-5)
    np.testing.assert_allclose(layer.explicit_call(x.numpy(), e),
                               y.detach().numpy(), atol=1e-5)

    with torch.no_grad():
        x_rec = layer.inverse(y)
    np.testing.assert_allclose(np.asarray(x_rec), x.numpy(), atol=1e-4)

    (y ** 2).sum().backward()
    grads_finite(layer)


@pytest.mark.parametrize("size", [2, 5])
def test_unitary_no_bias_forward_preserves_norm(size):
    layer = Unitary(size, size, bias=False)
    x = torch.randn(8, size)
    np.testing.assert_allclose(layer(x).norm(dim=1).detach(), x.norm(dim=1), rtol=1e-4)


def test_unitary_no_bias_explicit():
    Unitary(3, 3, bias=False).direct_to_explicit()


@pytest.mark.parametrize("case", golden.cases("unitary"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_unitary_matches_golden(case, explicit):
    check_golden("unitary", case, explicit)
