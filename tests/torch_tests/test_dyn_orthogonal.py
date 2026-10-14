# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the PyTorch `DynUnitary`."""

import pytest
import torch

import golden
from robustnn.dyn_orthogonal_torch import DynUnitary
from torch_utils import check_golden, grads_finite


# ------------------------------- DynUnitary -------------------------------
@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("io,nx", [(1, 1), (3, 4), (2, 6)])
@pytest.mark.parametrize("bias", [True, False])
def test_dyn_unitary(batches, io, nx, bias):
    layer = DynUnitary(io, nx, bias=bias)
    x, u = torch.randn(batches, nx), torch.randn(batches, io)
    assert tuple(layer.initialize_carry(batches).shape) == (batches, nx)

    x1, y = layer(x, u)
    assert tuple(x1.shape) == (batches, nx) and tuple(y.shape) == (batches, io)

    A, B, C, D = layer._blocks()
    G = torch.cat([torch.cat([A, B], 1), torch.cat([C, D], 1)], 0)
    torch.testing.assert_close(G @ G.T, torch.eye(nx + io), atol=1e-5, rtol=0)

    x_rec, u_rec = layer.inverse(x1, y)
    torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(u_rec, u, atol=1e-4, rtol=1e-4)

    (x1 ** 2).sum().add((y ** 2).sum()).backward()
    grads_finite(layer)


@pytest.mark.parametrize("case", golden.cases("dyn_unitary"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_dyn_unitary_matches_golden(case, explicit):
    check_golden("dyn_unitary", case, explicit)
