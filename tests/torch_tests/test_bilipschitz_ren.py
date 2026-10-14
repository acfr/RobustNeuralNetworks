# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the PyTorch `BiLipschitzREN`."""

import pytest
import torch
import torch.nn as nn

import golden
from robustnn.ren_torch import BiLipschitzREN, RENBase
from torch_utils import check_golden, grads_finite, rel_err


# ------------------------------ BiLipschitzREN ----------------------------
@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("nio", [1, 3])
@pytest.mark.parametrize("mu_nu", [(0.5, 4.0), (1.0, 2.0)])
@pytest.mark.parametrize("activation", [nn.ReLU(), nn.Tanh()])
def test_bilipschitz_ren(batches, nio, mu_nu, activation):
    mu, nu = mu_nu
    nx, nv = 4, 8
    model = BiLipschitzREN(nio, nx, nv, mu=mu, nu=nu, activation=activation)

    state = model.initialize_carry(batches)
    assert tuple(state.shape) == (batches, nx)
    u = torch.randn(batches, nio)
    new_state, y = model(state, u)
    assert tuple(new_state.shape) == (batches, nx)
    assert tuple(y.shape) == (batches, nio)

    with torch.no_grad():
        e = model.direct_to_explicit()
        _, y_exp = model.explicit_call(state, u, e)
    torch.testing.assert_close(y_exp, y, atol=1e-4, rtol=1e-4)

    if isinstance(activation, nn.ReLU):
        with torch.no_grad():
            _, u_rec = model.inverse(state, y)
        assert rel_err(u_rec, u) < 1e-3

    n = 300
    z = torch.zeros(n, nx)
    ua, ub = torch.randn(n, nio), torch.randn(n, nio)
    with torch.no_grad():
        _, ya = model.explicit_call(z, ua, e)
        _, yb = model.explicit_call(z, ub, e)
    gain = (ya - yb).norm(dim=1) / (ua - ub).norm(dim=1)
    assert gain.min().item() >= mu - 1e-3
    assert gain.max().item() <= nu + 1e-3

    (y ** 2).sum().backward()
    grads_finite(model)


@pytest.mark.parametrize("mu,nu", [(2.0, 1.0), (1.0, 1.0)])
def test_bilipschitz_ren_invalid_bounds(mu, nu):
    with pytest.raises(ValueError):
        BiLipschitzREN(2, 3, 4, mu=mu, nu=nu)


def test_ren_base_direct_to_explicit_not_implemented():
    base = RENBase(2, 3, 4, 2)
    with pytest.raises(NotImplementedError):
        base.direct_to_explicit()


@pytest.mark.parametrize("case", golden.cases("bilipschitz_ren"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_bilipschitz_ren_matches_golden(case, explicit):
    check_golden("bilipschitz_ren", case, explicit)
