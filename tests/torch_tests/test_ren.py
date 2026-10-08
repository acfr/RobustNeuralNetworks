# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the PyTorch `BiLipschitzREN`, `DynUnitary` and `CompositionREN`."""

import pytest
import torch
import torch.nn as nn

from robustnn.dyn_orthogonal_torch import DynUnitary
from robustnn.ren_composition_torch import CompositionREN
from robustnn.ren_torch import BiLipschitzREN, RENBase


def _grads_finite(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(bool(torch.isfinite(g).all()) for g in grads)


def _rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


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
        assert _rel_err(u_rec, u) < 1e-3

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
    _grads_finite(model)


@pytest.mark.parametrize("mu,nu", [(2.0, 1.0), (1.0, 1.0)])
def test_bilipschitz_ren_invalid_bounds(mu, nu):
    with pytest.raises(ValueError):
        BiLipschitzREN(2, 3, 4, mu=mu, nu=nu)


def test_ren_base_direct_to_explicit_not_implemented():
    base = RENBase(2, 3, 4, 2)
    with pytest.raises(NotImplementedError):
        base.direct_to_explicit()


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
    _grads_finite(layer)


# ------------------------------ CompositionREN ----------------------------
@pytest.mark.parametrize("batches", [1, 6])
@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("use_bias", [True, False])
def test_composition_ren(batches, L, use_bias):
    io, nx, nv, mu, nu = 2, 3, 6, 0.5, 5.0
    model = CompositionREN(io, nx, nv, num_layers=L, mu=mu, nu=nu, use_bias=use_bias)

    carry = model.initialize_carry(batches)
    u = torch.randn(batches, io)
    new_carry, y = model(carry, u)
    assert len(new_carry["rens"]) == L
    assert tuple(y.shape) == (batches, io)
    assert new_carry["dyn_in"] is None and new_carry["dyn_out"] is None

    with torch.no_grad():
        _, u_rec = model.inverse(carry, y)
    assert _rel_err(u_rec, u) < 1e-3

    n = 300
    zc = {"rens": [torch.zeros(n, nx) for _ in range(L)],
          "dyn_in": None, "dyn_out": None}
    ua, ub = torch.randn(n, io), torch.randn(n, io)
    with torch.no_grad():
        _, ya = model(zc, ua)
        _, yb = model(zc, ub)
    gain = (ya - yb).norm(dim=1) / (ua - ub).norm(dim=1)
    assert gain.min().item() >= mu - 1e-3
    assert gain.max().item() <= nu + 1e-3
    assert model.get_bounds() == (mu, nu)

    (y ** 2).sum().backward()
    _grads_finite(model)


@pytest.mark.parametrize("at_input,at_output",
                         [(True, False), (False, True), (True, True)])
def test_composition_ren_dyn_orth_noncausal_inverse(at_input, at_output):
    io, nx, L = 2, 3, 2
    model = CompositionREN(io, nx, 6, num_layers=L, mu=0.5, nu=5.0,
                           dyn_orth_at_input=at_input, dyn_orth_at_output=at_output,
                           dyn_state_multiplier=4)
    carry = model.initialize_carry(5)
    assert (carry["dyn_in"] is not None) == at_input
    assert (carry["dyn_out"] is not None) == at_output

    u = torch.randn(5, io)
    new_carry, y = model(carry, u)
    assert tuple(y.shape) == (5, io)
    if at_input:
        assert tuple(new_carry["dyn_in"].shape) == (5, 4 * nx)

    with torch.no_grad():
        _, u_rec = model.inverse_noncausal(carry, new_carry, y)
    assert _rel_err(u_rec, u) < 1e-3


def test_composition_ren_ignore_inverse_with_dyn_in():
    model = CompositionREN(2, 3, 6, num_layers=2, mu=0.5, nu=5.0,
                           dyn_orth_at_input=True, dyn_state_multiplier=4)
    carry = model.initialize_carry(5)
    u = torch.randn(5, 2)
    _, y = model(carry, u)
    with torch.no_grad():
        _, sig_rec = model.inverse(carry, y)
        _, dyn_out = model.dyn_in(carry["dyn_in"], u)
    assert _rel_err(sig_rec, dyn_out) < 1e-3


@pytest.mark.parametrize("kw", [dict(num_layers=0), dict(mu=2.0, nu=1.0),
                                dict(mu=1.0, nu=1.0)])
def test_composition_ren_invalid_args(kw):
    args = dict(num_layers=2, mu=0.5, nu=5.0)
    args.update(kw)
    with pytest.raises(ValueError):
        CompositionREN(2, 3, 6, **args)
