# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the PyTorch `Unitary`, `MonLipNet`, `BiLipNet` and `PLNet`."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from robustnn.bilipnet_torch import BiLipNet
from robustnn.monlipnet_torch import MonLipNet
from robustnn.orthogonal_torch import Unitary
from robustnn.plnet_torch import PLNet


def _gains(f, n, size):
    ua, ub = torch.randn(n, size), torch.randn(n, size)
    with torch.no_grad():
        return (f(ua) - f(ub)).norm(dim=1) / (ua - ub).norm(dim=1)


def _grads_finite(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(bool(torch.isfinite(g).all()) for g in grads)


relu_np = lambda x: np.maximum(0, x)


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
    _grads_finite(layer)


@pytest.mark.parametrize("size", [2, 5])
def test_unitary_no_bias_forward_preserves_norm(size):
    layer = Unitary(size, size, bias=False)
    x = torch.randn(8, size)
    np.testing.assert_allclose(layer(x).norm(dim=1).detach(), x.norm(dim=1), rtol=1e-4)


@pytest.mark.xfail(raises=AttributeError, strict=True,
                   reason="Unitary.direct_to_explicit assumes a bias exists")
def test_unitary_no_bias_explicit():
    Unitary(3, 3, bias=False).direct_to_explicit()


# ------------------------------- MonLipNet --------------------------------
@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("units", [[3], [4, 4], [3, 3, 3]])
@pytest.mark.parametrize("fixed", ["mu_nu", "mu_tau", "nu_tau"])
def test_monlipnet(batches, units, fixed):
    size, mu, nu, tau = 3, 0.5, 4.0, 4.0
    net = MonLipNet(size, units, mu=mu, nu=nu, tau=tau,
                    is_mu_fixed="mu" in fixed, is_nu_fixed="nu" in fixed,
                    is_tau_fixed="tau" in fixed)
    x = torch.randn(batches, size)
    y = net(x)
    assert tuple(y.shape) == (batches, size)
    np.testing.assert_allclose(
        net.explicit_call(x.numpy(), net.direct_to_explicit()),
        y.detach().numpy(), atol=1e-4)

    lo, hi, _ = net.get_bounds()
    gains = _gains(net, 200, size)
    assert gains.min() >= float(lo) - 1e-3
    assert gains.max() <= float(hi) + 1e-3

    (y ** 2).sum().backward()
    _grads_finite(net)


@pytest.mark.parametrize("units", [[3], [4, 4]])
def test_monlipnet_inverse_round_trip(units):
    net = MonLipNet(3, units, mu=0.5, nu=4.0, tau=4.0, is_tau_fixed=True)
    x = torch.randn(6, 3)
    with torch.no_grad():
        y = net(x)
    x_rec = net.inverse(y.numpy(), alpha=0.1, iterations=500)
    np.testing.assert_allclose(x_rec, x.numpy(), atol=1e-3)


@pytest.mark.parametrize("kw", [dict(mu=1.0), dict(), dict(nu=2.0)])
def test_monlipnet_needs_two_bounds(kw):
    with pytest.raises(ValueError):
        MonLipNet(2, [2], **kw)


# ------------------------------- BiLipNet ---------------------------------
@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_bilipnet(batches, depth):
    size, mu, nu = 3, 0.5, 4.0
    net = BiLipNet(size, [4, 4], mu=mu, nu=nu, tau=nu / mu,
                   is_mu_fixed=True, is_nu_fixed=True, depth=depth)
    x = torch.randn(batches, size)
    y = net(x)
    assert tuple(y.shape) == (batches, size)
    np.testing.assert_allclose(
        net.explicit_call(x.numpy(), net.direct_to_explicit()),
        y.detach().numpy(), atol=1e-4)

    lo, hi, _ = net.get_bounds()
    np.testing.assert_allclose([lo, hi], [mu, nu], rtol=1e-4)
    gains = _gains(net, 200, size)
    assert gains.min() >= mu - 1e-3
    assert gains.max() <= nu + 1e-3

    (y ** 2).sum().backward()
    _grads_finite(net)


@pytest.mark.parametrize("depth", [1, 2])
def test_bilipnet_inverse_round_trip(depth):
    net = BiLipNet(3, [4], mu=0.5, nu=4.0, tau=8.0, is_mu_fixed=True,
                   is_nu_fixed=True, depth=depth)
    x = torch.randn(6, 3)
    with torch.no_grad():
        y = net(x)
    x_rec = net.inverse(y.numpy(), alphas=[0.1] * depth,
                        inverse_activation_fns=[relu_np] * depth,
                        iterations=[500] * depth, Lambdas=[1.0] * depth)
    np.testing.assert_allclose(x_rec, x.numpy(), atol=1e-3)


# --------------------------------- PLNet ----------------------------------
def _plnet(**kw):
    bilip = BiLipNet(3, [4], mu=0.5, nu=4.0, tau=8.0, is_mu_fixed=True,
                     is_nu_fixed=True, depth=2)
    return PLNet(BiLipBlock=bilip, **kw)


@pytest.mark.parametrize("batches", [1, 6])
def test_plnet(batches):
    net = _plnet()
    x = torch.randn(batches, 3)
    y = net(x)
    assert tuple(y.shape) == (batches,)
    assert bool((y >= 0).all())
    np.testing.assert_allclose(
        net.explicit_call(x.numpy(), net.direct_to_explicit()),
        y.detach().numpy(), atol=1e-4)
    y.sum().backward()
    _grads_finite(net)


def test_plnet_zero_at_optimal_point():
    net = _plnet()
    x = torch.randn(4, 3)
    e = net.direct_to_explicit(x_optimal=x[:1].numpy())
    y = net.explicit_call(x[:1].numpy(), e)
    np.testing.assert_allclose(y, 0.0, atol=1e-6)
    assert bool((net.explicit_call(x.numpy(), e) >= 0).all())
