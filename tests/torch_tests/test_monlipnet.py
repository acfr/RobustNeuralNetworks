# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the PyTorch `MonLipNet`."""

import numpy as np
import pytest
import torch

import golden
from robustnn.monlipnet_torch import MonLipNet
from torch_utils import check_golden, empirical_gains, grads_finite


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
    gains = empirical_gains(net, 200, size)
    assert gains.min() >= float(lo) - 1e-3
    assert gains.max() <= float(hi) + 1e-3

    (y ** 2).sum().backward()
    grads_finite(net)


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


@pytest.mark.parametrize("case", golden.cases("monlipnet"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_monlipnet_matches_golden(case, explicit):
    check_golden("monlipnet", case, explicit)
