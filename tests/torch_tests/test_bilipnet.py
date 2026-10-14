# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the PyTorch `BiLipNet`."""

import numpy as np
import pytest
import torch

import golden
from robustnn.bilipnet_torch import BiLipNet
from torch_utils import check_golden, empirical_gains, grads_finite


relu_np = lambda x: np.maximum(0, x)


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
    gains = empirical_gains(net, 200, size)
    assert gains.min() >= mu - 1e-3
    assert gains.max() <= nu + 1e-3

    (y ** 2).sum().backward()
    grads_finite(net)


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


@pytest.mark.parametrize("case", golden.cases("bilipnet"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_bilipnet_matches_golden(case, explicit):
    check_golden("bilipnet", case, explicit)
