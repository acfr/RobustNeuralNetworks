# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the PyTorch `PLNet`."""

import numpy as np
import pytest
import torch

import golden
from robustnn.bilipnet_torch import BiLipNet
from robustnn.plnet_torch import PLNet
from torch_utils import check_golden, grads_finite


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
    grads_finite(net)


def test_plnet_zero_at_optimal_point():
    net = _plnet()
    x = torch.randn(4, 3)
    e = net.direct_to_explicit(x_optimal=x[:1].numpy())
    y = net.explicit_call(x[:1].numpy(), e)
    np.testing.assert_allclose(y, 0.0, atol=1e-6)
    assert bool((net.explicit_call(x.numpy(), e) >= 0).all())


@pytest.mark.parametrize("case", golden.cases("plnet"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_plnet_matches_golden(case, explicit):
    check_golden("plnet", case, explicit)
