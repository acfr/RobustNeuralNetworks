# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `MonLipNet` (JAX and PyTorch)."""

import numpy as np
import pytest

import golden
from backends import check_contract, check_golden, make_inputs

NETWORK = "monlipnet"


@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("units", [[3], [4, 4], [3, 3, 3]])
@pytest.mark.parametrize("fixed", ["mu_nu", "mu_tau", "nu_tau"])
def test_monlipnet_contract_and_bounds(backend, batches, units, fixed):
    size = 3
    config = {"input_size": size, "units": units, "mu": 0.5, "nu": 4.0, "tau": 4.0,
              "is_mu_fixed": "mu" in fixed, "is_nu_fixed": "nu" in fixed,
              "is_tau_fixed": "tau" in fixed}
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"y": (batches, size)})

    lo, hi = net.bounds()
    gains = net.gains(200)
    assert gains.min() >= lo - 1e-3
    assert gains.max() <= hi + 1e-3


@pytest.mark.parametrize("units", [[3], [4, 4]])
def test_monlipnet_inverse_round_trip(backend, units):
    config = {"input_size": 3, "units": units, "mu": 0.5, "nu": 4.0, "tau": 4.0,
              "is_tau_fixed": True}
    inputs = make_inputs(NETWORK, config, 6)
    net = backend.build(NETWORK, config, inputs)
    rec = net.invert(inputs, net.forward(inputs))
    np.testing.assert_allclose(rec["x"], inputs["x"], atol=1e-3)


def test_monlipnet_fixing_all_raises(jax_backend):
    config = {"input_size": 2, "units": [2], "is_mu_fixed": True,
              "is_nu_fixed": True, "is_tau_fixed": True}
    with pytest.raises(ValueError):
        jax_backend.build(NETWORK, config, make_inputs(NETWORK, config, 1))


@pytest.mark.parametrize("kw", [dict(mu=1.0), dict(), dict(nu=2.0)])
def test_monlipnet_needs_two_bounds(torch_backend, kw):
    # JAX raises when all of mu/nu/tau are fixed (above); PyTorch instead
    # requires two of them to be specified at construction.
    config = {"input_size": 2, "units": [2], **kw}
    with pytest.raises(ValueError):
        torch_backend.build(NETWORK, config, make_inputs(NETWORK, config, 1))


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_monlipnet_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
