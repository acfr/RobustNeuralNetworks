# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `BiLipNet` (JAX and PyTorch)."""

import numpy as np
import pytest

import golden
from backends import ATOL, assert_close, check_contract, check_golden, make_inputs

NETWORK = "bilipnet"


def _config(depth=2, units=(4, 4), size=3, mu=0.5, nu=4.0, **kw):
    return {"input_size": size, "units": list(units), "mu": mu, "nu": nu,
            "tau": nu / mu, "is_mu_fixed": True, "is_nu_fixed": True,
            "depth": depth, **kw}


@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_bilipnet_contract_and_bounds(backend, batches, depth):
    size, mu, nu = 3, 0.5, 4.0
    config = _config(depth=depth, size=size, mu=mu, nu=nu)
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"y": (batches, size)})

    assert_close(net.bounds(), [mu, nu])
    gains = net.gains(200)
    assert gains.min() >= mu - 1e-3
    assert gains.max() <= nu + 1e-3


@pytest.mark.parametrize("depth", [1, 2])
def test_bilipnet_inverse_round_trip(backend, depth):
    config = _config(depth=depth, units=(4,), nu=4.0, act_fn="relu")
    inputs = make_inputs(NETWORK, config, 6)
    net = backend.build(NETWORK, config, inputs)
    rec = net.invert(inputs, net.forward(inputs))
    np.testing.assert_allclose(rec["x"], inputs["x"], atol=1e-3)


def test_bilipnet_no_bias(jax_backend):
    # TODO(torch): BiLipNet has no `use_bias` option in PyTorch.
    config = _config(use_bias=False)
    inputs = make_inputs(NETWORK, config, 5)
    net = jax_backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"y": (5, 3)})


def test_bilipnet_fixing_all_raises(jax_backend):
    config = {"input_size": 2, "units": [2], "is_mu_fixed": True,
              "is_nu_fixed": True, "is_tau_fixed": True}
    with pytest.raises(ValueError):
        jax_backend.build(NETWORK, config, make_inputs(NETWORK, config, 1))


def test_bilipnet_needs_two_bounds(torch_backend):
    # JAX raises when all of mu/nu/tau are fixed (above); PyTorch instead
    # requires two of them to be specified at construction.
    config = {"input_size": 2, "units": [2], "mu": 1.0}
    with pytest.raises(ValueError):
        torch_backend.build(NETWORK, config, make_inputs(NETWORK, config, 1))


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_bilipnet_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
