# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `PLNet` (JAX and PyTorch)."""

import numpy as np
import pytest

import golden
from backends import check_contract, check_golden, make_inputs

NETWORK = "plnet"

BILIP = {"input_size": 3, "units": [4], "mu": 0.5, "nu": 4.0, "tau": 8.0,
         "is_mu_fixed": True, "is_nu_fixed": True, "depth": 2}


@pytest.mark.parametrize("batches", [1, 6])
def test_plnet_contract_and_nonnegative(backend, batches):
    config = {"BiLipBlock": BILIP}
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    out = check_contract(net, inputs, {"y": (batches,)})
    assert (out["y"] >= 0).all()


def test_plnet_zero_at_optimal_point(backend):
    x = make_inputs(NETWORK, {"BiLipBlock": BILIP}, 4)["x"]
    config = {"BiLipBlock": BILIP, "optimal_point": x[:1].tolist()}
    net = backend.build(NETWORK, config, {"x": x})
    y_opt = net.explicit({"x": x[:1]})["y"]
    np.testing.assert_allclose(y_opt, 0.0, atol=1e-6)
    assert (net.explicit({"x": x})["y"] >= 0).all()


def test_plnet_add_constant(jax_backend):
    # TODO(torch): the explicit path with `add_constant=True` currently fails in
    # PyTorch (see test_plnet_matches_golden[explicit-add_constant]); share this
    # test once that is fixed.
    config = {"BiLipBlock": BILIP, "add_constant": True, "c": 1.5}
    inputs = make_inputs(NETWORK, config, 4)
    net = jax_backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"y": (4,)})


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_plnet_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
