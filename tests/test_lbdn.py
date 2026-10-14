# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `LBDN`.

TODO(torch): LBDN is not yet implemented in PyTorch, so every test here is
skipped for the torch backend. They will run for it once `backends.py` lists
`lbdn` as supported.
"""

import numpy as np
import pytest

import golden
from backends import check_contract, check_golden, make_inputs

NETWORK = "lbdn"


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "identity"])
@pytest.mark.parametrize("hidden", [[4], [8, 16]])
@pytest.mark.parametrize("batches", [1, 5])
def test_lbdn_contract_and_lipschitz(backend, activation, hidden, batches):
    nin, nout, gamma = 5, 2, 3.0
    config = {"input_size": nin, "hidden_sizes": hidden, "output_size": nout,
              "gamma": gamma, "activation": activation}
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"y": (batches, nout)})
    assert net.gains(200).max() <= gamma + 1e-3


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("trainable_lipschitz", [True, False])
def test_lbdn_options(backend, use_bias, trainable_lipschitz):
    config = {"input_size": 3, "hidden_sizes": [6], "output_size": 2, "gamma": 2.0,
              "use_bias": use_bias, "trainable_lipschitz": trainable_lipschitz}
    inputs = make_inputs(NETWORK, config, 4)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"y": (4, 2)})


def test_lbdn_init_output_zero(backend):
    config = {"input_size": 3, "hidden_sizes": [6], "output_size": 2,
              "init_output_zero": True}
    inputs = make_inputs(NETWORK, config, 4)
    net = backend.build(NETWORK, config, inputs)
    np.testing.assert_allclose(net.forward(inputs)["y"], 0.0, atol=1e-6)


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_lbdn_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
