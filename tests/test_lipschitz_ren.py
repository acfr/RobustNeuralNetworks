# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `LipschitzREN`.

TODO(torch): LipschitzREN is not yet implemented in PyTorch, so every test here
is skipped for the torch backend.
"""

import pytest

import golden
from backends import check_contract, check_golden, make_inputs

NETWORK = "lipschitz_ren"


@pytest.mark.parametrize("gamma", [0.5, 1.0, 5.0])
@pytest.mark.parametrize("activation", ["relu", "tanh", "identity"])
def test_lipschitz_ren(backend, gamma, activation):
    nu, nx, nv, ny = 4, 3, 5, 2
    config = {"input_size": nu, "state_size": nx, "features": nv, "output_size": ny,
              "gamma": gamma, "activation": activation}
    inputs = make_inputs(NETWORK, config, 4)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"next_state": (4, nx), "y": (4, ny)})

    # Same initial state => one-step output gain is bounded by gamma.
    assert net.gains(200).max() <= gamma + 1e-3


def test_lipschitz_ren_identity_output_unsupported(backend):
    config = {"input_size": 3, "state_size": 3, "features": 4, "output_size": 3,
              "identity_output": True}
    with pytest.raises(NotImplementedError):
        backend.build(NETWORK, config, make_inputs(NETWORK, config, 2))


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_lipschitz_ren_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
