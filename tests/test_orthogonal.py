# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `Unitary` (JAX and PyTorch)."""

import numpy as np
import pytest

import golden
from backends import ATOL, assert_close, check_contract, check_golden, make_inputs

NETWORK = "unitary"


@pytest.mark.parametrize("batches", [1, 7])
@pytest.mark.parametrize("size", [1, 2, 5])
def test_unitary_contract_and_orthogonal(backend, batches, size):
    config = {"input_size": size, "use_bias": True}
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"y": (batches, size)})

    R = net.orthogonal_matrix()
    np.testing.assert_allclose(R @ R.T, np.eye(size), atol=ATOL)


@pytest.mark.parametrize("size", [2, 5])
def test_unitary_no_bias_preserves_norm(backend, size):
    config = {"input_size": size, "use_bias": False}
    inputs = make_inputs(NETWORK, config, 8)
    net = backend.build(NETWORK, config, inputs)
    y = net.forward(inputs)["y"]
    assert_close(np.linalg.norm(y, axis=1), np.linalg.norm(inputs["x"], axis=1))


def test_unitary_no_bias_explicit(backend):
    config = {"input_size": 3, "use_bias": False}
    inputs = make_inputs(NETWORK, config, 4)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"y": (4, 3)})


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
def test_unitary_inverse_round_trip(backend, case):
    # The golden parameters have a non-zero bias, which the inverse has to undo.
    inputs = case["inputs"]
    net = backend.build(NETWORK, case["config"], inputs, case["params"])
    rec = net.invert(inputs, net.forward(inputs))
    np.testing.assert_allclose(rec["x"], inputs["x"], atol=1e-4)


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_unitary_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
