# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `DynUnitary` (JAX and PyTorch)."""

import numpy as np
import pytest

import golden
from backends import ATOL, check_contract, check_golden, make_inputs

NETWORK = "dyn_unitary"


@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("io,nx", [(1, 1), (3, 4), (2, 6)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_dyn_unitary(backend, batches, io, nx, use_bias):
    config = {"input_size": io, "state_size": nx, "use_bias": use_bias}
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    out = check_contract(net, inputs, {"next_state": (batches, nx), "y": (batches, io)})

    # The stacked state-space matrix is orthogonal.
    G = net.orthogonal_matrix()
    np.testing.assert_allclose(G @ G.T, np.eye(nx + io), atol=ATOL)

    # The non-causal inverse recovers the previous state and the input.
    rec = net.invert(inputs, out)
    np.testing.assert_allclose(rec["state"], inputs["state"], atol=1e-4)
    np.testing.assert_allclose(rec["u"], inputs["u"], atol=1e-4)


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_dyn_unitary_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
