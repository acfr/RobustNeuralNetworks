# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `BiLipschitzREN` (JAX and PyTorch)."""

import numpy as np
import pytest

import golden
from backends import check_contract, check_golden, make_inputs

NETWORK = "bilipschitz_ren"


def _config(nio=3, mu=0.5, nu=4.0, activation="relu"):
    return {"input_size": nio, "state_size": 4, "features": 8, "output_size": nio,
            "mu": mu, "nu": nu, "activation": activation, "init_method": "long_memory"}


@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("nio", [1, 3])
@pytest.mark.parametrize("mu_nu", [(0.5, 4.0), (1.0, 2.0)])
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_bilipschitz_ren(backend, batches, nio, mu_nu, activation):
    mu, nu = mu_nu
    config = _config(nio, mu, nu, activation)
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    out = check_contract(net, inputs, {"next_state": (batches, 4), "y": (batches, nio)})

    if activation == "relu":  # the inverse solver is only checked for relu
        rec = net.invert(inputs, out)
        np.testing.assert_allclose(rec["u"], inputs["u"], atol=1e-3, rtol=1e-3)

    gains = net.gains(300)
    assert gains.min() >= mu - 1e-3
    assert gains.max() <= nu + 1e-3


@pytest.mark.parametrize("mu,nu", [(2.0, 1.0), (1.0, 1.0)])
def test_bilipschitz_ren_invalid_bounds(torch_backend, mu, nu):
    # TODO(jax): the JAX BiLipschitzREN accepts mu >= nu (neither the constructor,
    # `init` nor `check_valid_qsr` raise), unlike PyTorch and the JAX CompositionREN.
    # Share this test once JAX validates its bounds.
    config = _config(mu=mu, nu=nu)
    with pytest.raises(ValueError):
        torch_backend.build(NETWORK, config, make_inputs(NETWORK, config, 2))


def test_bilipschitz_ren_valid_qsr(jax_backend):
    config = _config()
    net = jax_backend.build(NETWORK, config, make_inputs(NETWORK, config, 2))
    net.model.check_valid_qsr()


def test_bilipschitz_ren_requires_square(jax_backend):
    config = _config() | {"output_size": 2}
    with pytest.raises(ValueError):
        jax_backend.build(NETWORK, config, make_inputs(NETWORK, config, 2))


def test_ren_base_direct_to_explicit_not_implemented(torch_backend):
    from robustnn.ren_torch import RENBase

    with pytest.raises(NotImplementedError):
        RENBase(2, 3, 4, 2).direct_to_explicit()


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_bilipschitz_ren_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
