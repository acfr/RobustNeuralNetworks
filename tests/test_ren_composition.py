# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `CompositionREN` (JAX and PyTorch)."""

import numpy as np
import pytest

import golden
from backends import check_contract, check_golden, make_inputs

NETWORK = "ren_composition"


def _config(io=2, nx=3, L=2, mu=0.5, nu=5.0, **kw):
    return {"input_size": io, "state_size": nx, "features": 6, "num_layers": L,
            "mu": mu, "nu": nu, "init_method": "long_memory",
            "dyn_state_multiplier": 4, **kw}


@pytest.mark.parametrize("batches", [1, 6])
@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_composition_ren(backend, batches, L, use_bias, activation):
    io, nx, mu, nu = 2, 3, 0.5, 5.0
    config = _config(io, nx, L, mu, nu, use_bias=use_bias, activation=activation)
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    out = check_contract(net, inputs, {"y": (batches, io)})

    carry = out["carry"]
    assert carry["rens"].shape == (L, batches, nx)
    assert carry["dyn_in"] is None and carry["dyn_out"] is None

    if activation == "relu":
        # The inverse solver is only accurate to ~1e-3 for smooth activations
        # at the default iteration count, so only check it for relu.
        rec = net.invert(inputs, out)
        np.testing.assert_allclose(rec["u"], inputs["u"], atol=1e-3, rtol=1e-3)

    gains = net.gains(300)
    assert gains.min() >= mu - 1e-3
    assert gains.max() <= nu + 1e-3
    assert net.bounds() == (mu, nu)


@pytest.mark.parametrize("at_input,at_output",
                         [(True, False), (False, True), (True, True)])
def test_composition_ren_dyn_orth_noncausal_inverse(backend, at_input, at_output):
    io, nx = 2, 3
    config = _config(io, nx, dyn_orth_at_input=at_input, dyn_orth_at_output=at_output)
    inputs = make_inputs(NETWORK, config, 5)
    net = backend.build(NETWORK, config, inputs)
    out = net.forward(inputs)
    assert out["y"].shape == (5, io)
    if at_input:
        assert out["carry"]["dyn_in"].shape == (5, 4 * nx)

    rec = net.invert(inputs, out, noncausal=True)
    np.testing.assert_allclose(rec["u"], inputs["u"], atol=1e-3, rtol=1e-3)


def test_composition_ren_ignore_inverse_with_dyn_in(backend):
    config = _config(dyn_orth_at_input=True)
    inputs = make_inputs(NETWORK, config, 5)
    net = backend.build(NETWORK, config, inputs)
    rec = net.invert(inputs, net.forward(inputs))
    # The causal inverse skips the dynamic input layer, so it returns that
    # layer's output rather than the network input.
    np.testing.assert_allclose(rec["u"], net.dyn_in_output(inputs), atol=1e-3, rtol=1e-3)


def test_composition_ren_ignore_inverse_rejects_dyn_out(backend):
    config = _config(dyn_orth_at_output=True)
    inputs = make_inputs(NETWORK, config, 3)
    net = backend.build(NETWORK, config, inputs)
    with pytest.raises(ValueError):
        net.invert(inputs, net.forward(inputs))


@pytest.mark.parametrize("kw", [dict(L=0), dict(mu=2.0, nu=1.0), dict(mu=1.0, nu=1.0)])
def test_composition_ren_invalid_args(backend, kw):
    config = _config(**kw)
    with pytest.raises(ValueError):
        backend.build(NETWORK, config, make_inputs(NETWORK, config, 2))


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_ren_composition_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
