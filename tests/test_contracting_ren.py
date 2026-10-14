# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `ContractingREN`.

TODO(torch): ContractingREN is not yet implemented in PyTorch, so the shared
tests are skipped for the torch backend and the JAX-only tests are not collected
for it.
"""

import numpy as np
import pytest

import golden
from backends import check_contract, check_golden, make_inputs

NETWORK = "contracting_ren"


def _config(nu=3, nx=3, nv=4, ny=2, **kw):
    return {"input_size": nu, "state_size": nx, "features": nv, "output_size": ny, **kw}


@pytest.mark.parametrize("batches", [1, 4])
@pytest.mark.parametrize("activation", ["relu", "tanh", "identity"])
@pytest.mark.parametrize("init_method", ["random", "long_memory"])
@pytest.mark.parametrize("do_polar_param", [True, False])
@pytest.mark.parametrize("sizes", [(1, 1, 1, 1), (5, 3, 4, 2)])
def test_contracting_ren(backend, batches, activation, init_method, do_polar_param, sizes):
    nu, nx, nv, ny = sizes
    config = _config(nu, nx, nv, ny, activation=activation, init_method=init_method,
                     do_polar_param=do_polar_param)
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"next_state": (batches, nx), "y": (batches, ny)})


def test_contracting_ren_contraction(jax_backend):
    # TODO(torch): needs the P matrix of the PyTorch parameterisation.
    import jax
    import jax.numpy as jnp
    from helpers import compute_p_contractingren

    config = _config(5, 3, 4, 2, activation="tanh")
    inputs = make_inputs(NETWORK, config, 8)
    net = jax_backend.build(NETWORK, config, inputs)
    model, params = net.model, net.params
    x1, u = jnp.asarray(inputs["state"]), jnp.asarray(inputs["u"])
    x0 = 10 * jax.random.normal(jax.random.key(0), x1.shape)
    xn0, _ = model.apply(params, x0, u)
    xn1, _ = model.apply(params, x1, u)
    P = compute_p_contractingren(model, params)

    def norm2(x):
        return jnp.sum((x @ P.T) * x, axis=-1)
    assert bool(jnp.all(norm2(xn0 - xn1) - norm2(x0 - x1) <= 1e-4))


def test_ren_identity_output(backend):
    config = _config(3, 3, 4, 3, identity_output=True)
    inputs = make_inputs(NETWORK, config, 4)
    net = backend.build(NETWORK, config, inputs)
    # identity output: y_t = x_t, i.e. the state before the update
    np.testing.assert_allclose(net.forward(inputs)["y"], inputs["state"], atol=1e-5)


def test_ren_init_output_zero(backend):
    config = _config(3, 3, 4, 2, init_output_zero=True)
    inputs = make_inputs(NETWORK, config, 4)
    net = backend.build(NETWORK, config, inputs)
    np.testing.assert_allclose(net.forward(inputs)["y"], 0.0, atol=1e-6)


def test_ren_identity_output_size_mismatch(backend):
    config = _config(3, 3, 4, 2, identity_output=True)
    with pytest.raises(ValueError):
        backend.build(NETWORK, config, make_inputs(NETWORK, config, 2))


def test_ren_bad_init_method(backend):
    config = _config(init_method="nope")
    with pytest.raises(ValueError):
        backend.build(NETWORK, config, make_inputs(NETWORK, config, 2))


@pytest.mark.parametrize("horizon", [1, 5])
def test_simulate_sequence_matches_loop(jax_backend, horizon):
    # TODO(torch): `simulate_sequence` has no PyTorch counterpart yet.
    import jax
    import jax.numpy as jnp

    nu, nx, ny = 3, 4, 2
    config = _config(nu, nx, 5, ny, activation="tanh")
    inputs = make_inputs(NETWORK, config, 4)
    net = jax_backend.build(NETWORK, config, inputs)
    model, params = net.model, net.params
    x0 = jnp.asarray(inputs["state"])
    seq = jax.random.normal(jax.random.key(0), (horizon, 4, nu))

    xT, ys = model.simulate_sequence(params, x0, seq)
    assert ys.shape == (horizon, 4, ny)

    x, outs = x0, []
    for t in range(horizon):
        x, y = model.apply(params, x, seq[t])
        outs.append(y)
    np.testing.assert_allclose(xT, x, atol=1e-4)
    np.testing.assert_allclose(ys, jnp.stack(outs), atol=1e-4)


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_contracting_ren_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
