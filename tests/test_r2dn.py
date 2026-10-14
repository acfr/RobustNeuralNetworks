# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the contracting R2DN.

TODO(torch): R2DN is not yet implemented in PyTorch, so the shared tests are
skipped for the torch backend and the JAX-only tests are not collected for it.
"""

import pytest

import golden
from backends import check_contract, check_golden, make_inputs

NETWORK = "r2dn"


def _config(hidden, init_method="long_memory"):
    return {"input_size": 5, "state_size": 3, "features": 4, "output_size": 2,
            "hidden": hidden, "init_method": init_method}


@pytest.mark.parametrize("batches", [1, 4])
@pytest.mark.parametrize("hidden", [[2], [2, 2], [4, 3, 2]])
@pytest.mark.parametrize("init_method", ["random", "long_memory"])
def test_contracting_r2dn(backend, batches, hidden, init_method):
    config = _config(hidden, init_method)
    inputs = make_inputs(NETWORK, config, batches)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"next_state": (batches, 3), "y": (batches, 2)})


def test_r2dn_simulate_and_contraction(jax_backend):
    import jax
    import jax.numpy as jnp
    from helpers import compute_p_contractingr2dn

    nu, nx, ny, horizon = 5, 3, 2, 3
    config = _config([2, 2])
    inputs = make_inputs(NETWORK, config, 4)
    net = jax_backend.build(NETWORK, config, inputs)
    model, params = net.model, net.params
    x1, u = jnp.asarray(inputs["state"]), jnp.asarray(inputs["u"])

    seq = jax.random.normal(jax.random.key(0), (horizon, 4, nu))
    xT, ys = jax.jit(model.simulate_sequence)(params, x1, seq)
    assert ys.shape == (horizon, 4, ny) and xT.shape == x1.shape

    x0 = 10 * jax.random.normal(jax.random.key(1), x1.shape)
    xn0, _ = model.apply(params, x0, u)
    xn1, _ = model.apply(params, x1, u)
    P = compute_p_contractingr2dn(model, params)

    def norm2(x):
        return jnp.sum((x @ P.T) * x, axis=-1)
    assert bool(jnp.all(norm2(xn0 - xn1) - norm2(x0 - x1) <= 1e-4))


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_r2dn_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
