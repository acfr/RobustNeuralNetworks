# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `ContractingREN`."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import compute_p_contractingren
from robustnn import ren_jax as ren
import golden
from jax_utils import ACTIVATIONS, check_golden, check_ren_forward, setup_ren


# --------------------------- Contracting REN -----------------------------
@pytest.mark.parametrize("batches", [1, 4])
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("init_method", ["random", "long_memory"])
@pytest.mark.parametrize("do_polar_param", [True, False])
@pytest.mark.parametrize("sizes", [(1, 1, 1, 1), (5, 3, 4, 2)])
def test_contracting_ren(key, batches, activation, init_method, do_polar_param, sizes):
    nu, nx, nv, ny = sizes
    model = ren.ContractingREN(nu, nx, nv, ny, activation=activation,
                               init_method=init_method, do_polar_param=do_polar_param)
    params, states, inputs = setup_ren(model, key, batches, nu)
    check_ren_forward(model, params, states, inputs, nx, ny)


def test_contracting_ren_contraction(key):
    model = ren.ContractingREN(5, 3, 4, 2, activation=nn.tanh)
    k1, k2 = jax.random.split(key)
    params, x1, u = setup_ren(model, key, 8, 5)
    x0 = 10 * jax.random.normal(k1, x1.shape)
    xn0, _ = model.apply(params, x0, u)
    xn1, _ = model.apply(params, x1, u)
    P = compute_p_contractingren(model, params)

    def norm2(x):
        return jnp.sum((x @ P.T) * x, axis=-1)
    assert bool(jnp.all(norm2(xn0 - xn1) - norm2(x0 - x1) <= 1e-4))


def test_ren_identity_output_and_zero_output(key):
    model = ren.ContractingREN(3, 3, 4, 3, identity_output=True)
    params, states, inputs = setup_ren(model, key, 4, 3)
    _, y = model.apply(params, states, inputs)
    # identity output: y_t = x_t, i.e. the state before the update
    np.testing.assert_allclose(y, states, atol=1e-5)

    zero = ren.ContractingREN(3, 3, 4, 2, init_output_zero=True)
    params, states, inputs = setup_ren(zero, key, 4, 3)
    _, y = zero.apply(params, states, inputs)
    np.testing.assert_allclose(y, 0.0, atol=1e-6)


def test_ren_identity_output_size_mismatch(key):
    model = ren.ContractingREN(3, 3, 4, 2, identity_output=True)
    with pytest.raises(ValueError):
        setup_ren(model, key, 2, 3)


def test_ren_bad_init_method(key):
    model = ren.ContractingREN(3, 3, 4, 2, init_method="nope")
    with pytest.raises(ValueError):
        setup_ren(model, key, 2, 3)

# ------------------------------ simulate_sequence -------------------------
@pytest.mark.parametrize("horizon", [1, 5])
def test_simulate_sequence_matches_loop(key, horizon):
    nu, nx, nv, ny = 3, 4, 5, 2
    model = ren.ContractingREN(nu, nx, nv, ny, activation=nn.tanh)
    params, x0, u0 = setup_ren(model, key, 4, nu)
    seq = jax.random.normal(key, (horizon, 4, nu))

    xT, ys = model.simulate_sequence(params, x0, seq)
    assert ys.shape == (horizon, 4, ny)

    x, outs = x0, []
    for t in range(horizon):
        x, y = model.apply(params, x, seq[t])
        outs.append(y)
    np.testing.assert_allclose(xT, x, atol=1e-4)
    np.testing.assert_allclose(ys, jnp.stack(outs), atol=1e-4)


@pytest.mark.parametrize("case", golden.cases("contracting_ren"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_contracting_ren_matches_golden(case, explicit):
    check_golden("contracting_ren", case, explicit)
