# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `ContractingR2DN`."""

import jax
import jax.numpy as jnp
import pytest

from helpers import compute_p_contractingr2dn
from robustnn import r2dn_jax as r2dn
import golden
from jax_utils import check_golden, check_ren_forward, setup_ren


# ----------------------------------- R2DN ---------------------------------
@pytest.mark.parametrize("batches", [1, 4])
@pytest.mark.parametrize("hidden", [(2,), (2, 2), (4, 3, 2)])
@pytest.mark.parametrize("init_method", ["random", "long_memory"])
def test_contracting_r2dn(key, batches, hidden, init_method):
    nu, nx, nv, ny = 5, 3, 4, 2
    model = r2dn.ContractingR2DN(nu, nx, nv, ny, hidden, init_method=init_method)
    params, states, inputs = setup_ren(model, key, batches, nu)
    check_ren_forward(model, params, states, inputs, nx, ny)


def test_r2dn_simulate_and_contraction(key):
    nu, nx, nv, ny, horizon = 5, 3, 4, 2, 3
    model = r2dn.ContractingR2DN(nu, nx, nv, ny, (2, 2), init_method="long_memory")
    params, x1, u = setup_ren(model, key, 4, nu)

    seq = jax.random.normal(key, (horizon, 4, nu))
    xT, ys = jax.jit(model.simulate_sequence)(params, x1, seq)
    assert ys.shape == (horizon, 4, ny) and xT.shape == x1.shape

    x0 = 10 * jax.random.normal(key, x1.shape)
    xn0, _ = model.apply(params, x0, u)
    xn1, _ = model.apply(params, x1, u)
    P = compute_p_contractingr2dn(model, params)

    def norm2(x):
        return jnp.sum((x @ P.T) * x, axis=-1)
    assert bool(jnp.all(norm2(xn0 - xn1) - norm2(x0 - x1) <= 1e-4))


@pytest.mark.parametrize("case", golden.cases("r2dn"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_r2dn_matches_golden(case, explicit):
    check_golden("r2dn", case, explicit)
