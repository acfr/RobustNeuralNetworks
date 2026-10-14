# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `BiLipschitzREN`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robustnn import ren_jax as ren
import golden
from jax_utils import check_golden, check_ren_forward, setup_ren


# ----------------------------- BiLipschitz REN ----------------------------
@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("nio", [1, 3])
@pytest.mark.parametrize("mu_nu", [(0.5, 4.0), (1.0, 2.0)])
def test_bilipschitz_ren(key, batches, nio, mu_nu):
    mu, nu = mu_nu
    nx, nv = 4, 8
    model = ren.BiLipschitzREN(nio, nx, nv, nio, mu=mu, nu=nu,
                               init_method="long_memory")
    model.check_valid_qsr()
    params, states, inputs = setup_ren(model, key, batches, nio)
    check_ren_forward(model, params, states, inputs, nx, nio)

    e = model.direct_to_explicit(params)
    _, y = model.explicit_call(params, states, inputs, e)
    e_inv = model.direct_to_explicit_inverse(params)
    _, u_rec = model.inverse_call(params, states, y, e_inv)
    np.testing.assert_allclose(u_rec, inputs, atol=1e-3, rtol=1e-3)

    n = 300
    ka, kb = jax.random.split(key)
    ua = jax.random.normal(ka, (n, nio))
    ub = jax.random.normal(kb, (n, nio))
    z = jnp.zeros((n, nx))
    _, ya = model.explicit_call(params, z, ua, e)
    _, yb = model.explicit_call(params, z, ub, e)
    gain = jnp.linalg.norm(ya - yb, axis=1) / jnp.linalg.norm(ua - ub, axis=1)
    assert gain.min() >= mu - 1e-3
    assert gain.max() <= nu + 1e-3


def test_bilipschitz_ren_requires_square(key):
    model = ren.BiLipschitzREN(3, 3, 4, 2)
    with pytest.raises(ValueError):
        setup_ren(model, key, 2, 3)


@pytest.mark.parametrize("case", golden.cases("bilipschitz_ren"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_bilipschitz_ren_matches_golden(case, explicit):
    check_golden("bilipschitz_ren", case, explicit)
