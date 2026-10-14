# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `LipschitzREN`."""

import jax
import jax.numpy as jnp
import pytest

from robustnn import ren_jax as ren
import golden
from jax_utils import ACTIVATIONS, check_golden, check_ren_forward, setup_ren


# ----------------------------- Lipschitz REN -----------------------------
@pytest.mark.parametrize("gamma", [0.5, 1.0, 5.0])
@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_lipschitz_ren(key, gamma, activation):
    nu, nx, nv, ny = 4, 3, 5, 2
    model = ren.LipschitzREN(nu, nx, nv, ny, gamma=gamma, activation=activation)
    params, states, inputs = setup_ren(model, key, 4, nu)
    check_ren_forward(model, params, states, inputs, nx, ny)

    # Same initial state => one-step output gain is bounded by gamma.
    n = 200
    ka, kb = jax.random.split(key)
    ua = jax.random.normal(ka, (n, nu))
    ub = jax.random.normal(kb, (n, nu))
    x = jnp.zeros((n, nx))
    e = model.direct_to_explicit(params)
    _, ya = model.explicit_call(params, x, ua, e)
    _, yb = model.explicit_call(params, x, ub, e)
    gain = jnp.linalg.norm(ya - yb, axis=1) / jnp.linalg.norm(ua - ub, axis=1)
    assert gain.max() <= gamma + 1e-3


def test_lipschitz_ren_identity_output_unsupported(key):
    with pytest.raises(NotImplementedError):
        model = ren.LipschitzREN(3, 3, 4, 3, identity_output=True)
        setup_ren(model, key, 2, 3)


@pytest.mark.parametrize("case", golden.cases("lipschitz_ren"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_lipschitz_ren_matches_golden(case, explicit):
    check_golden("lipschitz_ren", case, explicit)
