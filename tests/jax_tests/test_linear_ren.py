# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX linear RENs (`ContractingLinREN`, `LipschitzLinREN`, `GeneralLinREN`)."""

import jax
import jax.numpy as jnp
import pytest

from robustnn import linear_ren
import golden
from jax_utils import check_golden, check_ren_forward, make_qsr, setup_ren


# --------------------------------- Linear RENs ----------------------------
@pytest.mark.parametrize("cls", ["contracting", "lipschitz", "general"])
@pytest.mark.parametrize("batches", [1, 4])
def test_linear_ren(key, cls, batches):
    nu, nx, ny = 5, 3, 2
    if cls == "contracting":
        model = linear_ren.ContractingLinREN(nu, nx, 0, ny)
    elif cls == "lipschitz":
        model = linear_ren.LipschitzLinREN(nu, nx, 0, ny, gamma=2.0)
    else:
        Q, S, R = make_qsr(key, nu, ny)
        model = linear_ren.GeneralLinREN(nu, nx, 0, ny, Q=Q, S=S, R=R,
                                         init_method="long_memory")
        model.check_valid_qsr()
    params, states, inputs = setup_ren(model, key, batches, nu)
    check_ren_forward(model, params, states, inputs, nx, ny)


def test_linear_ren_lipschitz_bound(key):
    nu, nx, ny, gamma = 4, 3, 2, 2.0
    model = linear_ren.LipschitzLinREN(nu, nx, 0, ny, gamma=gamma)
    params, _, _ = setup_ren(model, key, 2, nu)
    n = 200
    ka, kb = jax.random.split(key)
    ua = jax.random.normal(ka, (n, nu))
    ub = jax.random.normal(kb, (n, nu))
    z = jnp.zeros((n, nx))
    e = model.direct_to_explicit(params)
    _, ya = model.explicit_call(params, z, ua, e)
    _, yb = model.explicit_call(params, z, ub, e)
    gain = jnp.linalg.norm(ya - yb, axis=1) / jnp.linalg.norm(ua - ub, axis=1)
    assert gain.max() <= gamma + 1e-3


def test_linear_ren_requires_zero_features(key):
    with pytest.raises(ValueError):
        model = linear_ren.ContractingLinREN(3, 3, 4, 2)
        setup_ren(model, key, 2, 3)


@pytest.mark.parametrize("case", golden.cases("contracting_linear_ren"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_contracting_linear_ren_matches_golden(case, explicit):
    check_golden("contracting_linear_ren", case, explicit)


@pytest.mark.parametrize("case", golden.cases("lipschitz_linear_ren"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_lipschitz_linear_ren_matches_golden(case, explicit):
    check_golden("lipschitz_linear_ren", case, explicit)


@pytest.mark.parametrize("case", golden.cases("general_linear_ren"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_general_linear_ren_matches_golden(case, explicit):
    check_golden("general_linear_ren", case, explicit)
