# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX QSR-constrained `GeneralREN`."""

import jax.numpy as jnp
import pytest

from robustnn import ren_jax as ren
import golden
from jax_utils import ACTIVATIONS, check_golden, check_ren_forward, make_qsr, setup_ren


# ------------------------------ General REN ------------------------------
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("sizes", [(5, 3, 4, 2), (2, 2, 3, 2)])
def test_general_ren(key, activation, sizes):
    nu, nx, nv, ny = sizes
    Q, S, R = make_qsr(key, nu, ny)
    model = ren.GeneralREN(nu, nx, nv, ny, Q=Q, S=S, R=R, activation=activation,
                           init_method="long_memory")
    model.check_valid_qsr()
    params, states, inputs = setup_ren(model, key, 4, nu)
    check_ren_forward(model, params, states, inputs, nx, ny)


def test_general_ren_invalid_qsr(key):
    nu, ny = 3, 2
    Q, S, R = make_qsr(key, nu, ny)
    bad = [
        dict(Q=jnp.eye(ny + 1), S=S, R=R),                 # wrong Q size
        dict(Q=Q, S=jnp.ones((ny, nu)), R=R),              # wrong S size
        dict(Q=Q, S=S, R=jnp.eye(nu + 1)),                 # wrong R size
        dict(Q=jnp.eye(ny), S=S, R=R),                     # Q not neg. definite
        dict(Q=Q, S=S, R=-jnp.eye(nu)),                    # Schur complement not PD
    ]
    for kw in bad:
        model = ren.GeneralREN(nu, 3, 4, ny, **kw)
        with pytest.raises(ValueError):
            model.check_valid_qsr()


@pytest.mark.parametrize("case", golden.cases("general_ren"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_general_ren_matches_golden(case, explicit):
    check_golden("general_ren", case, explicit)
