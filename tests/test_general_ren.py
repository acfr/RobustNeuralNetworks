# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for `GeneralREN`.

TODO(torch): GeneralREN is not yet implemented in PyTorch, so the shared tests
are skipped for the torch backend and the JAX-only tests are not collected for it.
"""

import numpy as np
import pytest

import golden
from backends import build_jax_model, check_contract, check_golden, make_inputs, make_qsr

NETWORK = "general_ren"


def _config(nu, nx, nv, ny, **kw):
    Q, S, R = make_qsr(nu, ny)
    return {"input_size": nu, "state_size": nx, "features": nv, "output_size": ny,
            "Q": Q, "S": S, "R": R, **kw}


@pytest.mark.parametrize("activation", ["relu", "tanh", "identity"])
@pytest.mark.parametrize("sizes", [(5, 3, 4, 2), (2, 2, 3, 2)])
def test_general_ren(backend, activation, sizes):
    nu, nx, nv, ny = sizes
    config = _config(nu, nx, nv, ny, activation=activation, init_method="long_memory")
    inputs = make_inputs(NETWORK, config, 4)
    net = backend.build(NETWORK, config, inputs)
    check_contract(net, inputs, {"next_state": (4, nx), "y": (4, ny)})


def test_general_ren_valid_qsr(jax_backend):
    # TODO(torch): `check_valid_qsr` has no PyTorch counterpart yet.
    config = _config(5, 3, 4, 2)
    build_jax_model(NETWORK, config).check_valid_qsr()


def test_general_ren_invalid_qsr(jax_backend):
    nu, nx, nv, ny = 3, 3, 4, 2
    Q, S, R = make_qsr(nu, ny)
    bad = [
        dict(Q=np.eye(ny + 1), S=S, R=R),                  # wrong Q size
        dict(Q=Q, S=np.ones((ny, nu)), R=R),               # wrong S size
        dict(Q=Q, S=S, R=np.eye(nu + 1)),                  # wrong R size
        dict(Q=np.eye(ny), S=S, R=R),                      # Q not neg. definite
        dict(Q=Q, S=S, R=-np.eye(nu)),                     # Schur complement not PD
    ]
    for kw in bad:
        config = {"input_size": nu, "state_size": nx, "features": nv,
                  "output_size": ny, **kw}
        with pytest.raises(ValueError):
            build_jax_model(NETWORK, config).check_valid_qsr()


@pytest.mark.parametrize("case", golden.cases(NETWORK), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_general_ren_matches_golden(backend, case, explicit):
    check_golden(backend, NETWORK, case, explicit)
