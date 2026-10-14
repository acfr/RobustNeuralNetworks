# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the linear RENs (contracting, Lipschitz and general).

TODO(torch): the linear RENs are not yet implemented in PyTorch, so every test
here is skipped for the torch backend.
"""

import pytest

import golden
from backends import build_jax_model, check_contract, check_golden, make_inputs, make_qsr

NU, NX, NY = 5, 3, 2


def _config(kind, **kw):
    cfg = {"input_size": NU, "state_size": NX, "features": 0, "output_size": NY}
    if kind == "lipschitz":
        cfg["gamma"] = 2.0
    if kind == "general":
        Q, S, R = make_qsr(NU, NY)
        cfg |= {"Q": Q, "S": S, "R": R, "init_method": "long_memory"}
    return cfg | kw


@pytest.mark.parametrize("kind", ["contracting", "lipschitz", "general"])
@pytest.mark.parametrize("batches", [1, 4])
def test_linear_ren(backend, kind, batches):
    network = f"{kind}_linear_ren"
    config = _config(kind)
    inputs = make_inputs(network, config, batches)
    net = backend.build(network, config, inputs)
    check_contract(net, inputs, {"next_state": (batches, NX), "y": (batches, NY)})


def test_general_linear_ren_valid_qsr(jax_backend):
    # TODO(torch): `check_valid_qsr` has no PyTorch counterpart yet.
    build_jax_model("general_linear_ren", _config("general")).check_valid_qsr()


def test_linear_ren_lipschitz_bound(backend):
    network, gamma = "lipschitz_linear_ren", 2.0
    config = _config("lipschitz", input_size=4, gamma=gamma)
    net = backend.build(network, config, make_inputs(network, config, 2))
    assert net.gains(200).max() <= gamma + 1e-3


def test_linear_ren_requires_zero_features(backend):
    network = "contracting_linear_ren"
    config = _config("contracting", features=4)
    with pytest.raises(ValueError):
        backend.build(network, config, make_inputs(network, config, 2))


GOLDEN = [(n, c) for n in ("contracting_linear_ren", "lipschitz_linear_ren",
                           "general_linear_ren") for c in golden.cases(n)]


@pytest.mark.parametrize("network,case", GOLDEN,
                         ids=[f"{n}-{golden.case_id(c)}" for n, c in GOLDEN])
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_linear_ren_matches_golden(backend, network, case, explicit):
    check_golden(backend, network, case, explicit)
