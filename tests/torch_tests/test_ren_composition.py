# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the PyTorch `CompositionREN`."""

import pytest
import torch

import golden
from robustnn.ren_composition_torch import CompositionREN
from torch_utils import check_golden, grads_finite, rel_err


# ------------------------------ CompositionREN ----------------------------
@pytest.mark.parametrize("batches", [1, 6])
@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("use_bias", [True, False])
def test_composition_ren(batches, L, use_bias):
    io, nx, nv, mu, nu = 2, 3, 6, 0.5, 5.0
    model = CompositionREN(io, nx, nv, num_layers=L, mu=mu, nu=nu, use_bias=use_bias)

    carry = model.initialize_carry(batches)
    u = torch.randn(batches, io)
    new_carry, y = model(carry, u)
    assert len(new_carry["rens"]) == L
    assert tuple(y.shape) == (batches, io)
    assert new_carry["dyn_in"] is None and new_carry["dyn_out"] is None

    with torch.no_grad():
        _, u_rec = model.inverse(carry, y)
    assert rel_err(u_rec, u) < 1e-3

    n = 300
    zc = {"rens": [torch.zeros(n, nx) for _ in range(L)],
          "dyn_in": None, "dyn_out": None}
    ua, ub = torch.randn(n, io), torch.randn(n, io)
    with torch.no_grad():
        _, ya = model(zc, ua)
        _, yb = model(zc, ub)
    gain = (ya - yb).norm(dim=1) / (ua - ub).norm(dim=1)
    assert gain.min().item() >= mu - 1e-3
    assert gain.max().item() <= nu + 1e-3
    assert model.get_bounds() == (mu, nu)

    (y ** 2).sum().backward()
    grads_finite(model)


@pytest.mark.parametrize("at_input,at_output",
                         [(True, False), (False, True), (True, True)])
def test_composition_ren_dyn_orth_noncausal_inverse(at_input, at_output):
    io, nx, L = 2, 3, 2
    model = CompositionREN(io, nx, 6, num_layers=L, mu=0.5, nu=5.0,
                           dyn_orth_at_input=at_input, dyn_orth_at_output=at_output,
                           dyn_state_multiplier=4)
    carry = model.initialize_carry(5)
    assert (carry["dyn_in"] is not None) == at_input
    assert (carry["dyn_out"] is not None) == at_output

    u = torch.randn(5, io)
    new_carry, y = model(carry, u)
    assert tuple(y.shape) == (5, io)
    if at_input:
        assert tuple(new_carry["dyn_in"].shape) == (5, 4 * nx)

    with torch.no_grad():
        _, u_rec = model.inverse_noncausal(carry, new_carry, y)
    assert rel_err(u_rec, u) < 1e-3


def test_composition_ren_ignore_inverse_with_dyn_in():
    model = CompositionREN(2, 3, 6, num_layers=2, mu=0.5, nu=5.0,
                           dyn_orth_at_input=True, dyn_state_multiplier=4)
    carry = model.initialize_carry(5)
    u = torch.randn(5, 2)
    _, y = model(carry, u)
    with torch.no_grad():
        _, sig_rec = model.inverse(carry, y)
        _, dyn_out = model.dyn_in(carry["dyn_in"], u)
    assert rel_err(sig_rec, dyn_out) < 1e-3


@pytest.mark.parametrize("kw", [dict(num_layers=0), dict(mu=2.0, nu=1.0),
                                dict(mu=1.0, nu=1.0)])
def test_composition_ren_invalid_args(kw):
    args = dict(num_layers=2, mu=0.5, nu=5.0)
    args.update(kw)
    with pytest.raises(ValueError):
        CompositionREN(2, 3, 6, **args)


@pytest.mark.parametrize("case", golden.cases("ren_composition"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_ren_composition_matches_golden(case, explicit):
    check_golden("ren_composition", case, explicit)
