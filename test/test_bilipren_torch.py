# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Tests for the PyTorch bi-Lipschitz REN (`BiLipschitzREN`) and its composition
(`CompositionREN`): forward evaluation, inverse round-trip, empirical
bi-Lipschitz bounds, and gradient flow.
'''

import torch

from robustnn.ren_torch import BiLipschitzREN
from robustnn.ren_composition_torch import CompositionREN


def test_bilipschitz_ren_torch():
    torch.manual_seed(0)
    io, nx, nv = 3, 4, 8
    mu, nu = 0.5, 4.0
    model = BiLipschitzREN(io, nx, nv, mu=mu, nu=nu)

    batches = 5
    state = model.initialize_carry(batches)
    u = torch.randn(batches, io)
    _, y = model(state, u)
    assert tuple(y.shape) == (batches, io)

    with torch.no_grad():
        _, u_rec = model.inverse(state, y)
    rel_err = ((u_rec - u).norm() / u.norm()).item()
    assert rel_err < 1e-4, f"inverse round-trip error too large: {rel_err}"

    z = torch.zeros(300, nx)
    ua, ub = torch.randn(300, io), torch.randn(300, io)
    with torch.no_grad():
        e = model.direct_to_explicit()
        _, ya = model.explicit_call(z, ua, e)
        _, yb = model.explicit_call(z, ub, e)
    gain = (ya - yb).norm(dim=1) / (ua - ub).norm(dim=1)
    assert gain.min().item() >= mu - 1e-3
    assert gain.max().item() <= nu + 1e-3

    (y ** 2).sum().backward()
    gnorm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert bool(torch.isfinite(gnorm))


def test_composition_ren_torch():
    torch.manual_seed(1)
    io, nx, nv, L = 2, 4, 8, 3
    mu, nu = 0.5, 5.0
    model = CompositionREN(io, nx, nv, num_layers=L, mu=mu, nu=nu)

    batches = 6
    states = model.initialize_carry(batches)
    u = torch.randn(batches, io)
    new_states, y = model(states, u)
    assert len(new_states["rens"]) == L
    assert tuple(y.shape) == (batches, io)

    with torch.no_grad():
        _, u_rec = model.inverse(states, y)
    rel_err = ((u_rec - u).norm() / u.norm()).item()
    assert rel_err < 1e-4, f"composition inverse error too large: {rel_err}"

    zc = {"rens": [torch.zeros(300, nx) for _ in range(L)],
          "dyn_in": None, "dyn_out": None}
    ua, ub = torch.randn(300, io), torch.randn(300, io)
    with torch.no_grad():
        _, ca = model(zc, ua)
        _, cb = model(zc, ub)
    gain = (ca - cb).norm(dim=1) / (ua - ub).norm(dim=1)
    assert gain.min().item() >= mu - 1e-3
    assert gain.max().item() <= nu + 1e-3
    assert model.get_bounds() == (mu, nu)

    (y ** 2).sum().backward()
    gnorm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert bool(torch.isfinite(gnorm))


def test_composition_ren_torch_dyn_orth():
    """Composition with a dynamic-orthogonal input layer: both inverse modes."""
    torch.manual_seed(2)
    io, nx, nv, L = 2, 3, 6, 2
    mu, nu = 0.5, 5.0
    model = CompositionREN(io, nx, nv, num_layers=L, mu=mu, nu=nu,
                           dyn_orth_at_input=True, dyn_state_multiplier=4)

    batches = 5
    carry = model.initialize_carry(batches)
    assert carry["dyn_in"] is not None and carry["dyn_out"] is None
    u = torch.randn(batches, io)
    new_carry, y = model(carry, u)
    assert len(new_carry["rens"]) == L
    assert tuple(new_carry["dyn_in"].shape) == (batches, 4 * nx)
    assert tuple(y.shape) == (batches, io)

    # Non-causal inverse recovers the true input exactly.
    with torch.no_grad():
        _, u_rec = model.inverse_noncausal(carry, new_carry, y)
    rel_err = ((u_rec - u).norm() / u.norm()).item()
    assert rel_err < 1e-4, f"noncausal inverse error too large: {rel_err}"

    # Causal "ignore" inverse recovers the dyn-orth forward output.
    with torch.no_grad():
        _, sig_rec = model.inverse(carry, y)
        _, dyn_out = model.dyn_in(carry["dyn_in"], u)
    rel_err2 = ((sig_rec - dyn_out).norm() / dyn_out.norm()).item()
    assert rel_err2 < 1e-4, f"ignore inverse mismatch: {rel_err2}"


if __name__ == "__main__":
    test_bilipschitz_ren_torch()
    test_composition_ren_torch()
    test_composition_ren_torch_dyn_orth()
    print("All PyTorch bi-Lipschitz REN tests passed.")
