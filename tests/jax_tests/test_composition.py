# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `DynUnitary` and `CompositionREN`."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from robustnn.dyn_orthogonal_jax import DynUnitary
from robustnn.ren_composition_jax import CompositionREN


# ------------------------------ DynUnitary --------------------------------
@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("io,nx", [(1, 1), (3, 4), (2, 6)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_dyn_unitary(key, batches, io, nx, use_bias):
    layer = DynUnitary(io, nx, use_bias=use_bias)
    x = jax.random.normal(key, (batches, nx))
    u = jax.random.normal(jax.random.key(1), (batches, io))
    params = layer.init(key, x, u)

    assert layer.initialize_carry(key, (batches, io)).shape == (batches, nx)
    e = layer.direct_to_explicit(params)
    x1, y = layer.explicit_call(params, x, u, e)
    assert x1.shape == (batches, nx) and y.shape == (batches, io)
    x1b, yb = layer.apply(params, x, u)
    np.testing.assert_allclose(x1, x1b, atol=1e-5)
    np.testing.assert_allclose(y, yb, atol=1e-5)

    # Stacked state-space matrix is orthogonal.
    G = jnp.block([[e.A, e.B], [e.C, e.D]])
    np.testing.assert_allclose(G @ G.T, jnp.eye(nx + io), atol=1e-5)

    # Non-causal inverse recovers the previous state and the input.
    x_rec, u_rec = layer.inverse_call(params, x1, y, e)
    np.testing.assert_allclose(x_rec, x, atol=1e-4)
    np.testing.assert_allclose(u_rec, u, atol=1e-4)

    g = jax.grad(lambda p: sum(jnp.sum(a ** 2) for a in layer.apply(p, x, u)))(params)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


# ----------------------------- CompositionREN -----------------------------
def _model(io=2, nx=3, nv=6, L=2, mu=0.5, nu=5.0, **kw):
    return CompositionREN(io, nx, nv, num_layers=L, mu=mu, nu=nu,
                          init_method="long_memory", **kw)


def _setup(model, key, batches, io):
    k1, k2 = jax.random.split(key)
    carry = model.initialize_carry(k1, (batches, io))
    inputs = jax.random.normal(k2, (batches, io))
    params = model.init(key, carry, inputs)
    return params, carry, inputs


def _zero_carry(model, n, nx, L):
    c = model.initialize_carry(jax.random.key(0), (n, model.input_size))
    return jax.tree.map(jnp.zeros_like, c)


@pytest.mark.parametrize("batches", [1, 6])
@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("activation", [nn.relu, nn.tanh])
def test_composition_ren(key, batches, L, use_bias, activation):
    io, nx, mu, nu = 2, 3, 0.5, 5.0
    model = _model(io=io, nx=nx, L=L, mu=mu, nu=nu, use_bias=use_bias,
                   activation=activation)
    params, carry, inputs = _setup(model, key, batches, io)

    new_carry, y = model.apply(params, carry, inputs)
    assert len(new_carry["rens"]) == L
    assert y.shape == (batches, io)
    assert new_carry["dyn_in"] is None and new_carry["dyn_out"] is None

    e = model.direct_to_explicit(params)
    _, y_exp = model.explicit_call(params, carry, inputs, e)
    np.testing.assert_allclose(y, y_exp, atol=1e-5)

    if activation is nn.relu:
        # The inverse solver is only accurate to ~1e-3 for smooth activations
        # at the default iteration count, so only check it for relu.
        e_inv = model.direct_to_explicit_inverse(params)
        _, u_rec = model.inverse_call(params, carry, y, e_inv)
        np.testing.assert_allclose(u_rec, inputs, atol=1e-3, rtol=1e-3)

    n = 300
    ka, kb = jax.random.split(key)
    ua = jax.random.normal(ka, (n, io))
    ub = jax.random.normal(kb, (n, io))
    z = {"rens": [jnp.zeros((n, nx)) for _ in range(L)],
         "dyn_in": None, "dyn_out": None}
    _, ya = model.explicit_call(params, z, ua, e)
    _, yb = model.explicit_call(params, z, ub, e)
    gain = jnp.linalg.norm(ya - yb, axis=1) / jnp.linalg.norm(ua - ub, axis=1)
    assert gain.min() >= mu - 1e-3
    assert gain.max() <= nu + 1e-3
    assert model.get_bounds(params) == (mu, nu)

    def loss(p):
        return jnp.sum(model.apply(p, carry, inputs)[1] ** 2)
    g = jax.grad(loss)(params)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


@pytest.mark.parametrize("at_input,at_output",
                         [(True, False), (False, True), (True, True)])
def test_composition_ren_dyn_orth_noncausal_inverse(key, at_input, at_output):
    io, nx, L = 2, 3, 2
    model = _model(io=io, nx=nx, nv=6, L=L, dyn_orth_at_input=at_input,
                   dyn_orth_at_output=at_output, dyn_state_multiplier=4)
    params, carry, inputs = _setup(model, key, 5, io)
    assert (carry["dyn_in"] is not None) == at_input
    assert (carry["dyn_out"] is not None) == at_output

    new_carry, y = model.apply(params, carry, inputs)
    assert y.shape == (5, io)
    if at_input:
        assert new_carry["dyn_in"].shape == (5, 4 * nx)

    e_inv = model.direct_to_explicit_inverse(params)
    _, u_rec = model.inverse_call_noncausal(params, carry, new_carry, y, e_inv)
    np.testing.assert_allclose(u_rec, inputs, atol=1e-3, rtol=1e-3)


def test_composition_ren_ignore_inverse_with_dyn_in(key):
    model = _model(dyn_orth_at_input=True, dyn_state_multiplier=4)
    params, carry, inputs = _setup(model, key, 5, 2)
    _, y = model.apply(params, carry, inputs)
    e_inv = model.direct_to_explicit_inverse(params)
    _, sig_rec = model.inverse_call(params, carry, y, e_inv)
    ed = e_inv.dyn_in
    expected = carry["dyn_in"] @ ed.C.T + inputs @ ed.D.T + ed.by
    np.testing.assert_allclose(sig_rec, expected, atol=1e-3, rtol=1e-3)


def test_composition_ren_ignore_inverse_rejects_dyn_out(key):
    model = _model(dyn_orth_at_output=True, dyn_state_multiplier=4)
    params, carry, inputs = _setup(model, key, 3, 2)
    _, y = model.apply(params, carry, inputs)
    e_inv = model.direct_to_explicit_inverse(params)
    with pytest.raises(ValueError):
        model.inverse_call(params, carry, y, e_inv)


@pytest.mark.parametrize("kw", [dict(L=0), dict(mu=2.0, nu=1.0), dict(mu=1.0, nu=1.0)])
def test_composition_ren_invalid_args(key, kw):
    with pytest.raises(ValueError):
        model = _model(**kw)
        _setup(model, key, 2, 2)
