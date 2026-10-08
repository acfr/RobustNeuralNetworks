# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX RENs: Contracting, Lipschitz, General (QSR) and
BiLipschitz, plus the linear RENs and the contracting R2DN."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from helpers import compute_p_contractingr2dn, compute_p_contractingren
from robustnn import linear_ren
from robustnn import r2dn_jax as r2dn
from robustnn import ren_jax as ren

ACTIVATIONS = [nn.relu, nn.tanh, lambda x: x]


def _setup(model, key, batches, nin, x_scale=1.0):
    k1, k2, k3 = jax.random.split(key, 3)
    states = model.initialize_carry(k1, (batches, nin))
    states = x_scale * jax.random.normal(k2, states.shape)
    inputs = jax.random.normal(k3, (batches, nin))
    params = model.init(key, states, inputs)
    return params, states, inputs


def _qsr(key, nu, ny):
    kx, ky, ks = jax.random.split(key, 3)
    X = jax.random.normal(kx, (ny, ny))
    Y = jax.random.normal(ky, (nu, nu))
    S = jax.random.normal(ks, (nu, ny))
    Q = -X.T @ X
    R = S @ jnp.linalg.solve(Q, S.T) + Y.T @ Y
    return Q, S, R


def _check_forward(model, params, states, inputs, nx, ny):
    batches = inputs.shape[0]
    explicit = model.direct_to_explicit(params)
    xn, y = model.explicit_call(params, states, inputs, explicit)
    assert xn.shape == (batches, nx)
    assert y.shape == (batches, ny)
    xn2, y2 = model.apply(params, states, inputs)
    np.testing.assert_allclose(xn, xn2, atol=1e-5)
    np.testing.assert_allclose(y, y2, atol=1e-5)

    # jit gives the same answer, and gradients w.r.t. everything are finite
    jit_xn, jit_y = jax.jit(model.apply)(params, states, inputs)
    np.testing.assert_allclose(y, jit_y, atol=1e-4)

    def loss(p, x, u):
        a, b = model.apply(p, x, u)
        return jnp.sum(a ** 2) + jnp.sum(b ** 2)
    g = jax.grad(loss, argnums=(0, 1, 2))(params, states, inputs)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


# --------------------------- Contracting REN -----------------------------
@pytest.mark.parametrize("batches", [1, 4])
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("init_method", ["random", "long_memory"])
@pytest.mark.parametrize("do_polar_param", [True, False])
@pytest.mark.parametrize("sizes", [(1, 1, 1, 1), (5, 3, 4, 2)])
def test_contracting_ren(key, batches, activation, init_method, do_polar_param, sizes):
    nu, nx, nv, ny = sizes
    model = ren.ContractingREN(nu, nx, nv, ny, activation=activation,
                               init_method=init_method, do_polar_param=do_polar_param)
    params, states, inputs = _setup(model, key, batches, nu)
    _check_forward(model, params, states, inputs, nx, ny)


def test_contracting_ren_contraction(key):
    model = ren.ContractingREN(5, 3, 4, 2, activation=nn.tanh)
    k1, k2 = jax.random.split(key)
    params, x1, u = _setup(model, key, 8, 5)
    x0 = 10 * jax.random.normal(k1, x1.shape)
    xn0, _ = model.apply(params, x0, u)
    xn1, _ = model.apply(params, x1, u)
    P = compute_p_contractingren(model, params)

    def norm2(x):
        return jnp.sum((x @ P.T) * x, axis=-1)
    assert bool(jnp.all(norm2(xn0 - xn1) - norm2(x0 - x1) <= 1e-4))


def test_ren_identity_output_and_zero_output(key):
    model = ren.ContractingREN(3, 3, 4, 3, identity_output=True)
    params, states, inputs = _setup(model, key, 4, 3)
    _, y = model.apply(params, states, inputs)
    # identity output: y_t = x_t, i.e. the state before the update
    np.testing.assert_allclose(y, states, atol=1e-5)

    zero = ren.ContractingREN(3, 3, 4, 2, init_output_zero=True)
    params, states, inputs = _setup(zero, key, 4, 3)
    _, y = zero.apply(params, states, inputs)
    np.testing.assert_allclose(y, 0.0, atol=1e-6)


def test_ren_identity_output_size_mismatch(key):
    model = ren.ContractingREN(3, 3, 4, 2, identity_output=True)
    with pytest.raises(ValueError):
        _setup(model, key, 2, 3)


def test_ren_bad_init_method(key):
    model = ren.ContractingREN(3, 3, 4, 2, init_method="nope")
    with pytest.raises(ValueError):
        _setup(model, key, 2, 3)


# ----------------------------- Lipschitz REN -----------------------------
@pytest.mark.parametrize("gamma", [0.5, 1.0, 5.0])
@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_lipschitz_ren(key, gamma, activation):
    nu, nx, nv, ny = 4, 3, 5, 2
    model = ren.LipschitzREN(nu, nx, nv, ny, gamma=gamma, activation=activation)
    params, states, inputs = _setup(model, key, 4, nu)
    _check_forward(model, params, states, inputs, nx, ny)

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
        _setup(model, key, 2, 3)


# ------------------------------ General REN ------------------------------
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("sizes", [(5, 3, 4, 2), (2, 2, 3, 2)])
def test_general_ren(key, activation, sizes):
    nu, nx, nv, ny = sizes
    Q, S, R = _qsr(key, nu, ny)
    model = ren.GeneralREN(nu, nx, nv, ny, Q=Q, S=S, R=R, activation=activation,
                           init_method="long_memory")
    model.check_valid_qsr()
    params, states, inputs = _setup(model, key, 4, nu)
    _check_forward(model, params, states, inputs, nx, ny)


def test_general_ren_invalid_qsr(key):
    nu, ny = 3, 2
    Q, S, R = _qsr(key, nu, ny)
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
    params, states, inputs = _setup(model, key, batches, nio)
    _check_forward(model, params, states, inputs, nx, nio)

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
        _setup(model, key, 2, 3)


# ------------------------------ simulate_sequence -------------------------
@pytest.mark.parametrize("horizon", [1, 5])
def test_simulate_sequence_matches_loop(key, horizon):
    nu, nx, nv, ny = 3, 4, 5, 2
    model = ren.ContractingREN(nu, nx, nv, ny, activation=nn.tanh)
    params, x0, u0 = _setup(model, key, 4, nu)
    seq = jax.random.normal(key, (horizon, 4, nu))

    xT, ys = model.simulate_sequence(params, x0, seq)
    assert ys.shape == (horizon, 4, ny)

    x, outs = x0, []
    for t in range(horizon):
        x, y = model.apply(params, x, seq[t])
        outs.append(y)
    np.testing.assert_allclose(xT, x, atol=1e-4)
    np.testing.assert_allclose(ys, jnp.stack(outs), atol=1e-4)


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
        Q, S, R = _qsr(key, nu, ny)
        model = linear_ren.GeneralLinREN(nu, nx, 0, ny, Q=Q, S=S, R=R,
                                         init_method="long_memory")
        model.check_valid_qsr()
    params, states, inputs = _setup(model, key, batches, nu)
    _check_forward(model, params, states, inputs, nx, ny)


def test_linear_ren_lipschitz_bound(key):
    nu, nx, ny, gamma = 4, 3, 2, 2.0
    model = linear_ren.LipschitzLinREN(nu, nx, 0, ny, gamma=gamma)
    params, _, _ = _setup(model, key, 2, nu)
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
        _setup(model, key, 2, 3)


# ----------------------------------- R2DN ---------------------------------
@pytest.mark.parametrize("batches", [1, 4])
@pytest.mark.parametrize("hidden", [(2,), (2, 2), (4, 3, 2)])
@pytest.mark.parametrize("init_method", ["random", "long_memory"])
def test_contracting_r2dn(key, batches, hidden, init_method):
    nu, nx, nv, ny = 5, 3, 4, 2
    model = r2dn.ContractingR2DN(nu, nx, nv, ny, hidden, init_method=init_method)
    params, states, inputs = _setup(model, key, batches, nu)
    _check_forward(model, params, states, inputs, nx, ny)


def test_r2dn_simulate_and_contraction(key):
    nu, nx, nv, ny, horizon = 5, 3, 4, 2, 3
    model = r2dn.ContractingR2DN(nu, nx, nv, ny, (2, 2), init_method="long_memory")
    params, x1, u = _setup(model, key, 4, nu)

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
