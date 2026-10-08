# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `MonLipNet` and `BiLipNet`."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from robustnn.bilipnet_jax import BiLipNet
from robustnn.monlipnet_jax import MonLipNet


def _gains(f, key, n, size):
    ka, kb = jax.random.split(key)
    ua = jax.random.normal(ka, (n, size))
    ub = jax.random.normal(kb, (n, size))
    return jnp.linalg.norm(f(ua) - f(ub), axis=1) / jnp.linalg.norm(ua - ub, axis=1)


# ----------------------------- MonLipNet ---------------------------------
@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("units", [[3], [4, 4], [3, 3, 3]])
@pytest.mark.parametrize("fixed", ["mu_nu", "mu_tau", "nu_tau"])
def test_monlipnet_shape_explicit_and_bounds(key, batches, units, fixed):
    size, mu, nu, tau = 3, 0.5, 4.0, 4.0
    flags = dict(is_mu_fixed="mu" in fixed, is_nu_fixed="nu" in fixed,
                 is_tau_fixed="tau" in fixed)
    net = MonLipNet(input_size=size, units=units, mu=mu, nu=nu, tau=tau, **flags)
    x = jax.random.normal(key, (batches, size))
    params = net.init(key, x)

    y = net.apply(params, x)
    assert y.shape == (batches, size)
    explicit = net.direct_to_explicit(params)
    np.testing.assert_allclose(net.explicit_call(params, x, explicit), y, atol=1e-5)

    lo, hi, _ = net.get_bounds(params)
    gains = _gains(lambda u: net.apply(params, u), key, 200, size)
    assert gains.min() >= float(lo) - 1e-3
    assert gains.max() <= float(hi) + 1e-3


def test_monlipnet_fixing_all_raises(key):
    net = MonLipNet(input_size=2, units=[2], is_mu_fixed=True,
                    is_nu_fixed=True, is_tau_fixed=True)
    with pytest.raises(ValueError):
        net.init(key, jnp.ones((1, 2)))


@pytest.mark.parametrize("units", [[3], [4, 4]])
def test_monlipnet_inverse_round_trip(key, units):
    net = MonLipNet(input_size=3, units=units, mu=0.5, nu=4.0, tau=4.0,
                    is_tau_fixed=True)
    x = jax.random.normal(key, (6, 3))
    params = net.init(key, x)
    y = net.apply(params, x)
    e_inv = net.direct_to_explicit_inverse(params, alpha=0.1, iterations=500)
    x_rec = net.inverse_call(params, y, e_inv)
    np.testing.assert_allclose(x_rec, x, atol=1e-3)


def test_monlipnet_jit_and_grad(key):
    net = MonLipNet(input_size=3, units=[4], mu=0.5, nu=4.0, tau=4.0,
                    is_tau_fixed=True)
    x = jax.random.normal(key, (5, 3))
    params = net.init(key, x)
    np.testing.assert_allclose(jax.jit(net.apply)(params, x),
                               net.apply(params, x), atol=1e-5)
    g = jax.grad(lambda p: jnp.sum(net.apply(p, x) ** 2))(params)
    flat = ravel_pytree(g)[0]
    assert bool(jnp.isfinite(flat).all()) and bool(jnp.any(flat != 0))


# ----------------------------- BiLipNet ----------------------------------
@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("use_bias", [True, False])
def test_bilipnet_shape_explicit_and_bounds(key, batches, depth, use_bias):
    size, mu, nu = 3, 0.5, 4.0
    net = BiLipNet(input_size=size, units=[4, 4], mu=mu, nu=nu, tau=nu / mu,
                   is_mu_fixed=True, is_nu_fixed=True, depth=depth,
                   use_bias=use_bias)
    x = jax.random.normal(key, (batches, size))
    params = net.init(key, x)

    y = net.apply(params, x)
    assert y.shape == (batches, size)
    explicit = net.direct_to_explicit(params)
    np.testing.assert_allclose(net.explicit_call(params, x, explicit), y, atol=1e-5)

    lo, hi, _ = net.get_bounds(params)
    np.testing.assert_allclose([lo, hi], [mu, nu], rtol=1e-5)
    gains = _gains(lambda u: net.apply(params, u), key, 200, size)
    assert gains.min() >= mu - 1e-3
    assert gains.max() <= nu + 1e-3


@pytest.mark.parametrize("depth", [1, 2])
def test_bilipnet_inverse_round_trip(key, depth):
    net = BiLipNet(input_size=3, units=[4], mu=0.5, nu=4.0, tau=8.0,
                   is_mu_fixed=True, is_nu_fixed=True, depth=depth, act_fn=nn.relu)
    x = jax.random.normal(key, (6, 3))
    params = net.init(key, x)
    y = net.apply(params, x)
    inv_act = nn.relu
    e_inv = net.direct_to_explicit_inverse(
        params, [0.1] * depth, [inv_act] * depth, [500] * depth, [1.0] * depth)
    x_rec = net.inverse_call(params, y, e_inv)
    np.testing.assert_allclose(x_rec, x, atol=1e-3)


def test_bilipnet_jit_and_grad(key):
    net = BiLipNet(input_size=3, units=[4], mu=0.5, nu=4.0, tau=8.0,
                   is_mu_fixed=True, is_nu_fixed=True)
    x = jax.random.normal(key, (5, 3))
    params = net.init(key, x)
    np.testing.assert_allclose(jax.jit(net.apply)(params, x),
                               net.apply(params, x), atol=1e-5)
    g = jax.grad(lambda p: jnp.sum(net.apply(p, x) ** 2))(params)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


def test_bilipnet_fixing_all_raises(key):
    net = BiLipNet(input_size=2, units=[2], is_mu_fixed=True,
                   is_nu_fixed=True, is_tau_fixed=True)
    with pytest.raises(ValueError):
        net.init(key, jnp.ones((1, 2)))
