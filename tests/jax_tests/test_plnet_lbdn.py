# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `PLNet` and `LBDN`."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from robustnn.bilipnet_jax import BiLipNet
from robustnn.lbdn_jax import LBDN
from robustnn.plnet_jax import PLNet


def _plnet(size=3, **kw):
    bilip = BiLipNet(input_size=size, units=[4], mu=0.5, nu=4.0, tau=8.0,
                     is_mu_fixed=True, is_nu_fixed=True, depth=2)
    return PLNet(BiLipBlock=bilip, **kw)


# ------------------------------- PLNet -----------------------------------
@pytest.mark.parametrize("batches", [1, 6])
def test_plnet_shape_and_explicit(key, batches):
    net = _plnet()
    x = jax.random.normal(key, (batches, 3))
    params = net.init(key, x)
    y = net.apply(params, x)
    assert y.shape == (batches,)
    assert bool(jnp.all(y >= 0))

    explicit = net.direct_to_explicit(params)
    np.testing.assert_allclose(net.explicit_call(params, x, explicit), y, atol=1e-5)


def test_plnet_zero_at_optimal_point(key):
    net = _plnet()
    x = jax.random.normal(key, (4, 3))
    params = net.init(key, x)
    explicit = net.direct_to_explicit(params, x_optimal=x[:1])
    y_opt = net.explicit_call(params, x[:1], explicit)
    np.testing.assert_allclose(y_opt, 0.0, atol=1e-6)
    assert bool(jnp.all(net.explicit_call(params, x, explicit) >= 0))


def test_plnet_add_constant(key):
    net = _plnet(add_constant=True, c=1.5)
    x = jax.random.normal(key, (4, 3))
    params = net.init(key, x)
    explicit = net.direct_to_explicit(params)
    np.testing.assert_allclose(net.explicit_call(params, x, explicit),
                               net.apply(params, x), atol=1e-5)


def test_plnet_grad(key):
    net = _plnet()
    x = jax.random.normal(key, (4, 3))
    params = net.init(key, x)
    g = jax.grad(lambda p: jnp.sum(net.apply(p, x)))(params)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


# -------------------------------- LBDN -----------------------------------
@pytest.mark.parametrize("activation", [nn.relu, nn.tanh, nn.sigmoid, lambda x: x])
@pytest.mark.parametrize("hidden", [(4,), (8, 16)])
@pytest.mark.parametrize("batches", [1, 5])
def test_lbdn_shape_explicit_and_lipschitz(key, activation, hidden, batches):
    nin, nout, gamma = 5, 2, 3.0
    net = LBDN(nin, hidden, nout, gamma=gamma, activation=activation)
    x = jax.random.normal(key, (batches, nin))
    params = net.init(key, x)

    y = net.apply(params, x)
    assert y.shape == (batches, nout)
    explicit = net.direct_to_explicit(params)
    np.testing.assert_allclose(net.explicit_call(params, x, explicit), y, atol=1e-5)

    ka, kb = jax.random.split(key)
    ua = jax.random.normal(ka, (200, nin))
    ub = jax.random.normal(kb, (200, nin))
    gain = (jnp.linalg.norm(net.apply(params, ua) - net.apply(params, ub), axis=1)
            / jnp.linalg.norm(ua - ub, axis=1))
    assert gain.max() <= gamma + 1e-3


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("trainable_lipschitz", [True, False])
def test_lbdn_options_and_grad(key, use_bias, trainable_lipschitz):
    net = LBDN(3, (6,), 2, gamma=2.0, use_bias=use_bias,
               trainable_lipschitz=trainable_lipschitz)
    x = jax.random.normal(key, (4, 3))
    params = net.init(key, x)
    np.testing.assert_allclose(jax.jit(net.apply)(params, x),
                               net.apply(params, x), atol=1e-5)
    g = jax.grad(lambda p: jnp.sum(net.apply(p, x) ** 2))(params)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


def test_lbdn_init_output_zero(key):
    net = LBDN(3, (6,), 2, init_output_zero=True)
    x = jax.random.normal(key, (4, 3))
    params = net.init(key, x)
    np.testing.assert_allclose(net.apply(params, x), 0.0, atol=1e-6)
