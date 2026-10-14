# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `LBDN`."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import golden
from jax_utils import check_golden
from robustnn.lbdn_jax import LBDN


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


@pytest.mark.parametrize("case", golden.cases("lbdn"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_lbdn_matches_golden(case, explicit):
    check_golden("lbdn", case, explicit)
