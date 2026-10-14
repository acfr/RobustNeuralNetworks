# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `PLNet`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import golden
from jax_utils import check_golden
from robustnn.bilipnet_jax import BiLipNet
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


@pytest.mark.parametrize("case", golden.cases("plnet"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_plnet_matches_golden(case, explicit):
    check_golden("plnet", case, explicit)
